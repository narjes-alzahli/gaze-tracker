"""
Gaze-based desktop application using OpenCV, MediaPipe Face Mesh, and PyAutoGUI.

Sections:
  - Webcam capture
  - Face mesh detection
  - Gaze estimation (iris -> screen coordinates)
  - Cursor visualization
  - Button UI and highlight logic
  - Optional 9-point calibration
"""

import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import json
import os
import time

import config

# Disable PyAutoGUI fail-safe for this app (we only use it for screen size)
pyautogui.FAILSAFE = False

# Exit keys: ESC or 'q'
ESC_KEY = 27


def _is_exit_key(key):
    """True if key is ESC or 'q'."""
    return key == ESC_KEY or key == ord("q")


def _set_window_on_top(winname):
    """Keep window on top so it pops up in front, not in background."""
    try:
        cv2.setWindowProperty(winname, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass


def _set_fullscreen(winname):
    """Make window fullscreen."""
    try:
        cv2.setWindowProperty(winname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except Exception:
        pass

# -----------------------------------------------------------------------------
# Webcam capture
# -----------------------------------------------------------------------------


class Camera:
    """Captures live video from the default webcam using OpenCV."""

    def __init__(self, index=config.CAMERA_INDEX, width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT):
        self.index = index
        self.width = width
        self.height = height
        self._cap = None

    def start(self):
        """Open the camera and set resolution."""
        self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            raise RuntimeError("Could not open webcam.")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return self

    def read(self):
        """Return (success, frame). Frame is BGR."""
        if self._cap is None:
            return False, None
        return self._cap.read()

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


# -----------------------------------------------------------------------------
# Face mesh detection
# -----------------------------------------------------------------------------


class FaceMeshDetector:
    """Detects face and iris landmarks using MediaPipe Face Mesh (with iris refinement)."""

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Required for iris landmarks (468-477)
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_rgb):
        """
        Process an RGB frame. Returns list of face landmark objects, or None.
        Each landmark has .x, .y, .z (normalized 0-1).
        """
        results = self.face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0].landmark

    def close(self):
        self.face_mesh.close()


# -----------------------------------------------------------------------------
# Gaze estimation
# -----------------------------------------------------------------------------


def _normalized_to_pixel(landmark, width, height):
    """Convert MediaPipe normalized (x,y) to pixel (x,y)."""
    return (
        int(landmark.x * width),
        int(landmark.y * height),
    )


def _gaze_from_eye(landmarks, eye_indices, iris_index, width, height):
    """
    Compute normalized gaze offset for one eye from iris position relative to eye corners.
    Returns (dx, dy) in range roughly [-1, 1] (left/up negative, right/down positive).
    """
    inner = landmarks[eye_indices[1]]
    outer = landmarks[eye_indices[0]]
    iris = landmarks[iris_index]

    # Eye vector and center
    ex = (outer.x + inner.x) / 2
    ey = (outer.y + inner.y) / 2
    # Horizontal and vertical extent (scale by eye width for aspect)
    eye_w = abs(inner.x - outer.x)
    eye_h = eye_w * 0.6  # Approximate eye aspect
    if eye_w < 1e-6:
        return 0.0, 0.0

    dx = (iris.x - ex) / eye_w
    dy = (iris.y - ey) / eye_h
    return dx, dy


def estimate_gaze_normalized(landmarks, width, height):
    """
    Estimate gaze direction from face landmarks (with iris).
    Returns (gaze_x, gaze_y) normalized in [0, 1] for screen mapping,
    or None if landmarks are invalid.
    """
    if landmarks is None or len(landmarks) <= config.RIGHT_IRIS_INDEX:
        return None

    left_dx, left_dy = _gaze_from_eye(
        landmarks, config.LEFT_EYE_INDICES, config.LEFT_IRIS_INDEX, width, height
    )
    right_dx, right_dy = _gaze_from_eye(
        landmarks, config.RIGHT_EYE_INDICES, config.RIGHT_IRIS_INDEX, width, height
    )
    # Average both eyes; dx, dy are in ~[-1,1]
    dx = (left_dx + right_dx) / 2
    dy = (left_dy + right_dy) / 2
    # Per-axis sensitivity (Y can differ so cursor doesn't move diagonally)
    sens_x = getattr(config, "GAZE_SENSITIVITY", 2.0)
    sens_y = getattr(config, "GAZE_SENSITIVITY_Y", None)
    if sens_y is None:
        sens_y = sens_x
    bias_x = getattr(config, "GAZE_CENTER_BIAS_X", 0.0)
    bias_y = getattr(config, "GAZE_CENTER_BIAS_Y", 0.0)
    gaze_x = np.clip(0.5 + dx * sens_x + bias_x, 0, 1)
    gaze_y = np.clip(0.5 + dy * sens_y + bias_y, 0, 1)
    # When camera is mirrored, flip horizontal so "look left" = cursor left
    if getattr(config, "MIRROR_CAMERA", False) and getattr(config, "GAZE_MIRROR_CORRECT_X", True):
        gaze_x = 1.0 - gaze_x
    if getattr(config, "GAZE_SWAP_XY", False):
        gaze_x, gaze_y = gaze_y, gaze_x
    return gaze_x, gaze_y


def gaze_to_screen_no_calibration(gaze_norm_x, gaze_norm_y, screen_width, screen_height):
    """Map normalized gaze [0,1] to screen coordinates (no calibration)."""
    x = int(gaze_norm_x * screen_width)
    y = int(gaze_norm_y * screen_height)
    return x, y


# -----------------------------------------------------------------------------
# Gaze smoothing
# -----------------------------------------------------------------------------


class GazeSmoother:
    """Exponential moving average smoothing for gaze (x, y) to reduce jitter."""

    def __init__(self, alpha=config.SMOOTHING_ALPHA):
        self.alpha = alpha
        self._x = None
        self._y = None

    def update(self, x, y):
        if self._x is None:
            self._x, self._y = float(x), float(y)
            return self._x, self._y
        self._x = self.alpha * x + (1 - self.alpha) * self._x
        self._y = self.alpha * y + (1 - self.alpha) * self._y
        return self._x, self._y

    def reset(self):
        self._x = self._y = None


# -----------------------------------------------------------------------------
# Calibration (9-point)
# -----------------------------------------------------------------------------


def get_calibration_screen_points(screen_width, screen_height, n=9):
    """
    Return n screen points in a 3x3 grid that covers the full screen including corners.
    Uses CALIBRATION_GRID_MARGIN so outer points are near the edges (e.g. 5% and 95%),
    not inset (old grid was 17% and 83%), so the affine isn't extrapolating at corners.
    """
    pts = []
    margin = getattr(config, "CALIBRATION_GRID_MARGIN", 0.05)  # 0.05 = 5% from edge
    if n == 9:
        # Normalized positions: (margin, margin), (0.5, margin), (1-margin, margin), ...
        for row in range(3):
            for col in range(3):
                if col == 0:
                    fx = margin
                elif col == 1:
                    fx = 0.5
                else:
                    fx = 1.0 - margin
                if row == 0:
                    fy = margin
                elif row == 1:
                    fy = 0.5
                else:
                    fy = 1.0 - margin
                x = int(fx * screen_width)
                y = int(fy * screen_height)
                pts.append((x, y))
    return pts


# Index of the center point in the 3x3 calibration grid (0-based)
CALIBRATION_CENTER_POINT_INDEX = 4


class CalibrationMapper:
    """
    Maps normalized gaze to screen using 9-point calibration.
    Uses a "center-neutral" step: the average gaze when you looked at the center dot
    is defined as (0.5, 0.5), so the cursor lines up at center when you look at center.
    Then a robust affine fit maps (gaze - neutral + 0.5) -> screen.
    """

    def __init__(self):
        self._gaze_points = []  # list of [gaze_x, gaze_y]
        self._screen_points = []  # list of [screen_x, screen_y]
        self._matrix = None  # 2x3 affine or None
        self._neutral = None  # (nx, ny) average gaze when looking at center; used so center stays at center

    def add_sample(self, gaze_x, gaze_y, screen_x, screen_y):
        self._gaze_points.append([gaze_x, gaze_y])
        self._screen_points.append([screen_x, screen_y])

    def is_complete(self, required=config.CALIBRATION_POINTS):
        return len(self._gaze_points) >= required

    def compute(self):
        """
        Compute neutral from center-point samples, then fit affine on center-corrected gaze.
        This makes "looking at center" map to screen center even if the raw gaze model is biased.
        """
        n_points = config.CALIBRATION_POINTS
        samples_per_point = len(self._gaze_points) // n_points if n_points else 0
        if samples_per_point < 1 or len(self._gaze_points) < 4:
            self._matrix = None
            self._neutral = None
            return False

        # Center point is index 4 in 3x3 grid; its samples are contiguous
        center_start = CALIBRATION_CENTER_POINT_INDEX * samples_per_point
        center_end = center_start + samples_per_point
        center_gaze = np.array(self._gaze_points[center_start:center_end], dtype=np.float32)
        self._neutral = (float(np.mean(center_gaze[:, 0])), float(np.mean(center_gaze[:, 1])))

        # Center-corrected gaze: so that neutral -> (0.5, 0.5)
        src = np.array(self._gaze_points, dtype=np.float32)
        nx, ny = self._neutral
        src_corrected = np.empty_like(src)
        src_corrected[:, 0] = src[:, 0] - nx + 0.5
        src_corrected[:, 1] = src[:, 1] - ny + 0.5
        dst = np.array(self._screen_points, dtype=np.float32)

        # Robust fit (LMEDS) so one bad sample doesn't pull the transform
        try:
            self._matrix, _ = cv2.estimateAffine2D(src_corrected, dst, method=cv2.LMEDS)
        except Exception:
            self._matrix, _ = cv2.estimateAffine2D(src_corrected, dst)
        if self._matrix is None:
            self._neutral = None
            return False
        return True

    def map_to_screen(self, gaze_x, gaze_y):
        """Map normalized gaze to (screen_x, screen_y). Uses neutral so center gaze -> screen center."""
        if self._matrix is None:
            w, h = pyautogui.size()
            return int(gaze_x * w), int(gaze_y * h)
        # Apply center correction: raw gaze -> center-neutralized (so neutral -> 0.5, 0.5)
        if self._neutral is not None:
            nx, ny = self._neutral
            gaze_x = gaze_x - nx + 0.5
            gaze_y = gaze_y - ny + 0.5
        pt = np.array([gaze_x, gaze_y, 1.0], dtype=np.float32)
        out = self._matrix @ pt
        return int(out[0]), int(out[1])

    def save(self, path="calibration.json"):
        data = {
            "gaze_points": self._gaze_points,
            "screen_points": self._screen_points,
            "matrix": self._matrix.tolist() if self._matrix is not None else None,
            "neutral": list(self._neutral) if self._neutral is not None else None,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path="calibration.json"):
        if not os.path.isfile(path):
            return False
        with open(path) as f:
            data = json.load(f)
        self._gaze_points = data.get("gaze_points", [])
        self._screen_points = data.get("screen_points", [])
        m = data.get("matrix")
        if m is not None:
            self._matrix = np.array(m, dtype=np.float32)
            if self._matrix.shape != (2, 3):
                self._matrix = None
        else:
            self._matrix = None
        n = data.get("neutral")
        self._neutral = tuple(n) if n and len(n) == 2 else None
        return True

    def reset(self):
        self._gaze_points = []
        self._screen_points = []
        self._matrix = None
        self._neutral = None


# -----------------------------------------------------------------------------
# Cursor visualization
# -----------------------------------------------------------------------------


def draw_gaze_cursor(frame, gaze_screen_x, gaze_screen_y, frame_width, frame_height,
                     screen_width, screen_height):
    """
    Draw a clear, nice-looking cursor at gaze position: outer glow, ring, fill, inner dot.
    """
    fx = int(gaze_screen_x / screen_width * frame_width)
    fy = int(gaze_screen_y / screen_height * frame_height)
    fx = np.clip(fx, 0, frame_width - 1)
    fy = np.clip(fy, 0, frame_height - 1)
    center = (fx, fy)
    # Scale radii to frame size if we're drawing on a scaled frame
    r_outer = max(4, config.CURSOR_OUTER_RADIUS * frame_width // 640)
    r_inner = max(2, config.CURSOR_INNER_RADIUS * frame_width // 640)
    # Soft outer ring
    cv2.circle(frame, center, r_outer + 4, config.CURSOR_GLOW_COLOR, 4)
    # White border ring
    cv2.circle(frame, center, r_outer, config.CURSOR_RING_COLOR, 2)
    # Filled main circle
    cv2.circle(frame, center, r_outer - 2, config.CURSOR_FILL_COLOR, -1)
    # Bright inner dot
    cv2.circle(frame, center, r_inner, config.CURSOR_RING_COLOR, -1)
    return frame


def draw_face_mesh_overlay(frame, landmarks, width, height):
    """Draw face mesh (landmarks + connections) and highlight iris/eye key points."""
    if landmarks is None:
        return frame
    # Draw all face landmarks as small dots for mesh effect (subset for performance)
    for i, lm in enumerate(landmarks):
        if i >= 468:  # Iris indices 468+
            continue
        pt = _normalized_to_pixel(lm, width, height)
        cv2.circle(frame, pt, 1, (180, 180, 180), -1)
    # Draw iris and eye corners in distinct colors
    for idx in [config.LEFT_IRIS_INDEX, config.RIGHT_IRIS_INDEX] + config.LEFT_EYE_INDICES + config.RIGHT_EYE_INDICES:
        if idx < len(landmarks):
            pt = _normalized_to_pixel(landmarks[idx], width, height)
            color = (0, 255, 255) if idx in (config.LEFT_IRIS_INDEX, config.RIGHT_IRIS_INDEX) else (255, 200, 0)
            cv2.circle(frame, pt, 4, color, -1)
    return frame


# -----------------------------------------------------------------------------
# Button UI and highlight logic
# -----------------------------------------------------------------------------


def build_button_rectangles(frame_width, frame_height):
    """
    Build list of (label, rect) where rect is (x, y, w, h) in frame coordinates.
    Buttons are drawn in a grid at the bottom of the frame.
    """
    n = len(config.BUTTON_LABELS)
    cols = config.BUTTON_GRID_COLS
    rows = config.BUTTON_GRID_ROWS
    pad_frac = config.BUTTON_PADDING_FRACTION
    # Use bottom portion of frame for buttons
    ui_height = int(frame_height * 0.35)
    ui_y0 = frame_height - ui_height
    cell_w = frame_width // cols
    cell_h = ui_height // rows
    padding_w = int(cell_w * pad_frac)
    padding_h = int(cell_h * pad_frac)
    rects = []
    for i, label in enumerate(config.BUTTON_LABELS):
        row = i // cols
        col = i % cols
        x = col * cell_w + padding_w
        y = ui_y0 + row * cell_h + padding_h
        w = cell_w - 2 * padding_w
        h = cell_h - 2 * padding_h
        rects.append((label, (x, y, w, h)))
    return rects


def screen_to_frame_point(screen_x, screen_y, frame_width, frame_height, screen_width, screen_height):
    """Map screen gaze position to frame coordinates (for hit-testing buttons drawn on frame)."""
    # Gaze is reported in screen space; we draw buttons on frame. So we need to map
    # screen (sx, sy) to frame (fx, fy). Our UI buttons are in frame coords.
    # We have: gaze_screen = mapping(gaze_norm). So gaze_norm -> screen.
    # Frame overlay: we draw cursor at frame pos = (gaze_norm * frame_width, gaze_norm * frame_height)
    # So for hit-test we should use normalized gaze mapped to frame size.
    fx = screen_x / screen_width * frame_width
    fy = screen_y / screen_height * frame_height
    return int(fx), int(fy)


def hit_test_buttons(screen_x, screen_y, button_rects, frame_width, frame_height, screen_width, screen_height):
    """
    Return the index of the button under (screen_x, screen_y), or None.
    button_rects from build_button_rectangles are in frame coords.
    """
    fx, fy = screen_to_frame_point(screen_x, screen_y, frame_width, frame_height, screen_width, screen_height)
    for i, (_, (x, y, w, h)) in enumerate(button_rects):
        if x <= fx <= x + w and y <= fy <= y + h:
            return i
    return None


def draw_buttons(frame, button_rects, highlighted_index):
    """Draw button rectangles and labels; highlight the one at highlighted_index."""
    for i, (label, (x, y, w, h)) in enumerate(button_rects):
        color = config.BUTTON_COLOR_HIGHLIGHT if i == highlighted_index else config.BUTTON_COLOR_NORMAL
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, config.BUTTON_THICKNESS)
        # Label text
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, config.BUTTON_FONT_SCALE, config.BUTTON_THICKNESS)
        tx = x + (w - tw) // 2
        ty = y + (h + th) // 2
        cv2.putText(
            frame, label, (tx, ty), font, config.BUTTON_FONT_SCALE,
            config.BUTTON_TEXT_COLOR, config.BUTTON_THICKNESS, cv2.LINE_AA
        )
    return frame


# -----------------------------------------------------------------------------
# Optional 9-point calibration flow (separate window with dots)
# -----------------------------------------------------------------------------

CALIBRATION_WINDOW_NAME = "Calibration - look at the dot"
# Calibration target: dark red, rings (easy to see and focus on)
CALIBRATION_DOT_RADIUS = 56
CALIBRATION_RING_COLOR = (80, 80, 255)     # BGR light red border
CALIBRATION_FILL_COLOR = (0, 0, 120)      # BGR dark red
CALIBRATION_GLOW_COLOR = (0, 0, 80)       # BGR darker red outer
CALIBRATION_BG_COLOR = (32, 32, 32)        # BGR dark


def _draw_calibration_screen(screen_width, screen_height, sx, sy, point_label, countdown_sec=None):
    """
    Fullscreen calibration frame: dark background, one clear target at (sx, sy)
    with outer glow, ring, and bright center (same visual language as cursor).
    """
    img = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    img[:] = CALIBRATION_BG_COLOR
    center = (sx, sy)
    r = CALIBRATION_DOT_RADIUS
    # Outer soft ring
    cv2.circle(img, center, r + 14, CALIBRATION_GLOW_COLOR, 6)
    # White border
    cv2.circle(img, center, r + 4, CALIBRATION_RING_COLOR, 3)
    # Filled circle (cyan)
    cv2.circle(img, center, r, CALIBRATION_FILL_COLOR, -1)
    cv2.circle(img, center, r, CALIBRATION_RING_COLOR, 2)
    # Bright inner dot
    cv2.circle(img, center, r // 3, CALIBRATION_RING_COLOR, -1)
    # Label at top
    label = f"Point {point_label} — look at the dot"
    cv2.putText(img, label, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    if countdown_sec is not None and countdown_sec > 0:
        cv2.putText(img, f"Hold still... {int(countdown_sec)}s", (50, screen_height - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2, cv2.LINE_AA)
    cv2.putText(img, "ESC or Q = exit calibration", (50, screen_height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1, cv2.LINE_AA)
    return img


def run_calibration(camera, detector, smoother, screen_width, screen_height, frame_width, frame_height):
    """
    Run 9-point calibration using a separate fullscreen-style window.
    Shows one dot at a time at the actual screen position to look at; collects gaze samples.
    Returns a CalibrationMapper with computed transform (or with no transform if skipped/failed).
    """
    mapper = CalibrationMapper()
    points = get_calibration_screen_points(screen_width, screen_height, config.CALIBRATION_POINTS)
    duration = config.CALIBRATION_POINT_DURATION_MS / 1000.0
    samples_per_point = getattr(config, "CALIBRATION_SAMPLES_PER_POINT", 20)

    # Fullscreen calibration window
    cv2.namedWindow(CALIBRATION_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CALIBRATION_WINDOW_NAME, screen_width, screen_height)
    cv2.moveWindow(CALIBRATION_WINDOW_NAME, 0, 0)
    _set_fullscreen(CALIBRATION_WINDOW_NAME)
    _set_window_on_top(CALIBRATION_WINDOW_NAME)

    for p_idx, (sx, sy) in enumerate(points):
        smoother.reset()
        print(f"Look at point {p_idx + 1}/{len(points)} at screen position ({sx}, {sy})")
        collected = 0
        t0 = time.perf_counter()

        while collected < samples_per_point:
            ok, frame = camera.read()
            if ok and frame is not None and config.MIRROR_CAMERA:
                frame = cv2.flip(frame, 1)
            if not ok or frame is None:
                elapsed = time.perf_counter() - t0
                cal_img = _draw_calibration_screen(
                    screen_width, screen_height, sx, sy, f"{p_idx + 1}/{len(points)}",
                    countdown_sec=max(0, duration - elapsed),
                )
                cv2.imshow(CALIBRATION_WINDOW_NAME, cal_img)
                if _is_exit_key(cv2.waitKey(1) & 0xFF):
                    cv2.destroyWindow(CALIBRATION_WINDOW_NAME)
                    return mapper
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = detector.process(frame_rgb)
            gaze_norm = estimate_gaze_normalized(landmarks, frame_width, frame_height)
            if gaze_norm is not None:
                gx, gy = smoother.update(gaze_norm[0], gaze_norm[1])
                mapper.add_sample(gx, gy, sx, sy)
                collected += 1

            elapsed = time.perf_counter() - t0
            cal_img = _draw_calibration_screen(
                screen_width, screen_height, sx, sy, f"{p_idx + 1}/{len(points)}",
                countdown_sec=max(0, duration - elapsed),
            )
            cv2.imshow(CALIBRATION_WINDOW_NAME, cal_img)
            if _is_exit_key(cv2.waitKey(1) & 0xFF):
                cv2.destroyWindow(CALIBRATION_WINDOW_NAME)
                return mapper

        # Hold the dot visible for the full duration
        while time.perf_counter() - t0 < duration:
            cal_img = _draw_calibration_screen(
                screen_width, screen_height, sx, sy, f"{p_idx + 1}/{len(points)}",
                countdown_sec=max(0, duration - (time.perf_counter() - t0)),
            )
            cv2.imshow(CALIBRATION_WINDOW_NAME, cal_img)
            if _is_exit_key(cv2.waitKey(30) & 0xFF):
                cv2.destroyWindow(CALIBRATION_WINDOW_NAME)
                return mapper

    cv2.destroyWindow(CALIBRATION_WINDOW_NAME)
    mapper.compute()
    return mapper


# -----------------------------------------------------------------------------
# Main application loop
# -----------------------------------------------------------------------------


def main():
    # Screen size (for mapping gaze to screen and for calibration)
    screen_width, screen_height = pyautogui.size()

    # Webcam
    camera = Camera()
    camera.start()
    frame_width = int(camera._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(camera._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Face mesh
    detector = FaceMeshDetector()
    smoother = GazeSmoother(alpha=config.SMOOTHING_ALPHA)

    # Calibration: try load, else offer to run
    cal_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")
    cal_mapper = CalibrationMapper()
    print("Run calibration? (y/n in console) — always asked even if calibration.json exists")
    run_cal = input().strip().lower() == "y"
    if run_cal:
        cal_mapper = run_calibration(camera, detector, smoother, screen_width, screen_height, frame_width, frame_height)
        if cal_mapper._matrix is not None:
            cal_mapper.save(cal_path)
            print("Calibration saved.")
    elif cal_mapper.load(cal_path):
        print("Loaded calibration from calibration.json")
    # If they said no and no file (or load failed), cal_mapper uses direct [0,1]->screen mapping

    # Button layout (in frame coords)
    button_rects = build_button_rectangles(frame_width, frame_height)

    window_name = "Gaze UI - Face mesh + gaze cursor (no auto-click)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    _set_fullscreen(window_name)
    _set_window_on_top(window_name)

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                continue
            if config.MIRROR_CAMERA:
                frame = cv2.flip(frame, 1)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = detector.process(frame_rgb)

            # Gaze: normalized -> smooth -> screen
            gaze_norm = estimate_gaze_normalized(landmarks, frame_width, frame_height)
            if gaze_norm is not None:
                smooth_x, smooth_y = smoother.update(gaze_norm[0], gaze_norm[1])
                screen_x, screen_y = cal_mapper.map_to_screen(smooth_x, smooth_y)
                # Optional pixel nudge (e.g. if center is still slightly off after calibration)
                screen_x += getattr(config, "CALIBRATION_OFFSET_PX_X", 0)
                screen_y += getattr(config, "CALIBRATION_OFFSET_PX_Y", 0)
                screen_x = np.clip(screen_x, 0, screen_width - 1)
                screen_y = np.clip(screen_y, 0, screen_height - 1)
            else:
                screen_x = screen_width // 2
                screen_y = screen_height // 2
                smoother.reset()

            # Overlay: face mesh key points
            draw_face_mesh_overlay(frame, landmarks, frame_width, frame_height)

            # Cursor at gaze position (drawn on camera-resolution frame)
            draw_gaze_cursor(frame, screen_x, screen_y, frame_width, frame_height, screen_width, screen_height)

            # Buttons and highlight (no click)
            highlighted = hit_test_buttons(screen_x, screen_y, button_rects, frame_width, frame_height, screen_width, screen_height)
            draw_buttons(frame, button_rects, highlighted)

            # Fullscreen: scale frame to screen size so it fills the window
            frame_display = cv2.resize(frame, (screen_width, screen_height), interpolation=cv2.INTER_LINEAR)
            cv2.imshow(window_name, frame_display)
            key = cv2.waitKey(1) & 0xFF
            if _is_exit_key(key):
                break
            if key == ord("r"):
                smoother.reset()
    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
