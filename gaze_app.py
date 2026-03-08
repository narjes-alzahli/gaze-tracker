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


def estimate_head_pose(landmarks):
    """
    Rough head yaw and pitch from face landmarks (normalized coords).
    Nose tip 1; left eye 33,133; right eye 263,362.
    Returns (head_yaw, head_pitch) in roughly [-1, 1] or (0, 0) if invalid.
    """
    if landmarks is None or len(landmarks) < 400:
        return 0.0, 0.0
    nose = landmarks[1]
    left_cx = (landmarks[33].x + landmarks[133].x) / 2
    left_cy = (landmarks[33].y + landmarks[133].y) / 2
    right_cx = (landmarks[263].x + landmarks[362].x) / 2
    right_cy = (landmarks[263].y + landmarks[362].y) / 2
    mid_x = (left_cx + right_cx) / 2
    mid_y = (left_cy + right_cy) / 2
    # Yaw: nose left of center -> negative (head turned right), nose right -> positive
    yaw = (nose.x - mid_x) * 4  # scale to roughly [-1,1]
    pitch = (nose.y - mid_y) * 4
    return float(np.clip(yaw, -1, 1)), float(np.clip(pitch, -1, 1))


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
# Calibration (16-point + head movement)
# -----------------------------------------------------------------------------


def get_calibration_screen_points(screen_width, screen_height, n=None):
    """
    Return n screen points in a grid that covers the full screen including corners.
    Uses CALIBRATION_GRID_MARGIN so outer points are near the edges.
    n=9 -> 3x3, n=16 -> 4x4 (Gaze Pointer style).
    """
    if n is None:
        n = config.CALIBRATION_POINTS
    pts = []
    margin = getattr(config, "CALIBRATION_GRID_MARGIN", 0.05)
    if n == 9:
        cols, rows = 3, 3
        x_frac = (margin, 0.5, 1.0 - margin)
        y_frac = (margin, 0.5, 1.0 - margin)
    elif n == 16:
        cols, rows = 4, 4
        x_frac = (margin, 1.0/3, 2.0/3, 1.0 - margin)
        y_frac = (margin, 1.0/3, 2.0/3, 1.0 - margin)
    else:
        return pts
    for row in range(rows):
        for col in range(cols):
            fx = x_frac[col]
            fy = y_frac[row]
            x = int(fx * screen_width)
            y = int(fy * screen_height)
            pts.append((x, y))
    return pts


def _calibration_center_point_index():
    n = config.CALIBRATION_POINTS
    if n == 9:
        return 4
    if n == 16:
        return 10  # row 2, col 2 in 4x4
    return n // 2


class CalibrationMapper:
    """
    Maps normalized gaze to screen using N-point calibration (e.g. 16-point).
    Uses a "center-neutral" step: the average gaze when you looked at the center dot
    is defined as (0.5, 0.5). Then a robust affine fit maps (gaze - neutral + 0.5) -> screen.
    Optionally stores head-movement calibration samples for future head-pose correction.
    """

    def __init__(self):
        self._gaze_points = []  # list of [gaze_x, gaze_y]
        self._screen_points = []  # list of [screen_x, screen_y]
        self._matrix = None  # 2x3 affine or None
        self._neutral = None  # (nx, ny) average gaze when looking at center
        self._head_samples = []  # list of {gaze, screen, head_yaw, head_pitch} for head calib

    def add_sample(self, gaze_x, gaze_y, screen_x, screen_y):
        self._gaze_points.append([gaze_x, gaze_y])
        self._screen_points.append([screen_x, screen_y])

    def add_head_sample(self, gaze_x, gaze_y, screen_x, screen_y, head_yaw=0.0, head_pitch=0.0):
        self._head_samples.append({
            "gaze": [gaze_x, gaze_y],
            "screen": [screen_x, screen_y],
            "head_yaw": head_yaw,
            "head_pitch": head_pitch,
        })

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

        center_idx = _calibration_center_point_index()
        center_start = center_idx * samples_per_point
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
            "head_samples": self._head_samples,
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
        self._head_samples = data.get("head_samples", [])
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
        self._head_samples = []
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
    """Draw button rectangles and labels; recalibrate button is drawn as circle with + (same as calibration target)."""
    recal_label = getattr(config, "RECALIBRATE_BUTTON_LABEL", "RECAL")
    target_color = getattr(config, "CALIBRATION_TARGET_COLOR", (0, 100, 0))
    for i, (label, (x, y, w, h)) in enumerate(button_rects):
        if label == recal_label:
            # Recalibrate button: circle with + inside (same shape as calibration target)
            cx, cy = x + w // 2, y + h // 2
            r = min(w, h) // 2 - 8
            r = max(r, 12)
            thickness = 4 if i == highlighted_index else 3
            _draw_calibration_target(frame, cx, cy, r, target_color, thickness)
            if i == highlighted_index:
                cv2.circle(frame, (cx, cy), r + 4, config.BUTTON_COLOR_HIGHLIGHT, 2)
        else:
            color = config.BUTTON_COLOR_HIGHLIGHT if i == highlighted_index else config.BUTTON_COLOR_NORMAL
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, config.BUTTON_THICKNESS)
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
# Calibration flow: 16-point with smooth dot movement + head-movement phase
# -----------------------------------------------------------------------------

# Target shape: circle with + inside (dark green on light grey); same shape used for recalibrate button
def _draw_calibration_target(img, center_x, center_y, radius, color_bgr, thickness=4):
    """Draw circle with + (cross) inside at (center_x, center_y)."""
    cx, cy = int(center_x), int(center_y)
    r = int(radius)
    cv2.circle(img, (cx, cy), r, color_bgr, thickness)
    # Plus: horizontal and vertical lines through center
    arm = int(r * 0.6)
    cv2.line(img, (cx - arm, cy), (cx + arm, cy), color_bgr, thickness)
    cv2.line(img, (cx, cy - arm), (cx, cy + arm), color_bgr, thickness)


def _draw_calibration_screen(screen_width, screen_height, sx, sy, point_label, countdown_sec=None,
                             draw_x=None, draw_y=None, instruction_text=None):
    """
    Calibration frame: light grey background, one target = circle with + in dark green.
    If draw_x, draw_y are set, draw the target there (for smooth animation); else at (sx, sy).
    """
    img = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    img[:] = getattr(config, "CALIBRATION_BG_COLOR", (200, 200, 200))
    cx = int(draw_x) if draw_x is not None else sx
    cy = int(draw_y) if draw_y is not None else sy
    r = getattr(config, "CALIBRATION_TARGET_RADIUS", 56)
    target_color = getattr(config, "CALIBRATION_TARGET_COLOR", (0, 100, 0))
    _draw_calibration_target(img, cx, cy, r, target_color)
    text_color = getattr(config, "CALIBRATION_TEXT_COLOR", (40, 40, 40))
    cv2.putText(img, str(point_label), (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 2, cv2.LINE_AA)
    if instruction_text:
        cv2.putText(img, instruction_text, (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2, cv2.LINE_AA)
    if countdown_sec is not None and countdown_sec > 0:
        cv2.putText(img, f"Hold still... {int(countdown_sec)}s", (50, screen_height - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2, cv2.LINE_AA)
    cv2.putText(img, "ESC or Q = exit calibration", (50, screen_height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 1, cv2.LINE_AA)
    return img


def _animate_dot_to_target(screen_width, screen_height, start_xy, end_xy, duration_sec,
                           point_label, window_name, exit_check_fn, wait_ms=20, instruction_text=None):
    """Animate the calibration dot from start_xy to end_xy over duration_sec. Returns True if completed, False if user exited."""
    t0 = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - t0
        if elapsed >= duration_sec:
            t = 1.0
        else:
            t = elapsed / duration_sec
            # Ease-in-out so movement is smooth at start and end
            t = t * t * (3.0 - 2.0 * t)
        x = start_xy[0] + t * (end_xy[0] - start_xy[0])
        y = start_xy[1] + t * (end_xy[1] - start_xy[1])
        cal_img = _draw_calibration_screen(
            screen_width, screen_height, end_xy[0], end_xy[1], point_label,
            draw_x=x, draw_y=y, instruction_text=instruction_text,
        )
        cv2.imshow(window_name, cal_img)
        if exit_check_fn(cv2.waitKey(wait_ms) & 0xFF):
            return False
        if t >= 1.0:
            break
    return True


def _collect_samples_at_point(camera, detector, smoother, mapper, screen_width, screen_height,
                              frame_width, frame_height, sx, sy, n_samples, duration_sec,
                              point_label, window_name, add_sample_fn, exit_check_fn):
    """Collect n_samples at (sx,sy); add_sample_fn(mapper, gx, gy, sx, sy, ...). Returns True if done, False if exit."""
    smoother.reset()
    collected = 0
    t0 = time.perf_counter()
    while collected < n_samples:
        ok, frame = camera.read()
        if ok and frame is not None and config.MIRROR_CAMERA:
            frame = cv2.flip(frame, 1)
        if not ok or frame is None:
            cal_img = _draw_calibration_screen(
                screen_width, screen_height, sx, sy, point_label,
                countdown_sec=max(0, duration_sec - (time.perf_counter() - t0)),
            )
            cv2.imshow(window_name, cal_img)
            if exit_check_fn(cv2.waitKey(1) & 0xFF):
                return False
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = detector.process(frame_rgb)
        gaze_norm = estimate_gaze_normalized(landmarks, frame_width, frame_height)
        if gaze_norm is not None:
            gx, gy = smoother.update(gaze_norm[0], gaze_norm[1])
            add_sample_fn(mapper, gx, gy, sx, sy, landmarks)
            collected += 1
        cal_img = _draw_calibration_screen(
            screen_width, screen_height, sx, sy, point_label,
            countdown_sec=max(0, duration_sec - (time.perf_counter() - t0)),
        )
        cv2.imshow(window_name, cal_img)
        if exit_check_fn(cv2.waitKey(1) & 0xFF):
            return False
    return True


def run_head_calibration(camera, detector, smoother, mapper, screen_width, screen_height,
                         frame_width, frame_height, window_name):
    """
    After 16-point calibration: guide user to turn head right→center, left→center, down→center, up→center.
    Smooth animated dot leads the eyes; we collect gaze+head samples at edge and center per direction.
    Uses the existing window (window_name).
    """
    center_x = screen_width // 2
    center_y = screen_height // 2
    margin = getattr(config, "CALIBRATION_GRID_MARGIN", 0.05)
    edge_frac = 1.0 - margin
    guide_dur = getattr(config, "HEAD_CALIBRATION_GUIDE_DURATION_SEC", 2.0)
    transition_dur = getattr(config, "HEAD_CALIBRATION_DOT_TRANSITION_SEC", 1.0)
    pos_dur = getattr(config, "HEAD_CALIBRATION_POSITION_DURATION_MS", 1200) / 1000.0
    n_samples = getattr(config, "HEAD_CALIBRATION_SAMPLES_PER_POSITION", 20)

    def add_head(m, gx, gy, sx, sy, landmarks):
        hy, hp = estimate_head_pose(landmarks) if landmarks else (0.0, 0.0)
        m._head_samples.append({"gaze": [gx, gy], "screen": [sx, sy], "head_yaw": hy, "head_pitch": hp})

    directions = [
        ("right", int(edge_frac * screen_width), center_y, "Turn head RIGHT", "Look at the dot on the right"),
        ("left", int(margin * screen_width), center_y, "Turn head LEFT", "Look at the dot on the left"),
        ("down", center_x, int(edge_frac * screen_height), "Turn head DOWN", "Look at the dot below"),
        ("up", center_x, int(margin * screen_height), "Turn head UP", "Look at the dot above"),
    ]
    for name, edge_x, edge_y, guide_label, dot_label in directions:
        # 1) Smooth guide: dot moves from center to edge so eyes follow
        if not _animate_dot_to_target(
            screen_width, screen_height, (center_x, center_y), (edge_x, edge_y), guide_dur,
            guide_label, window_name, _is_exit_key,
            instruction_text="Follow the dot with your eyes",
        ):
            return False
        # 2) Collect samples at edge
        if not _collect_samples_at_point(
            camera, detector, smoother, mapper, screen_width, screen_height,
            frame_width, frame_height, edge_x, edge_y, n_samples, pos_dur,
            dot_label, window_name,
            add_sample_fn=lambda m, gx, gy, sx, sy, lm: add_head(m, gx, gy, sx, sy, lm),
            exit_check_fn=_is_exit_key,
        ):
            return False
        # 3) Smooth transition: dot moves from edge to center
        if not _animate_dot_to_target(
            screen_width, screen_height, (edge_x, edge_y), (center_x, center_y), transition_dur,
            "Now look at the center", window_name, _is_exit_key,
        ):
            return False
        # 4) Collect samples at center
        if not _collect_samples_at_point(
            camera, detector, smoother, mapper, screen_width, screen_height,
            frame_width, frame_height, center_x, center_y, n_samples, pos_dur,
            "Look at the center dot", window_name,
            add_sample_fn=lambda m, gx, gy, sx, sy, lm: add_head(m, gx, gy, sx, sy, lm),
            exit_check_fn=_is_exit_key,
        ):
            return False
    return True


def run_calibration(camera, detector, smoother, screen_width, screen_height, frame_width, frame_height,
                  window_name):
    """
    Run 16-point calibration with smooth dot movement between points (eyes follow the dot).
    Then run head-movement calibration: right→center, left→center, down→center, up→center.
    Uses the existing window (window_name); does not create or destroy windows.
    Returns a CalibrationMapper with computed transform.
    """
    mapper = CalibrationMapper()
    points = get_calibration_screen_points(screen_width, screen_height, config.CALIBRATION_POINTS)
    n_points = len(points)
    duration = config.CALIBRATION_POINT_DURATION_MS / 1000.0
    samples_per_point = getattr(config, "CALIBRATION_SAMPLES_PER_POINT", 20)
    transition_sec = getattr(config, "CALIBRATION_DOT_TRANSITION_SEC", 1.2)

    # Start at first point (no transition for point 0)
    current_xy = (points[0][0], points[0][1])
    label_base = f"Point {{0}}/{n_points} — look at the dot"

    for p_idx in range(n_points):
        sx, sy = points[p_idx]
        point_label = label_base.format(p_idx + 1)

        # Smooth transition from previous to current point (so dot doesn't jump)
        if p_idx > 0:
            if not _animate_dot_to_target(
                screen_width, screen_height, current_xy, (sx, sy), transition_sec,
                point_label, window_name, _is_exit_key,
            ):
                mapper.compute()
                return mapper
        current_xy = (sx, sy)

        def add_grid(m, gx, gy, sx, sy, _):
            m.add_sample(gx, gy, sx, sy)

        if not _collect_samples_at_point(
            camera, detector, smoother, mapper, screen_width, screen_height,
            frame_width, frame_height, sx, sy, samples_per_point, duration,
            point_label, window_name, add_grid, _is_exit_key,
        ):
            mapper.compute()
            return mapper

    # Head-movement phase
    if not run_head_calibration(camera, detector, smoother, mapper, screen_width, screen_height,
                               frame_width, frame_height, window_name):
        mapper.compute()
        return mapper

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

    cal_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")
    cal_mapper = CalibrationMapper()
    window_name = "Gaze UI - Face mesh + gaze cursor (no auto-click)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    _set_fullscreen(window_name)
    _set_window_on_top(window_name)

    # First time (no calibration file): run calibration immediately in this window, no prompt
    if not os.path.isfile(cal_path):
        cal_mapper = run_calibration(camera, detector, smoother, screen_width, screen_height,
                                     frame_width, frame_height, window_name)
        if cal_mapper._matrix is not None:
            cal_mapper.save(cal_path)
    else:
        cal_mapper.load(cal_path)

    # Button layout (in frame coords); last button is recalibrate (circle with +)
    button_rects = build_button_rectangles(frame_width, frame_height)
    recalibrate_button_index = len(button_rects) - 1  # RECAL is last
    dwell_recal_sec = getattr(config, "DWELL_RECALIBRATE_SEC", 1.5)
    dwell_accumulator = 0.0
    last_frame_time = time.perf_counter()

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

            # Cursor at gaze position
            draw_gaze_cursor(frame, screen_x, screen_y, frame_width, frame_height, screen_width, screen_height)

            # Buttons and highlight; dwell on recalibrate button triggers recalibration
            highlighted = hit_test_buttons(screen_x, screen_y, button_rects, frame_width, frame_height, screen_width, screen_height)
            now = time.perf_counter()
            dt = now - last_frame_time
            last_frame_time = now
            if highlighted == recalibrate_button_index:
                dwell_accumulator += dt
                if dwell_accumulator >= dwell_recal_sec:
                    dwell_accumulator = 0.0
                    cal_mapper = run_calibration(camera, detector, smoother, screen_width, screen_height,
                                                 frame_width, frame_height, window_name)
                    if cal_mapper._matrix is not None:
                        cal_mapper.save(cal_path)
            else:
                dwell_accumulator = 0.0

            draw_buttons(frame, button_rects, highlighted)

            # Fullscreen: scale frame to screen size
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
