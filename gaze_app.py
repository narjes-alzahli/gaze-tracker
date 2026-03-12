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

# Optional: EyeGestures ML-based gaze (only used when config.USE_EYEGESTURES is True)
try:
    from eyeGestures import EyeGestures_v3
    _EYEGESTURES_AVAILABLE = True
except ImportError:
    EyeGestures_v3 = None
    _EYEGESTURES_AVAILABLE = False

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


# -----------------------------------------------------------------------------
# Blink detection (Eye Aspect Ratio)
# -----------------------------------------------------------------------------
# MediaPipe Face Mesh indices for EAR (6 points per eye: outer, inner, upper x2, lower x2)
_EAR_LEFT = (33, 133, 160, 158, 153, 144)   # outer, inner, upper, upper-mid, lower-mid, lower
_EAR_RIGHT = (263, 362, 387, 385, 380, 374)


def _dist(a, b):
    """Euclidean distance between two landmarks (normalized x,y)."""
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def compute_ear(landmarks):
    """
    Eye Aspect Ratio for left and right eye. Lower = more closed.
    Returns (ear_left, ear_right) or (None, None) if landmarks invalid.
    """
    if landmarks is None or len(landmarks) < 400:
        return None, None
    try:
        # EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        def ear_one(pts):
            p1, p4, p2, p3, p5, p6 = [landmarks[i] for i in pts]
            vert1 = _dist(p2, p6)
            vert2 = _dist(p3, p5)
            horiz = _dist(p1, p4)
            if horiz < 1e-6:
                return None
            return (vert1 + vert2) / (2.0 * horiz)
        ear_l = ear_one(_EAR_LEFT)
        ear_r = ear_one(_EAR_RIGHT)
        return ear_l, ear_r
    except (IndexError, KeyError):
        return None, None


class BlinkDetector:
    """Detects blink end (eyes were closed, now open) and reports so UI can show a message."""

    def __init__(self, ear_threshold=None, closed_frames_min=2):
        self.ear_threshold = getattr(config, "BLINK_EAR_THRESHOLD", 0.22) if ear_threshold is None else ear_threshold
        self.closed_frames_min = closed_frames_min
        self._closed_frames = 0
        self._was_closed = False

    def update(self, ear_left, ear_right):
        """
        Call each frame with current EAR. Returns True when a blink just ended
        (eyes were closed for at least closed_frames_min and are now open).
        """
        if ear_left is None or ear_right is None:
            return False
        both_closed = ear_left < self.ear_threshold and ear_right < self.ear_threshold
        if both_closed:
            self._closed_frames += 1
            self._was_closed = True
            return False
        # Eyes open (or at least one open)
        if self._was_closed and self._closed_frames >= self.closed_frames_min:
            self._was_closed = False
            self._closed_frames = 0
            return True
        self._was_closed = False
        self._closed_frames = 0
        return False


# -----------------------------------------------------------------------------
# Gaze smoothing
# -----------------------------------------------------------------------------


class GazeSmoother:
    """Exponential moving average for gaze (x, y) with optional per-frame step cap to reduce jitter and outlier jumps."""

    def __init__(self, alpha=config.SMOOTHING_ALPHA, max_step=None):
        self.alpha = alpha
        self.max_step = getattr(config, "SMOOTHING_MAX_STEP", None) if max_step is None else max_step
        self._x = None
        self._y = None

    def update(self, x, y):
        x, y = float(x), float(y)
        if self._x is None:
            self._x, self._y = x, y
            return self._x, self._y
        if self.max_step is not None and self.max_step > 0:
            dx = np.clip(x - self._x, -self.max_step, self.max_step)
            dy = np.clip(y - self._y, -self.max_step, self.max_step)
            x, y = self._x + dx, self._y + dy
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
        self._head_neutral = None  # (yaw, pitch) when looking at center
        self._head_gain = None  # (gain_x, gain_y) to correct gaze from head movement

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
        # Head-pose correction: from head-movement phase we have center samples (every 2nd group of 20)
        self._head_neutral = None
        self._head_gain = None
        if getattr(config, "USE_HEAD_POSE_CORRECTION", False) and len(self._head_samples) >= 40:
            # Center groups: indices 20-39, 60-79, 100-119, 140-159 (after right->center, left->center, etc.)
            center_ranges = [(20, 40), (60, 80), (100, 120), (140, 160)]
            hy_list = []
            hp_list = []
            for start, end in center_ranges:
                if end <= len(self._head_samples):
                    for h in self._head_samples[start:end]:
                        hy_list.append(h["head_yaw"])
                        hp_list.append(h["head_pitch"])
            if hy_list:
                n_yaw = float(np.median(hy_list))
                n_pitch = float(np.median(hp_list))
                self._head_neutral = (n_yaw, n_pitch)
                nx, ny = self._neutral
                gains_x = []
                gains_y = []
                for h in self._head_samples:
                    hy, hp = h["head_yaw"], h["head_pitch"]
                    gx, gy = h["gaze"][0], h["gaze"][1]
                    dyaw = hy - n_yaw
                    dpitch = hp - n_pitch
                    if abs(dyaw) > 0.02:
                        gains_x.append((gx - nx) / dyaw)
                    if abs(dpitch) > 0.02:
                        gains_y.append((gy - ny) / dpitch)
                self._head_gain = (
                    float(np.median(gains_x)) if gains_x else 0.0,
                    float(np.median(gains_y)) if gains_y else 0.0,
                )
        return True

    def map_to_screen(self, gaze_x, gaze_y, head_yaw=None, head_pitch=None):
        """Map normalized gaze to (screen_x, screen_y). Uses neutral so center gaze -> screen center. Optional head pose corrects for head movement."""
        if self._matrix is None:
            w, h = pyautogui.size()
            return int(gaze_x * w), int(gaze_y * h)
        # Optional head-pose correction: shift gaze so cursor doesn't drift when head turns
        if self._head_neutral is not None and self._head_gain is not None and head_yaw is not None and head_pitch is not None:
            n_yaw, n_pitch = self._head_neutral
            gx_gain, gy_gain = self._head_gain
            gaze_x = gaze_x - gx_gain * (head_yaw - n_yaw)
            gaze_y = gaze_y - gy_gain * (head_pitch - n_pitch)
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
            "head_neutral": list(self._head_neutral) if self._head_neutral is not None else None,
            "head_gain": list(self._head_gain) if self._head_gain is not None else None,
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
        hn = data.get("head_neutral")
        self._head_neutral = tuple(hn) if hn and len(hn) == 2 else None
        hg = data.get("head_gain")
        self._head_gain = tuple(hg) if hg and len(hg) == 2 else None
        return True

    def reset(self):
        self._gaze_points = []
        self._screen_points = []
        self._head_samples = []
        self._matrix = None
        self._neutral = None
        self._head_neutral = None
        self._head_gain = None


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


def draw_gaze_vectors(frame, landmarks, width, height, gaze_screen_x, gaze_screen_y, screen_width, screen_height):
    """Draw short vectors from each pupil (iris) toward the gaze point."""
    if landmarks is None or len(landmarks) <= config.RIGHT_IRIS_INDEX:
        return frame
    gx = gaze_screen_x / screen_width * width
    gy = gaze_screen_y / screen_height * height
    length = getattr(config, "GAZE_VECTOR_LENGTH_PX", 45)
    color = (0, 200, 255)  # BGR
    thickness = 2
    for idx in (config.LEFT_IRIS_INDEX, config.RIGHT_IRIS_INDEX):
        if idx < len(landmarks):
            ex, ey = landmarks[idx].x * width, landmarks[idx].y * height
            dx = gx - ex
            dy = gy - ey
            norm = np.sqrt(dx * dx + dy * dy)
            if norm > 1e-6:
                dx, dy = dx / norm, dy / norm
                end_x = int(ex + dx * length)
                end_y = int(ey + dy * length)
                cv2.arrowedLine(frame, (int(ex), int(ey)), (np.clip(end_x, 0, width - 1), np.clip(end_y, 0, height - 1)), color, thickness, tipLength=0.2)
    return frame


# -----------------------------------------------------------------------------
# Button UI: layout around center camera panel (display coords)
# -----------------------------------------------------------------------------


def get_ui_layout():
    """Return (display_w, display_h, camera_rect, button_rects). All in display coords.
    Choice buttons (A–H) surround the camera: top/bottom are wider, left/right are taller.
    RECAL is last and drawn as circle.
    """
    dw = getattr(config, "UI_DISPLAY_WIDTH", 1280)
    dh = getattr(config, "UI_DISPLAY_HEIGHT", 720)
    cw = getattr(config, "UI_CAMERA_PANEL_WIDTH", 480)
    ch = getattr(config, "UI_CAMERA_PANEL_HEIGHT", 360)
    margin = 24
    cx = (dw - cw) // 2
    cy = (dh - ch) // 2
    camera_rect = (cx, cy, cw, ch)
    labels = config.BUTTON_LABELS
    gap = 16          # gap from camera to buttons
    gap_between = 32  # gap between the two buttons on same side (top/bottom/left/right)
    # Top/bottom: wider (horizontal bars)
    btn_wide = 140
    btn_short = 64
    # Left/right: taller (vertical bars)
    btn_narrow = 64
    btn_tall = 140
    recal_size = 72
    # Top row: 2 wide buttons centered above camera
    top_y = cy - gap - btn_short
    top_center_x = cx + cw // 2
    top_left_x = top_center_x - btn_wide - gap_between // 2
    top_right_x = top_center_x + gap_between // 2
    # Bottom row: 2 wide buttons centered below camera
    bottom_y = cy + ch + gap
    bottom_center_x = cx + cw // 2
    bottom_left_x = bottom_center_x - btn_wide - gap_between // 2
    bottom_right_x = bottom_center_x + gap_between // 2
    # Left column: 2 tall buttons vertically centered
    left_x = cx - gap - btn_narrow
    left_center_y = cy + ch // 2
    left_top_y = left_center_y - btn_tall - gap_between // 2
    left_bottom_y = left_center_y + gap_between // 2
    # Right column: 2 tall buttons vertically centered
    right_x = cx + cw + gap
    right_center_y = cy + ch // 2
    right_top_y = right_center_y - btn_tall - gap_between // 2
    right_bottom_y = right_center_y + gap_between // 2
    rects = [
        (labels[0], (top_left_x, top_y, btn_wide, btn_short)),
        (labels[1], (top_right_x, top_y, btn_wide, btn_short)),
        (labels[2], (bottom_left_x, bottom_y, btn_wide, btn_short)),
        (labels[3], (bottom_right_x, bottom_y, btn_wide, btn_short)),
        (labels[4], (left_x, left_top_y, btn_narrow, btn_tall)),
        (labels[5], (left_x, left_bottom_y, btn_narrow, btn_tall)),
        (labels[6], (right_x, right_top_y, btn_narrow, btn_tall)),
        (labels[7], (right_x, right_bottom_y, btn_narrow, btn_tall)),
        (labels[8], (dw - margin - recal_size, dh - margin - recal_size, recal_size, recal_size)),  # RECAL corner
    ]
    return dw, dh, camera_rect, rects


def screen_to_display_point(screen_x, screen_y, display_w, display_h, screen_width, screen_height):
    """Map screen coords to display (composite) coords."""
    dx = screen_x / screen_width * display_w
    dy = screen_y / screen_height * display_h
    return int(dx), int(dy)


def hit_test_buttons_display(screen_x, screen_y, button_rects, display_w, display_h, screen_width, screen_height):
    """Hit-test buttons when using composite layout (button_rects in display coords)."""
    dx, dy = screen_to_display_point(screen_x, screen_y, display_w, display_h, screen_width, screen_height)
    for i, (_, (x, y, w, h)) in enumerate(button_rects):
        if x <= dx <= x + w and y <= dy <= y + h:
            return i
    return None


def _on_mouse_recal(event, x, y, flags, param):
    """Mouse callback: on left-click, if click is on RECAL button (in display coords), request recalibration."""
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    display_w, display_h, screen_width, screen_height, (rx, ry, rw, rh), recal_requested = param
    if screen_width <= 0 or screen_height <= 0:
        return
    dx = x * display_w / screen_width
    dy = y * display_h / screen_height
    if rx <= dx <= rx + rw and ry <= dy <= ry + rh:
        recal_requested[0] = True


def draw_composite(composite, camera_frame, camera_rect, button_rects, highlighted_index,
                   screen_x, screen_y, screen_width, screen_height):
    """Fill composite with light gray, blit camera in center, draw buttons and gaze cursor. composite modified in place."""
    dw, dh = composite.shape[1], composite.shape[0]
    composite[:] = getattr(config, "UI_BG_COLOR", (200, 200, 200))
    cx, cy, cw, ch = camera_rect
    cam_resized = cv2.resize(camera_frame, (cw, ch), interpolation=cv2.INTER_LINEAR)
    composite[cy:cy + ch, cx:cx + cw] = cam_resized
    recal_label = getattr(config, "RECALIBRATE_BUTTON_LABEL", "RECAL")
    target_color = getattr(config, "CALIBRATION_TARGET_COLOR", (0, 100, 0))
    for i, (label, (x, y, w, h)) in enumerate(button_rects):
        if label == recal_label:
            cx, cy = x + w // 2, y + h // 2
            r = min(w, h) // 2 - 8
            r = max(r, 12)
            thick = 4 if i == highlighted_index else 3
            _draw_calibration_target(composite, cx, cy, r, target_color, thick)
            if i == highlighted_index:
                cv2.circle(composite, (cx, cy), r + 4, config.BUTTON_COLOR_HIGHLIGHT, 2)
        else:
            color = config.BUTTON_COLOR_HIGHLIGHT if i == highlighted_index else config.BUTTON_COLOR_NORMAL
            cv2.rectangle(composite, (x, y), (x + w, y + h), color, config.BUTTON_THICKNESS)
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, config.BUTTON_FONT_SCALE, config.BUTTON_THICKNESS)
            tx = x + (w - tw) // 2
            ty = y + (h + th) // 2
            cv2.putText(composite, label, (tx, ty), font, config.BUTTON_FONT_SCALE, config.BUTTON_TEXT_COLOR, config.BUTTON_THICKNESS, cv2.LINE_AA)
    # Gaze cursor on composite (display coords)
    dx, dy = screen_to_display_point(screen_x, screen_y, dw, dh, screen_width, screen_height)
    dx, dy = np.clip(dx, 0, dw - 1), np.clip(dy, 0, dh - 1)
    r_outer = max(4, config.CURSOR_OUTER_RADIUS * dw // 640)
    r_inner = max(2, config.CURSOR_INNER_RADIUS * dw // 640)
    center = (dx, dy)
    cv2.circle(composite, center, r_outer + 4, config.CURSOR_GLOW_COLOR, 4)
    cv2.circle(composite, center, r_outer, config.CURSOR_RING_COLOR, 2)
    cv2.circle(composite, center, r_outer - 2, config.CURSOR_FILL_COLOR, -1)
    cv2.circle(composite, center, r_inner, config.CURSOR_RING_COLOR, -1)
    return composite


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

def _draw_calibration_target(img, center_x, center_y, radius, color_bgr, thickness=4):
    """Draw filled green target: solid circle + thick black outline and black + lines."""
    cx, cy = int(center_x), int(center_y)
    r = int(radius)
    # Filled circle in target color
    cv2.circle(img, (cx, cy), r, color_bgr, -1)
    # Outer black ring
    cv2.circle(img, (cx, cy), r, (0, 0, 0), max(thickness, 4))
    # Plus: horizontal and vertical lines through center in black
    arm = int(r * 0.6)
    line_thick = max(thickness, 4)
    cv2.line(img, (cx - arm, cy), (cx + arm, cy), (0, 0, 0), line_thick)
    cv2.line(img, (cx, cy - arm), (cx, cy + arm), (0, 0, 0), line_thick)


def _draw_calibration_screen(screen_width, screen_height, sx, sy, point_label=None, countdown_sec=None,
                             draw_x=None, draw_y=None, instruction_text=None, show_text=False):
    """
    Calibration frame: light grey background, one target = circle with + in dark green.
    When show_text is False, no text is drawn (clean screen with only the target).
    """
    img = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    img[:] = getattr(config, "CALIBRATION_BG_COLOR", (200, 200, 200))
    cx = int(draw_x) if draw_x is not None else sx
    cy = int(draw_y) if draw_y is not None else sy
    r = getattr(config, "CALIBRATION_TARGET_RADIUS", 56)
    target_color = getattr(config, "CALIBRATION_TARGET_COLOR", (0, 100, 0))
    _draw_calibration_target(img, cx, cy, r, target_color)
    if show_text and point_label is not None:
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


def _show_instruction_screen(screen_width, screen_height, window_name, lines, exit_check_fn):
    """
    Show a full-screen instruction (same bg as calibration). lines = list of strings.
    Waits for any key; returns False if user hits ESC/Q (exit), True to continue.
    No gaze or face data is collected here so calibration logic stays correct.
    """
    img = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    img[:] = getattr(config, "CALIBRATION_BG_COLOR", (200, 200, 200))
    text_color = getattr(config, "CALIBRATION_TEXT_COLOR", (40, 40, 40))
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale_title = 1.2
    scale_body = 0.9
    y = 120
    for i, line in enumerate(lines):
        if not line.strip():
            y += 28
            continue
        scale = scale_title if i == 0 else scale_body
        (tw, th), _ = cv2.getTextSize(line, font, scale, 2)
        x = (screen_width - tw) // 2
        cv2.putText(img, line, (x, y), font, scale, text_color, 2, cv2.LINE_AA)
        y += int(th * 1.3)
    prompt = "Press any key to continue"
    (pw, ph), _ = cv2.getTextSize(prompt, font, 0.9, 2)
    cv2.putText(img, prompt, ((screen_width - pw) // 2, screen_height - 60), font, 0.9, text_color, 2, cv2.LINE_AA)
    cv2.putText(img, "ESC or Q = exit", ((screen_width - 180) // 2, screen_height - 20), font, 0.6, text_color, 1, cv2.LINE_AA)
    cv2.imshow(window_name, img)
    key = cv2.waitKey(0) & 0xFF
    return not exit_check_fn(key)


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
            screen_width, screen_height, end_xy[0], end_xy[1],
            draw_x=x, draw_y=y, show_text=False,
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
                screen_width, screen_height, sx, sy, show_text=False,
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
            screen_width, screen_height, sx, sy, show_text=False,
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
    Instruction was already shown before this phase; here we only show the dot (no text).
    Uses the existing window (window_name).
    """
    smoother.reset()  # Clean state for head phase so first samples are not contaminated
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
    Instructions are shown once before each section; during calibration only the dot is shown.
    Uses the existing window (window_name); does not create or destroy windows.
    Returns a CalibrationMapper with computed transform.
    """
    mapper = CalibrationMapper()
    points = get_calibration_screen_points(screen_width, screen_height, config.CALIBRATION_POINTS)
    n_points = len(points)
    duration = config.CALIBRATION_POINT_DURATION_MS / 1000.0
    samples_per_point = getattr(config, "CALIBRATION_SAMPLES_PER_POINT", 20)
    transition_sec = getattr(config, "CALIBRATION_DOT_TRANSITION_SEC", 1.2)

    # Section 1: Look at the dot (16 points). Show instructions first; no gaze collected here.
    instructions_look = getattr(config, "CALIBRATION_INSTRUCTIONS_LOOK_AT_DOT", [
        "Look at the dot", "A dot will move to 16 points.",
        "Look at the dot and hold still at each position.", "", "Press any key to start.",
    ])
    if not _show_instruction_screen(screen_width, screen_height, window_name, instructions_look, _is_exit_key):
        mapper.compute()
        return mapper
    smoother.reset()  # Clean state before collecting so first point is not contaminated

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

    # Section 2 (optional): Move head for head-pose correction. Skip if RUN_HEAD_CALIBRATION is False.
    if getattr(config, "RUN_HEAD_CALIBRATION", False):
        instructions_head = getattr(config, "CALIBRATION_INSTRUCTIONS_MOVE_HEAD", [
            "Move your head",
            "Follow the dot: turn your head right, then center; left, then center; down, then center; up, then center.",
            "", "Press any key to start.",
        ])
        if not _show_instruction_screen(screen_width, screen_height, window_name, instructions_head, _is_exit_key):
            mapper.compute()
            return mapper
        if not run_head_calibration(camera, detector, smoother, mapper, screen_width, screen_height,
                                   frame_width, frame_height, window_name):
            mapper.compute()
            return mapper

    mapper.compute()
    return mapper


def run_calibration_eyegestures(camera, gestures, screen_width, screen_height, window_name):
    """
    Run EyeGestures V3 calibration: show dot at current target, collect samples until
    all 25 points are done (point moves when user looks at dot long enough). Returns True if completed.
    """
    context = getattr(config, "EYEGESTURES_CONTEXT", "main")
    transition_count = 0
    last_point = None
    num_points = 25  # EyeGestures CalibrationMatrix default
    bg_color = getattr(config, "CALIBRATION_BG_COLOR", (200, 200, 200))
    target_color = getattr(config, "CALIBRATION_TARGET_COLOR", (0, 100, 0))
    r = getattr(config, "CALIBRATION_TARGET_RADIUS", 56)

    while transition_count < num_points:
        ok, frame = camera.read()
        if not ok or frame is None:
            img = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
            img[:] = bg_color
            cv2.imshow(window_name, img)
            if _is_exit_key(cv2.waitKey(30) & 0xFF):
                return False
            continue
        # EyeGestures flips frame internally; pass raw BGR
        event, cevent = gestures.step(frame, True, screen_width, screen_height, context=context)
        if cevent is None:
            continue
        pt = cevent.point
        if last_point is not None and (abs(pt[0] - last_point[0]) > 5 or abs(pt[1] - last_point[1]) > 5):
            transition_count += 1
        last_point = np.array([pt[0], pt[1]])

        img = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        img[:] = bg_color
        cx, cy = int(pt[0]), int(pt[1])
        _draw_calibration_target(img, cx, cy, r, target_color)
        text_color = getattr(config, "CALIBRATION_TEXT_COLOR", (40, 40, 40))
        cv2.putText(img, f"Point {transition_count + 1}/{num_points}  Look at the dot", (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2, cv2.LINE_AA)
        cv2.putText(img, "ESC or Q = exit", (50, screen_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 1, cv2.LINE_AA)
        cv2.imshow(window_name, img)
        if _is_exit_key(cv2.waitKey(1) & 0xFF):
            return False
    return True


# -----------------------------------------------------------------------------
# Main application loop
# -----------------------------------------------------------------------------


def main(use_gestures=False):
    # Screen size (for mapping gaze to screen and for calibration)
    screen_width, screen_height = pyautogui.size()

    # Webcam
    camera = Camera()
    camera.start()
    frame_width = int(camera._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(camera._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    use_eyegestures = (getattr(config, "USE_EYEGESTURES", False) or use_gestures) and _EYEGESTURES_AVAILABLE
    gestures = None
    if use_eyegestures:
        try:
            calibration_radius = getattr(config, "EYEGESTURES_CALIBRATION_RADIUS", 1000)
            gestures = EyeGestures_v3(calibration_radius=calibration_radius)
        except Exception:
            use_eyegestures = False

    # Face mesh (used for blink + overlay in both modes; for gaze only when not EyeGestures)
    detector = FaceMeshDetector()
    if not use_eyegestures:
        smoother = GazeSmoother(alpha=config.SMOOTHING_ALPHA)
        cal_mapper = CalibrationMapper()
    else:
        smoother = None
        cal_mapper = None
    cal_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")
    eg_cal_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eyegestures_calibration.pkl")
    context = getattr(config, "EYEGESTURES_CONTEXT", "main")

    window_name = "Gaze UI - Face mesh + gaze cursor (no auto-click)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    _set_fullscreen(window_name)
    _set_window_on_top(window_name)

    if use_eyegestures:
        if os.path.isfile(eg_cal_path):
            try:
                with open(eg_cal_path, "rb") as f:
                    gestures.loadModel(f.read(), context=context)
            except Exception:
                if run_calibration_eyegestures(camera, gestures, screen_width, screen_height, window_name):
                    try:
                        with open(eg_cal_path, "wb") as f:
                            f.write(gestures.saveModel(context=context) or b"")
                    except Exception:
                        pass
        else:
            if run_calibration_eyegestures(camera, gestures, screen_width, screen_height, window_name):
                try:
                    with open(eg_cal_path, "wb") as f:
                        f.write(gestures.saveModel(context=context) or b"")
                except Exception:
                    pass
    else:
        # First time (no calibration file): run calibration immediately in this window, no prompt
        if not os.path.isfile(cal_path):
            cal_mapper = run_calibration(camera, detector, smoother, screen_width, screen_height,
                                         frame_width, frame_height, window_name)
            if cal_mapper._matrix is not None:
                cal_mapper.save(cal_path)
        else:
            cal_mapper.load(cal_path)

    # Layout: camera in center (smaller), buttons around it, light gray background
    display_w, display_h, camera_rect, button_rects = get_ui_layout()
    recalibrate_button_index = len(button_rects) - 1
    recal_rect = button_rects[recalibrate_button_index][1]  # (x, y, w, h) in display coords
    recal_click_requested = [False]
    cv2.setMouseCallback(window_name, _on_mouse_recal,
                         (display_w, display_h, screen_width, screen_height, recal_rect, recal_click_requested))
    dwell_recal_sec = getattr(config, "DWELL_RECALIBRATE_SEC", 1.5)
    dwell_accumulator = 0.0
    last_frame_time = time.perf_counter()
    composite = np.zeros((display_h, display_w, 3), dtype=np.uint8)
    blink_detector = BlinkDetector(
        closed_frames_min=getattr(config, "BLINK_CLOSED_FRAMES_MIN", 2),
    )
    blink_message_until = 0.0
    last_valid_frame = None  # reuse when camera temporarily fails so we still draw UI

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                # Still draw UI with last frame or placeholder so window isn't stuck gray
                if last_valid_frame is not None:
                    frame_display = last_valid_frame
                else:
                    frame_display = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                    frame_display[:] = getattr(config, "UI_BG_COLOR", (200, 200, 200))
                    cv2.putText(frame_display, "No camera", (frame_width // 4, frame_height // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 80, 80), 2, cv2.LINE_AA)
                landmarks = None
                screen_x, screen_y = screen_width // 2, screen_height // 2
                # still do draw_composite and imshow below
            else:
                frame_display = cv2.flip(frame, 1) if config.MIRROR_CAMERA else frame
                last_valid_frame = frame_display.copy()
                frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
                landmarks = detector.process(frame_rgb)

                ear_left, ear_right = compute_ear(landmarks)
                if blink_detector.update(ear_left, ear_right):
                    blink_message_until = time.perf_counter() + getattr(config, "BLINK_MESSAGE_DURATION_SEC", 2.0)

                if use_eyegestures:
                    # EyeGestures expects unflipped BGR; they flip internally. Can throw if no face.
                    try:
                        frame_for_eg = frame
                        event, _cevent = gestures.step(frame_for_eg, False, screen_width, screen_height, context=context)
                        if event is not None:
                            screen_x = int(np.clip(event.point[0], 0, screen_width - 1))
                            screen_y = int(np.clip(event.point[1], 0, screen_height - 1))
                        else:
                            screen_x, screen_y = screen_width // 2, screen_height // 2
                    except Exception:
                        screen_x, screen_y = screen_width // 2, screen_height // 2
                else:
                    gaze_norm = estimate_gaze_normalized(landmarks, frame_width, frame_height)
                    head_yaw, head_pitch = estimate_head_pose(landmarks) if landmarks else (None, None)
                    if gaze_norm is not None:
                        smooth_x, smooth_y = smoother.update(gaze_norm[0], gaze_norm[1])
                        screen_x, screen_y = cal_mapper.map_to_screen(smooth_x, smooth_y, head_yaw=head_yaw, head_pitch=head_pitch)
                        screen_x += getattr(config, "CALIBRATION_OFFSET_PX_X", 0)
                        screen_y += getattr(config, "CALIBRATION_OFFSET_PX_Y", 0)
                        screen_x = int(np.clip(screen_x, 0, screen_width - 1))
                        screen_y = int(np.clip(screen_y, 0, screen_height - 1))
                    else:
                        screen_x, screen_y = screen_width // 2, screen_height // 2
                        smoother.reset()

            # On camera frame: face mesh + short gaze vectors (no cursor here; cursor on composite)
            draw_face_mesh_overlay(frame_display, landmarks, frame_width, frame_height)
            draw_gaze_vectors(frame_display, landmarks, frame_width, frame_height, screen_x, screen_y, screen_width, screen_height)

            highlighted = hit_test_buttons_display(screen_x, screen_y, button_rects, display_w, display_h, screen_width, screen_height)
            now = time.perf_counter()
            dt = now - last_frame_time
            last_frame_time = now
            if highlighted == recalibrate_button_index:
                dwell_accumulator += dt
                if dwell_accumulator >= dwell_recal_sec:
                    dwell_accumulator = 0.0
                    if use_eyegestures:
                        gestures.reset(context=context)
                        if run_calibration_eyegestures(camera, gestures, screen_width, screen_height, window_name):
                            try:
                                with open(eg_cal_path, "wb") as f:
                                    f.write(gestures.saveModel(context=context) or b"")
                            except Exception:
                                pass
                    else:
                        cal_mapper = run_calibration(camera, detector, smoother, screen_width, screen_height,
                                                     frame_width, frame_height, window_name)
                        if cal_mapper._matrix is not None:
                            cal_mapper.save(cal_path)
            else:
                dwell_accumulator = 0.0

            if recal_click_requested[0]:
                recal_click_requested[0] = False
                if use_eyegestures:
                    gestures.reset(context=context)
                    if run_calibration_eyegestures(camera, gestures, screen_width, screen_height, window_name):
                        try:
                            with open(eg_cal_path, "wb") as f:
                                f.write(gestures.saveModel(context=context) or b"")
                        except Exception:
                            pass
                else:
                    cal_mapper = run_calibration(camera, detector, smoother, screen_width, screen_height,
                                                 frame_width, frame_height, window_name)
                    if cal_mapper._matrix is not None:
                        cal_mapper.save(cal_path)

            draw_composite(composite, frame_display, camera_rect, button_rects, highlighted,
                          screen_x, screen_y, screen_width, screen_height)
            if now < blink_message_until:
                dw, dh = composite.shape[1], composite.shape[0]
                msg = "Blinking"
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1.4
                thickness = 3
                (tw, th), _ = cv2.getTextSize(msg, font, scale, thickness)
                tx = (dw - tw) // 2
                ty = 56
                cv2.putText(composite, msg, (tx, ty), font, scale, (0, 0, 120), thickness + 2, cv2.LINE_AA)
                cv2.putText(composite, msg, (tx, ty), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
            frame_display = cv2.resize(composite, (screen_width, screen_height), interpolation=cv2.INTER_LINEAR)
            cv2.imshow(window_name, frame_display)
            key = cv2.waitKey(1) & 0xFF
            if _is_exit_key(key):
                break
            if key == ord("r") and smoother is not None:
                smoother.reset()
    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Gaze tracker")
    p.add_argument("--gestures", action="store_true", help="Use EyeGestures ML-based gaze")
    args = p.parse_args()
    main(use_gestures=args.gestures)
