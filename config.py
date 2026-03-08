"""
Configuration and constants for the gaze-based desktop application.
Easily extend by adding new buttons or changing layout here.
"""

# -----------------------------------------------------------------------------
# Screen and camera
# -----------------------------------------------------------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MIRROR_CAMERA = True  # Mirror image so it feels natural to the viewer

# -----------------------------------------------------------------------------
# MediaPipe Face Mesh landmark indices (with refine_landmarks=True for iris)
# -----------------------------------------------------------------------------
# Left eye: outer 33, inner 133; Right eye: outer 263, inner 362
LEFT_EYE_INDICES = [33, 133]   # outer, inner
RIGHT_EYE_INDICES = [263, 362]
# Iris centers (only present when refine_landmarks=True)
LEFT_IRIS_INDEX = 468
RIGHT_IRIS_INDEX = 473

# -----------------------------------------------------------------------------
# Gaze smoothing and sensitivity
# -----------------------------------------------------------------------------
SMOOTHING_ALPHA = 0.2   # Lower = smoother, less jitter (slightly more lag)
GAZE_SENSITIVITY = 2.2  # Scale iris movement to screen; >1 = more sensitive
GAZE_SENSITIVITY_Y = None  # If set, use for vertical (else same as GAZE_SENSITIVITY); use if cursor moves diagonally
# If cursor is off when you look at screen center, tweak these (e.g. cursor right of center -> negative X)
GAZE_CENTER_BIAS_X = 0.0
GAZE_CENTER_BIAS_Y = 0.0
# Fix wrong direction: when using mirrored camera, horizontal gaze is flipped so left/right match
GAZE_MIRROR_CORRECT_X = True  # Set False if left/right are swapped after enabling mirror
# If cursor moves diagonally (e.g. look up -> cursor goes right), try swapping axes
GAZE_SWAP_XY = False

# -----------------------------------------------------------------------------
# Calibration
# -----------------------------------------------------------------------------
CALIBRATION_POINTS = 16  # 4x4 grid (like Gaze Pointer)
# How close to screen edges the outer dots are (0.05 = 5% inset so corners are calibrated, not extrapolated)
CALIBRATION_GRID_MARGIN = 0.05  # Outer points at 5% and 95%; center at 50%. Covers full screen.
# Smooth movement: dot travels from one point to the next so eyes can follow (no jump)
CALIBRATION_DOT_TRANSITION_SEC = 1.2  # Time for dot to move between points
CALIBRATION_SAMPLES_PER_POINT = 28   # More samples = more stable fit; center uses these for "neutral"
CALIBRATION_POINT_DURATION_MS = 1500  # Time to look at each point once arrived
# Head-movement calibration (after 16 points): right→center, left→center, down→center, up→center
HEAD_CALIBRATION_GUIDE_DURATION_SEC = 2.0   # Time to show arrow/smooth guide before dot appears
HEAD_CALIBRATION_SAMPLES_PER_POSITION = 20  # Samples at edge and at center per direction
HEAD_CALIBRATION_POSITION_DURATION_MS = 1200  # How long to hold at each position (edge or center)
HEAD_CALIBRATION_DOT_TRANSITION_SEC = 1.0   # Smooth move from edge to center
# Fine-tune cursor after calibration (pixels): if center is still off, nudge without re-calibrating
CALIBRATION_OFFSET_PX_X = 0
CALIBRATION_OFFSET_PX_Y = 0

# -----------------------------------------------------------------------------
# Button UI layout (labels and relative positions for a grid)
# -----------------------------------------------------------------------------
# Last button is recalibrate: drawn as circle with + (same shape as calibration target)
BUTTON_LABELS = ["YES", "NO", "HELP", "WATER", "FOOD", "RECAL"]
# Layout: 3 cols x 2 rows = 6 (RECAL is bottom-right)
BUTTON_GRID_COLS = 3
BUTTON_GRID_ROWS = 2
# Recalibrate button: same label as in BUTTON_LABELS; drawn as circle with + (calibration target shape)
RECALIBRATE_BUTTON_LABEL = "RECAL"
# Dwell time on recalibrate button (seconds) to trigger recalibration
DWELL_RECALIBRATE_SEC = 1.5

# Button appearance
BUTTON_PADDING_FRACTION = 0.08  # Padding as fraction of button size
BUTTON_FONT_SCALE = 1.2
BUTTON_THICKNESS = 3
BUTTON_COLOR_NORMAL = (70, 70, 70)       # BGR dark gray
BUTTON_COLOR_HIGHLIGHT = (50, 200, 255)  # BGR orange
BUTTON_TEXT_COLOR = (255, 255, 255)

# -----------------------------------------------------------------------------
# Calibration UI (light grey background, target = circle with + in dark green)
# -----------------------------------------------------------------------------
CALIBRATION_BG_COLOR = (200, 200, 200)   # BGR light grey
CALIBRATION_TARGET_COLOR = (0, 100, 0)    # BGR dark green (circle + plus)
CALIBRATION_TARGET_RADIUS = 56            # Circle radius for calibration target
CALIBRATION_TEXT_COLOR = (40, 40, 40)     # BGR dark grey text on light background

# -----------------------------------------------------------------------------
# Cursor / gaze dot (nicer look: ring + inner dot)
# -----------------------------------------------------------------------------
CURSOR_OUTER_RADIUS = 18   # Outer ring
CURSOR_INNER_RADIUS = 6    # Inner bright dot
CURSOR_RING_COLOR = (255, 255, 255)   # BGR white border
CURSOR_FILL_COLOR = (255, 255, 180)   # BGR soft cyan
CURSOR_GLOW_COLOR = (200, 220, 200)   # BGR soft outer ring
