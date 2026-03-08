# Nexa Gaze Tracker

A Python desktop application that uses your webcam and gaze direction to highlight on-screen buttons (no automatic clicking). Built with **OpenCV**, **MediaPipe Face Mesh**, and **PyAutoGUI**.

## Features

- **Fullscreen** main window and calibration window
- **Live webcam** with face mesh overlay and iris/eye landmarks
- **Gaze estimation** from MediaPipe iris landmarks (sensitivity and smoothing in `config.py`)
- **Cursor** with ring + inner dot; calibration targets use the same style
- **Large option buttons** (YES, NO, HELP, WATER, FOOD) that **highlight when your gaze is on them** (no click)
- **16-point calibration** (4×4 grid) with **smooth dot movement** between points so your eyes can follow the target (no jump). Then **head-movement calibration**: the dot leads you to turn head right→center, left→center, down→center, up→center (with smooth animation so you know where to look). Center-neutral mapping; grid uses `CALIBRATION_GRID_MARGIN` (default 5%). Re-run calibration after changing mirror/sensitivity.
- **Mirrored camera** (configurable) so the feed feels natural
- **Center**: Calibration now defines "center" from your center-dot samples. If the cursor is still slightly off after calibration, use `CALIBRATION_OFFSET_PX_X/Y` in `config.py` (pixel nudge) or, without calibration, `GAZE_CENTER_BIAS_X/Y`.
- **Gaze direction**: with mirror on, `GAZE_MIRROR_CORRECT_X` keeps left/right correct; `GAZE_SENSITIVITY_Y` and `GAZE_SWAP_XY` fix diagonal or wrong-axis movement
- Modular, well-commented code for easy extension

## Setup

```bash
cd c:\asl\ready-gazer
pip install -r requirements.txt
```

## Run

```bash
python gaze_app.py
```

- **ESC** or **Q** – Quit
- **R** – Reset gaze smoother (if cursor drifts)

On first run you’ll be asked whether to run calibration. Answer **y** to do **16-point calibration** (the dot moves smoothly between points—follow it with your eyes), then **head calibration** (turn head right, left, down, up following the dot; then to center). The result is saved to `calibration.json` for next time.

## Structure

- `config.py` – Camera index, resolution, landmark indices, smoothing, button layout, colors
- `gaze_app.py` – Main app with sections:
  - Webcam capture (`Camera`)
  - Face mesh detection (`FaceMeshDetector`, MediaPipe with `refine_landmarks=True` for iris)
  - Gaze estimation (iris relative to eye corners → normalized → screen)
  - Smoothing (`GazeSmoother`, EMA)
  - Cursor visualization and button UI / highlight logic
  - Optional 16-point + head-movement calibration (`CalibrationMapper`, smooth dot animation)

Dependencies: **OpenCV**, **MediaPipe**, **PyAutoGUI**, **NumPy** (and standard library).
