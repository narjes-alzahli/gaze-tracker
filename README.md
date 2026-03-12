# Gaze Tracker

Webcam-based gaze tracking: cursor follows your eyes and highlights on-screen buttons. No automatic clicking. Uses **OpenCV**, **MediaPipe Face Mesh**, and **PyAutoGUI**.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python gaze_app.py
```

Use **EyeGestures** (ML-based gaze, requires `eyeGestures` package):

```bash
python gaze_app.py --gestures
```

- **ESC** or **Q** — quit  
- **R** — reset gaze smoother

On first run, calibration runs automatically (16-point grid; follow the dot with your eyes). Result is saved to `calibration.json`. Use the **RECAL** button (or dwell on it) to recalibrate.

## Config

- `config.py` — camera, resolution, smoothing, button labels, calibration options  
- `gaze_app.py` — main app (camera, face mesh, gaze estimation, calibration, UI)

Dependencies: OpenCV, MediaPipe, PyAutoGUI, NumPy.
