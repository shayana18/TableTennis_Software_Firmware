# Table Tennis Robot - Vision System

Stereo vision system for autonomous table tennis robot ball tracking.

## Project Structure

```
top_level_control/
├── calibration/                  # Camera calibration
│   ├── calib.py                  # Calibration script (intrinsic + extrinsic)
│   ├── calibration_settings.yaml # Settings for calibration
│   └── camera_parameters/        # Generated calibration data (after running calib.py)
│
├── config/
│   └── stereo_config.yaml        # Stereo camera configuration
│
├── tracking/                     # Ball tracking modules
│   ├── ball_tracker.py           # Single camera ball detection (HSV-based)
│   ├── stereo_detector.py        # Detect ball in both cameras (no 3D)
│   └── stereo_triangulator.py    # Full 3D tracking with triangulation
│
└── scripts/                      # Test scripts
    ├── test_single_camera.py     # Test detection on one camera
    ├── test_stereo_detection.py  # Test detection on both cameras
    └── test_triangulation.py     # Test 3D triangulation (verify calibration!)
```

## Setup Workflow

### Step 1: Camera Calibration

Run calibration to get intrinsic and extrinsic parameters:

```bash
cd calibration
python calib.py calibration_settings.yaml
```

This will:
1. Capture calibration frames for each camera (checkerboard)
2. Calculate intrinsic parameters (camera matrix, distortion)
3. Capture stereo pairs
4. Calculate extrinsic parameters (R, T between cameras)

**Goal: RMS error < 0.5 is good, < 0.3 is excellent**

### Step 2: Test Single Camera Detection

Make sure ball detection works on each camera:

```bash
cd scripts
python test_single_camera.py
```

Controls:
- `q` - Quit
- `t` - Open HSV tuner (adjust if ball not detected)
- `m` - Show mask
- `0-9` - Switch camera ID

### Step 3: Test Stereo Detection

Test that ball is detected in BOTH cameras simultaneously:

```bash
python test_stereo_detection.py
```

Both cameras must detect the ball before triangulation will work!

### Step 4: Test Triangulation (Verify Calibration!)

This is the key test to verify your calibration works:

```bash
python test_triangulation.py
```

**How to verify:**
1. Hold ball at a known distance (measure with ruler/tape)
2. Check if displayed Z (depth) matches your measurement
3. Test at multiple distances: 500mm, 1000mm, 1500mm
4. If Z is way off, recalibrate!

Controls:
- `q` - Quit
- `s` - Save measurement
- `r` - Reset measurements

## Module Usage

### BallTracker (single camera)
```python
from tracking import BallTracker

tracker = BallTracker()
result = tracker.detect(frame)

if result['found']:
    center = result['center']  # (x, y) pixels
    radius = result['radius']  # pixels
```

### StereoDetector (both cameras, no 3D)
```python
from tracking import StereoDetector

detector = StereoDetector(cam_left_id=0, cam_right_id=1)
detector.start_cameras()

result = detector.detect()
if result['both_found']:
    # Ready for triangulation!
    left_pt = result['left_detection']['center']
    right_pt = result['right_detection']['center']

detector.stop_cameras()
```

### StereoTriangulator (full 3D tracking)
```python
from tracking import StereoTriangulator

triangulator = StereoTriangulator(
    calibration_dir='calibration/camera_parameters',
    cam_left_id=0,
    cam_right_id=1
)
triangulator.start_cameras()

result = triangulator.update()
if result['found_3d']:
    X, Y, Z = result['position_3d']  # millimeters
    print(f"Ball at: ({X}, {Y}, {Z}) mm")

triangulator.stop_cameras()
```

## Troubleshooting

### Ball not detected
- Run `test_single_camera.py` with HSV tuner (`t` key)
- Adjust H, S, V ranges until ball shows white in mask
- Update `DEFAULT_COLOR_LOWER/UPPER` in `ball_tracker.py`

### Z (depth) is wrong
- Recalibrate! Make sure checkerboard is detected clearly
- Verify checkerboard size in `calibration_settings.yaml`
- Use more calibration frames (15-20 is good)
- Make sure checkerboard is at various angles/distances

### Cameras not found
- Check camera IDs (try 0, 1, 2, etc.)
- On Windows: Device Manager > Cameras
- On Linux: `ls /dev/video*`

### Large Y difference between detections
- Cameras may not be level/aligned
- Recalibrate with cameras in final position
- Check `max_y_diff` parameter (default 50px)