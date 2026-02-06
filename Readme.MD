# Table Tennis Robot - Vision System

Stereo vision system for real-time ball tracking and trajectory prediction.

---

## System Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                             VISION PIPELINE                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  CAMERAS          DETECTION         TRIANGULATION        PREDICTION          │
│                                                                              │
│ ┌───────┐       ┌─────────┐        ┌───────────┐                            │
│ │ Left  │──────▶│  Ball   │───────▶│           │      ┌──────────────┐      │
│ │Camera │       │Detector │        │    3D     │─────▶│  Trajectory  │      │
│ └───────┘       │(HSV+LAB)│        │Triangulate│      │  Predictor   │      │
│                 └─────────┘        │           │      │              │      │
│ ┌───────┐       ┌─────────┐        │  (X,Y,Z)  │      │  ┌────────┐ │      │
│ │ Right │──────▶│  Ball   │───────▶│           │      │  │ Rising? │ │      │
│ │Camera │       │Detector │        └───────────┘      │  └───┬────┘ │      │
│ └───────┘       └─────────┘                           │    Y │ N    │      │
│                                                       │  ┌──┘ └──┐  │      │
│                                                       │ APEX  Z-PLANE│      │
│                                                       └──────┬───────┘      │
│                                                              ▼              │
│                                                        ┌──────────┐         │
│                                                        │  ROBOT   │         │
│                                                        │ CONTROL  │         │
│                                                        └──────────┘         │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Hardware

**Cameras:** Arducam OV9782 Global Shutter USB (x2)
- Resolution: 1280 x 720
- Frame Rate: 120 fps
- Global shutter eliminates motion blur

---

## Project Structure

```
top_level_control/
├── camera_calibration/           # Calibration tools & output
│   ├── calibration_settings.yaml
│   ├── calibrate.py
│   ├── calib_extrinsic_only.py
│   └── camera_parameters/        # Generated .dat files
│
├── config/                       # Runtime configuration
│   ├── stereo_config.yaml
│   ├── ball_thresholds.json
│   └── ball_thresholds_stereo.json
│
├── tracking/                     # Detection & triangulation
│   ├── ball_tracker.py
│   ├── stereo_detector.py
│   └── stereo_triangulator.py
│
├── trajectory/                   # Prediction module
│   ├── position_buffer.py
│   ├── velocity_estimator.py
│   ├── physics_model.py
│   └── trajectory_predictor.py
│
└── tests/                        # Verification scripts
    ├── test_single_camera.py
    ├── test_stereo_detection.py
    ├── test_triangulation.py
    └── verify_triangulation.py
```

---

## Pipeline Stages

### Stage 1: Camera Calibration

**Purpose:** Determine camera properties and spatial relationship.

**Process:**
1. Capture checkerboard images from each camera
2. Calculate intrinsic parameters (focal length, distortion)
3. Capture synchronized stereo pairs
4. Calculate extrinsic parameters (rotation, translation between cameras)

**Output:** Calibration files in `camera_parameters/`

**Quality Target:** RMSE < 0.3 (excellent), < 0.5 (acceptable)

---

### Stage 2: Ball Detection

**Purpose:** Find the ball's pixel position in each camera frame.

**Process:**
1. Normalize lighting using CLAHE
2. Detect orange color in HSV color space
3. Detect orange color in LAB color space
4. Fuse both masks (55% HSV + 45% LAB)
5. Filter by shape (circularity, solidity, area)
6. Return center (x, y) and radius in pixels

**Output:** Ball center coordinates for left and right camera

---

### Stage 3: 3D Triangulation

**Purpose:** Calculate the ball's real-world 3D position.

**Process:**
1. Get ball pixel coordinates from both cameras
2. Apply calibration parameters (projection matrices)
3. Use Direct Linear Transform (DLT) to solve for 3D point
4. Output position in calibration units (cm)

**Output:** (X, Y, Z) position relative to left camera origin

**Coordinate System:**
```
        Y (up)
        │
        │
        │
Left ───┼─────── Right
Camera  │        Camera
        │
        └──────── X (right)
       /
      /
     Z (depth toward ball)
```

---

### Stage 4: Trajectory Prediction

**Purpose:** Predict where the ball will be in the future and choose the best interception point.

**Process:**
1. **Collect:** Store recent 3D positions in a circular buffer (last 10 frames) with timestamps
2. **Velocity:** Estimate ball velocity via linear regression on position history
3. **Gravity correction:** The regression gives average velocity over the buffer window, not the instantaneous velocity at the latest frame. The predictor corrects Vy to the current moment using the known gravitational acceleration.
4. **Physics:** Apply exact kinematic equations with gravity
   - X and Z: Constant velocity (no air resistance)
   - Y: Accelerates downward at 981 cm/s² (gravity in +Y direction, camera coords)
5. **Two-case interception strategy:**

| Condition | Strategy | Intercept Point |
|-----------|----------|-----------------|
| Ball is rising (Vy < 0) | **Apex** | Trajectory peak where vertical velocity = 0 |
| Ball is flat or falling (Vy >= 0) | **Z-Plane** | Fixed Z distance where robot can reach |

The apex strategy catches tossed/lobbed balls at their highest point. The Z-plane strategy handles flat or descending shots approaching the robot.

**Output:** Interception point (X, Y, Z), time until arrival, and strategy used (apex or z-plane)

**Coordinate System Conventions:**
- +Y is **down** (standard camera image coordinates)
- Ball going **up** physically = Y decreasing = Vy < 0
- Gravity pulls ball **down** = acts in +Y direction

---

## Setup Workflow

### 1. Calibrate Cameras

Print a checkerboard pattern and run calibration:
```
cd camera_calibration
python calibrate.py calibration_settings.yaml
```

Move the checkerboard to various positions and angles. The script will:
- Capture individual camera frames (intrinsic calibration)
- Capture synchronized stereo frames (extrinsic calibration)
- Save parameters to `camera_parameters/`

### 2. Tune Ball Detection

Adjust color thresholds for your lighting conditions:
```
cd tests
python test_single_camera.py
```

Use the threshold tuner to find HSV/LAB values where the ball appears white in the mask. Save thresholds to `config/ball_thresholds_stereo.json`.

### 3. Verify Triangulation

Test that 3D measurements are accurate:
```
python verify_triangulation.py
```

Place the ball at known distances (measure with tape) and compare against displayed Z values. Accuracy should be within 5%.

### 4. Test Trajectory Prediction

Run the full pipeline:
```
cd top_level_control
python -m trajectory.test_trajectory_prediction
```

Toss and throw the ball to verify both strategies:
- **Toss upward** — OSD should show `[APEX]` with intercept at the peak
- **Throw flat/downward** — OSD should show `[Z-PLANE]` with intercept at the robot Z distance
- Velocity values are reasonable (200-800 cm/s for typical throws)
- Predicted intercept point updates smoothly
- Time-to-intercept decreases as ball approaches

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Ball not detected | Tune HSV/LAB thresholds with test_single_camera.py |
| Z depth is wrong | Recalibrate cameras, verify checkerboard square size |
| Cameras won't open | Check USB connections, try different camera IDs |
| Low frame rate | Ensure MJPG codec is active, use DirectShow backend |
| Erratic predictions | Check calibration accuracy, increase min_points |
| Ball detected in only one camera | Adjust thresholds separately for each camera |

---

## Key Parameters

| Parameter | Location | Description |
|-----------|----------|-------------|
| `checkerboard_box_size_scale` | calibration_settings.yaml | Square size in cm (e.g., 3.17) |
| `frame_width/height` | stereo_config.yaml | Camera resolution (1280x720) |
| `camera_left/right id` | stereo_config.yaml | USB camera device IDs |
| `robot_z` | trajectory test | Z distance where robot intercepts |
| `min_points` | trajectory_predictor | Points needed before prediction (3) |
| `gravity` | physics_model | 981 cm/s² (or 9810 mm/s²) |

---

## Units

All 3D measurements use the same units as `checkerboard_box_size_scale`:
- If checkerboard squares are 3.17 **cm** → outputs are in **cm**
- If checkerboard squares are 31.7 **mm** → outputs are in **mm**

Typical table tennis table is ~274 cm long. Robot should intercept at Z ≈ 30-80 cm from cameras depending on setup.