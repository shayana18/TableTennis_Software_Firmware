# Table Tennis Robot - Stereo Vision System

Real-time stereo vision pipeline for tracking a table tennis ball in 3D and predicting its trajectory to compute where a robot should move to intercept it.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Hardware](#hardware)
3. [Software Dependencies](#software-dependencies)
4. [Project Structure](#project-structure)
5. [Configuration System](#configuration-system)
6. [Pipeline Stages](#pipeline-stages)
   - [Stage 1: Camera Calibration](#stage-1-camera-calibration)
   - [Stage 2: Ball Detection (MOG2)](#stage-2-ball-detection-mog2)
   - [Stage 3: Stereo Triangulation](#stage-3-stereo-triangulation)
   - [Stage 4: Trajectory Prediction](#stage-4-trajectory-prediction)
7. [Module Reference](#module-reference)
   - [tracking/ball_tracker.py](#trackingball_trackerpy)
   - [tracking/stereo_detector.py](#trackingstereo_detectorpy)
   - [tracking/stereo_triangulator.py](#trackingstereo_triangulatorpy)
   - [trajectory/position_buffer.py](#trajectoryposition_bufferpy)
   - [trajectory/velocity_estimator.py](#trajectoryvelocity_estimatorpy)
   - [trajectory/physics_model.py](#trajectoryphysics_modelpy)
   - [trajectory/trajectory_predictor.py](#trajectorytrajectory_predictorpy)
8. [Test Scripts](#test-scripts)
9. [Detection Experiments](#detection-experiments)
10. [Setup Workflow](#setup-workflow)
11. [Coordinate System](#coordinate-system)
12. [Units](#units)
13. [Key Parameters Reference](#key-parameters-reference)
14. [Troubleshooting](#troubleshooting)

---

## System Overview

```
+------------------------------------------------------------------------------+
|                             VISION PIPELINE                                  |
+------------------------------------------------------------------------------+
|                                                                              |
|  CAMERAS          DETECTION         TRIANGULATION        PREDICTION          |
|                                                                              |
| +-------+       +---------+        +-----------+                             |
| | Left  |------>|  MOG2   |------->|           |      +--------------+       |
| |Camera |       |  Ball   |        |    3D     |----->|  Trajectory  |       |
| +-------+       |Detector |        |Triangulate|      |  Predictor   |       |
|                 +---------+        |           |      |              |       |
| +-------+       +---------+        |  (X,Y,Z)  |      |  +--------+ |       |
| | Right |------>|  MOG2   |------->|           |      |  | Rising? | |       |
| |Camera |       |  Ball   |        +-----------+      |  +---+----+ |       |
| +-------+       |Detector |                           |    Y | N    |       |
|                 +---------+                           |  +--+ +--+  |       |
|                                                       | APEX  Z-PL  |       |
|                                                       +------+------+       |
|                                                              v              |
|                                                        +----------+         |
|                                                        |  ROBOT   |         |
|                                                        | (X,Y,T)  |         |
|                                                        +----------+         |
+------------------------------------------------------------------------------+
```

**End-to-end data flow:**

1. Two Arducam OV9782 cameras capture synchronized frames at up to 100 fps
2. MOG2 background subtraction isolates moving objects in each frame
3. Contour scoring (area, circularity, proximity, orange color boost) picks the ball
4. DLT triangulation computes the ball's 3D position from stereo pixel correspondences
5. A circular buffer stores timestamped 3D positions; linear regression estimates velocity
6. Semi-implicit Euler integration with gravity and air drag predicts future trajectory
7. Two-case interception strategy outputs the (X, Y, time) coordinates for the robot

---

## Hardware

**Cameras:** Arducam OV9782 Global Shutter USB Camera (x2)

| Property | Value |
|----------|-------|
| Sensor | OmniVision OV9782 |
| Resolution | 1280 x 800 |
| Frame Rate | 100 fps (MJPG codec) |
| Shutter | Global (no motion blur) |
| Interface | USB 2.0 / UVC |
| Trigger Mode | External trigger via `CAP_PROP_BACKLIGHT=1` |
| Max Trigger FPS | ~90 fps with MJPG |

**Why global shutter?** A rolling shutter camera scans the sensor line-by-line, causing motion blur and geometric distortion on fast-moving objects. A global shutter exposes every pixel simultaneously, giving sharp images of a ball traveling at 200-800 cm/s.

**Physical setup:**
- Both cameras are mounted side-by-side facing the playing area
- Camera 0 (left) is the world origin in the calibration coordinate system
- Baseline (distance between cameras) is ~26.4 cm (read from calibration output)

---

## Software Dependencies

```
imutils
numpy<2.0.0
opencv-python
matplotlib
scipy
pyyaml
```

Install: `pip install -r requirements.txt`

**Python version:** 3.8+ recommended (tested on 3.10/3.11)

**Platform notes:**
- On Windows, cameras default to DirectShow backend (`cv2.CAP_DSHOW`)
- On Linux/WSL, cameras use V4L2 backend
- MJPG codec is required for full frame rate

---

## Project Structure

```
TableTennis_Software_Firmware/
|-- Readme.MD                          # This file
|-- requirements.txt                   # Python dependencies
|
+-- top_level_control/                 # ALL source code lives here
    |
    +-- config/                        # Runtime configuration
    |   |-- camera_config.py           # SINGLE SOURCE OF TRUTH for camera params
    |   |-- ball_thresholds.json       # Legacy: single-camera detection thresholds
    |   +-- ball_thresholds_stereo.json# Legacy: per-camera detection thresholds
    |
    +-- camera_calibration/            # Calibration tools
    |   |-- calib.py                   # Full calibration (intrinsic + extrinsic)
    |   |-- recalib_extrinsic.py       # Re-do extrinsics only (keep intrinsics)
    |   |-- calibration_settings.yaml  # Checkerboard params, frame counts
    |   +-- camera_parameters/         # Generated calibration output (.dat files)
    |       |-- camera0_intrinsics.dat
    |       |-- camera0_intrinsics_rmse.dat
    |       |-- camera0_rot_trans.dat
    |       |-- camera1_intrinsics.dat
    |       |-- camera1_intrinsics_rmse.dat
    |       |-- camera1_rot_trans.dat
    |       +-- stereo_calibration_rmse.dat
    |
    +-- tracking/                      # Detection and triangulation module
    |   |-- __init__.py
    |   |-- ball_tracker.py            # MOG2 ball detector (core detection engine)
    |   |-- stereo_detector.py         # Stereo detection (no 3D, just 2D in both views)
    |   +-- stereo_triangulator.py     # Stereo detection + DLT 3D triangulation
    |
    +-- trajectory/                    # Trajectory prediction module
    |   |-- __init__.py
    |   |-- position_buffer.py         # Circular buffer for timestamped 3D positions
    |   |-- velocity_estimator.py      # Velocity from linear regression on positions
    |   |-- physics_model.py           # Kinematics: gravity + air drag
    |   |-- trajectory_predictor.py    # Main predictor (combines all above)
    |   +-- test_trajectory_prediction.py  # End-to-end pipeline test
    |
    +-- scripts/                       # Test and utility scripts
    |   |-- test_single_camera.py      # Single camera MOG2 detection test
    |   |-- test_stereo_detection.py   # Stereo 2D detection test
    |   +-- test_triangulation.py      # Stereo 3D triangulation test
    |
    +-- detection_experiments/         # R&D experiments (not used in pipeline)
        |-- step1_ROI.py               # ROI cropping exploration
        |-- step2_background_subtraction.py  # MOG2 parameter exploration
        |-- step3_contour_filtering.py # Contour scoring (SOURCE for ball_tracker.py)
        +-- step4_KF.py               # Kalman filter exploration (not used)
```

---

## Configuration System

### camera_config.py (Single Source of Truth)

**File:** `top_level_control/config/camera_config.py`

All camera parameters are defined as top-level constants in this file. Every script in the project imports from here. There is no YAML or JSON configuration for camera settings.

```python
CAMERA_LEFT_ID  = 0       # USB device index for left camera (camera0)
CAMERA_RIGHT_ID = 2       # USB device index for right camera (camera1)
FRAME_WIDTH     = 1280    # Horizontal resolution
FRAME_HEIGHT    = 800     # Vertical resolution
FPS             = 100     # Target framerate
FOURCC          = 'MJPG'  # Video codec (required for full fps)
TRIGGER_MODE    = True    # Hardware sync via external trigger signal
```

**Key functions:**

| Function | Purpose |
|----------|---------|
| `load_camera_settings()` | Returns a dict of camera settings (backward compatibility) |
| `configure_camera(cap, width, height)` | Applies codec/resolution/fps/trigger to a `cv2.VideoCapture` object |

`configure_camera()` sets MJPG codec, resolution, FPS, manual exposure, and optionally external trigger mode. It returns a dict with the actual accepted values so you can verify the camera accepted your settings.

### calibration_settings.yaml

**File:** `top_level_control/camera_calibration/calibration_settings.yaml`

Only contains calibration-specific parameters (not camera settings):

```yaml
mono_calibration_frames: 15       # Frames to capture per camera for intrinsics
stereo_calibration_frames: 15     # Frames to capture for stereo extrinsics
view_resize: 2                    # Display downscale factor during calibration
checkerboard_box_size_scale: 3.17 # Checkerboard square size in cm
checkerboard_rows: 4              # Inner corners (rows)
checkerboard_columns: 7           # Inner corners (columns)
cooldown: 100                     # Frames between captures
```

The `checkerboard_box_size_scale` value determines the unit system for **all** 3D measurements in the entire pipeline. If your squares are 3.17 cm, all outputs (X, Y, Z positions, velocities, interception coordinates) are in centimeters.

### Legacy Threshold Files

`ball_thresholds.json` and `ball_thresholds_stereo.json` were used by the old HSV+LAB color-space detection engine. They still exist but are **not used** by the current MOG2 detection engine. The `load_thresholds()` methods in `StereoTriangulator` and `StereoDetector` call no-op setters, so loading them won't crash anything, but they have no effect.

---

## Pipeline Stages

### Stage 1: Camera Calibration

**Purpose:** Determine each camera's internal lens properties (intrinsics) and the spatial relationship between the two cameras (extrinsics). This is a prerequisite for 3D triangulation.

#### What is Calibration?

Camera calibration solves two problems:

1. **Intrinsics** (per camera): What is the focal length? Where is the optical center? What lens distortion exists? These are encoded in a 3x3 camera matrix and a 5-element distortion vector.

2. **Extrinsics** (stereo relationship): Where is camera 1 relative to camera 0? This is encoded as a 3x3 rotation matrix R and a 3x1 translation vector T.

Together, intrinsics and extrinsics form a **projection matrix** P = K @ [R|T] that maps any 3D world point to a 2D pixel coordinate.

#### Calibration Scripts

**Full calibration:** `calib.py`
```bash
cd top_level_control/camera_calibration
python calib.py calibration_settings.yaml
```

This runs three phases:
1. **Camera 0 intrinsics** — Show checkerboard to left camera from various angles (15 frames)
2. **Camera 1 intrinsics** — Same for right camera (15 frames)
3. **Stereo extrinsics** — Show checkerboard visible to BOTH cameras simultaneously (15 frames)

**Extrinsic-only recalibration:** `recalib_extrinsic.py`
```bash
cd top_level_control/camera_calibration
python recalib_extrinsic.py calibration_settings.yaml
```

Use this when you've moved the cameras but the individual lens properties haven't changed. It loads the existing intrinsics and only recaptures the stereo frames.

#### Calibration Output Files

All output is saved to `camera_calibration/camera_parameters/`:

| File | Contents |
|------|----------|
| `camera0_intrinsics.dat` | Camera matrix (3x3) + distortion coefficients (1x5) |
| `camera1_intrinsics.dat` | Same for right camera |
| `camera0_rot_trans.dat` | R = identity (3x3), T = [0,0,0] (left camera is world origin) |
| `camera1_rot_trans.dat` | R (3x3 rotation) + T (3x1 translation) relative to camera 0 |
| `camera0_intrinsics_rmse.dat` | Reprojection error for camera 0 calibration |
| `camera1_intrinsics_rmse.dat` | Reprojection error for camera 1 calibration |
| `stereo_calibration_rmse.dat` | Stereo calibration RMSE |

**Intrinsics `.dat` format:**
```
intrinsic:
903.51  0.0     662.65
0.0     905.24  417.56
0.0     0.0     1.0
distortion:
-0.416  0.289  0.001  -0.000  -0.148
```

**Extrinsics `.dat` format:**
```
R:
0.992  -0.012   0.127
0.004   0.998   0.060
-0.128  -0.059   0.990
T:
-26.409
0.174
2.205
```

**Quality targets:**
- RMSE < 0.3: Excellent
- RMSE < 0.5: Good
- RMSE < 1.0: Acceptable
- RMSE > 1.0: Redo calibration with better checkerboard coverage

---

### Stage 2: Ball Detection (MOG2)

**Purpose:** Find the ball's pixel (x, y) position in each camera frame.

**Engine:** MOG2 (Mixture of Gaussians v2) background subtraction + contour scoring. This replaces the older HSV+LAB color-space fusion approach and is **lighting-invariant** — no manual threshold tuning required.

#### How It Works

```
Frame (BGR)
    |
    v
MOG2 Background Subtractor
    |  history=300, varThreshold=40, detectShadows=False
    |  learningRate=0.05 (warmup), 0.002 (runtime)
    v
Foreground Mask (binary)
    |
    v
Morphological Cleanup
    |  MORPH_OPEN (3x3 ellipse) - remove noise
    |  MORPH_CLOSE (7x7 ellipse) - fill gaps
    v
Clean Foreground Mask
    |
    v
findContours (RETR_EXTERNAL)
    |
    v
Filter: area 150-1100 px, circularity > 0.45
    |
    v
Score each candidate:
    |  circularity * 0.4
    |  + proximity to last known position (up to 0.3)
    |  + orange color boost (0.3 if HSV hue 5-25, sat>80, val>80)
    |  + no-history bonus (0.15 if no previous position)
    v
Best candidate -> center (x,y), radius, confidence
```

#### Warmup Phase

MOG2 needs to build a background model before it can detect foreground objects. The tracker processes the first **300 frames** at a high learning rate (0.05) before switching to runtime mode (0.002). During warmup, `detect()` returns `found=False` and all test scripts display a "Warming up... XX%" overlay.

#### Circularity Formula

```
circularity = 4 * pi * area / (perimeter^2)
```

A perfect circle has circularity = 1.0. The minimum threshold is 0.45, which allows slightly irregular contours (the ball may be partially occluded or noisy).

#### Orange Color Boost

Even though the primary detection is motion-based, the scorer gives a 0.3 bonus to contours whose mean color falls in the orange range:
- HSV Hue: 5-25
- HSV Saturation: > 80
- HSV Value: > 80

This helps disambiguate the ball from other moving objects (e.g., a hand).

#### detect() Return Contract

Every consumer in the pipeline relies on this exact return dict:

```python
{
    'found': bool,           # True if ball was detected
    'center': (int, int),    # Pixel (x, y) or None
    'radius': float,         # Enclosing circle radius in px, or None
    'contour': np.array,     # Raw OpenCV contour, or None
    'confidence': float,     # 0.0 to 1.0 (= min(1.0, score))
    'mask': np.array         # Binary foreground mask (full resolution)
}
```

When `return_debug=True`, an additional key is included:
```python
    'debug': {'fg_mask': np.array}  # Raw MOG2 foreground mask
```

---

### Stage 3: Stereo Triangulation

**Purpose:** Compute the ball's real-world 3D position (X, Y, Z) from its pixel coordinates in both cameras.

#### Algorithm: Direct Linear Transform (DLT)

Given projection matrices P0 and P1 (computed from calibration), and pixel coordinates (u0, v0) and (u1, v1), the DLT constructs a 4x4 system:

```
A = [ v0 * P0[2,:] - P0[1,:]  ]
    [ P0[0,:] - u0 * P0[2,:]  ]
    [ v1 * P1[2,:] - P1[1,:]  ]
    [ P1[0,:] - u1 * P1[2,:]  ]
```

Solve via SVD: `_, _, Vh = np.linalg.svd(A)` → The last row of Vh, dehomogenized, gives (X, Y, Z).

#### Disparity Check

Before triangulating, the system checks that `disparity = left_x - right_x > 0`. If the ball appears further right in the right camera than in the left camera, it's behind the cameras and triangulation is skipped.

Additionally, the triangulated Z must be positive (ball in front of cameras).

#### update() Return Contract

```python
{
    'left_frame': np.array,       # Raw BGR frame from left camera
    'right_frame': np.array,      # Raw BGR frame from right camera
    'left_detection': dict,       # detect() result from left tracker
    'right_detection': dict,      # detect() result from right tracker
    'found_3d': bool,             # True if 3D position computed
    'position_3d': (X, Y, Z),     # 3D coordinates in calibration units (cm)
    'disparity': float            # Pixel disparity (left_x - right_x)
}
```

---

### Stage 4: Trajectory Prediction

**Purpose:** Given a stream of 3D positions, predict the ball's future trajectory and compute where and when the robot should intercept it.

The trajectory module has four components:

```
position_buffer.py  -->  velocity_estimator.py  -->  trajectory_predictor.py
                                                              |
                                                       physics_model.py
```

#### Step 4a: Position Buffer

A circular buffer (deque, max 10 entries) storing `{x, y, z, timestamp}` dicts. Timestamps use `time.perf_counter()` for sub-millisecond precision. Oldest entries are automatically discarded when the buffer is full.

#### Step 4b: Velocity Estimation

**Method:** Linear regression (`numpy.polyfit` degree 1) on each axis independently.

```
For each axis (X, Y, Z):
    velocity = slope of best-fit line through (time, position) pairs
```

The regression method is more noise-robust than simple `(last - first) / dt`, especially when individual triangulation measurements have small errors.

**Important — Vy bias correction:** Linear regression on parabolic Y data (Y is affected by gravity) gives the velocity at the **buffer midpoint**, not at the latest timestamp. The predictor corrects this:

```python
Vy_corrected = Vy_regression + gravity * (t_latest - t_mean)
```

This ensures the predicted trajectory starts with the correct instantaneous velocity.

#### Step 4c: Physics Model

Semi-implicit Euler integration at 1ms timesteps:

```
For each timestep dt:
    speed = sqrt(vx^2 + vy^2 + vz^2)
    drag  = -k * speed                      # deceleration factor

    ax = drag * vx                           # X drag only
    ay = drag * vy + gravity_sign * gravity  # Y drag + gravity
    az = drag * vz                           # Z drag only

    vx += ax * dt    (velocity updated first)
    vy += ay * dt
    vz += az * dt

    x += vx * dt     (then position updated with new velocity)
    y += vy * dt
    z += vz * dt
```

**Physical constants (CGS for cm/s system):**

| Constant | Value | Description |
|----------|-------|-------------|
| Gravity | 981.0 cm/s^2 | Standard gravitational acceleration |
| Ball mass | 2.7 g | Table tennis ball |
| Ball radius | 2.0 cm | Table tennis ball |
| Air density | 0.001225 g/cm^3 | Sea level |
| Drag coefficient Cd | 0.45 | At typical playing speeds |
| Drag constant k | ~0.00128 cm^-1 | `(rho * Cd * A) / (2 * m)` |

At typical speeds (~300 cm/s), drag is ~12% of gravity — enough to reduce prediction overshoot by 10-15%.

When drag is disabled (`enable_drag=False`), the model falls back to exact analytical kinematics (no numerical integration needed):
```
X(t) = X0 + Vx * t
Y(t) = Y0 + Vy * t + 0.5 * g * t^2
Z(t) = Z0 + Vz * t
```

#### Step 4d: Two-Case Interception Strategy

| Condition | Strategy | Intercept Point | Use Case |
|-----------|----------|-----------------|----------|
| Vy < 0 (ball rising) | **APEX** | Trajectory peak where Vy = 0 | Tossed/lobbed balls |
| Vy >= 0 (ball flat/falling) | **Z-PLANE** | Fixed Z distance where robot can reach | Flat/descending shots |

- **Apex** catches lobbed balls at their highest point, giving the robot maximum time to position itself.
- **Z-Plane** intercepts fast, flat shots at a configurable Z distance (`robot_z`, default 50 cm).

If the apex strategy fails (e.g., ball is not actually rising), the system automatically falls back to Z-plane.

#### predict() Return Contract

```python
{
    'valid': bool,                    # True if prediction succeeded
    'intercept_x': float,            # X position at interception (cm)
    'intercept_y': float,            # Y position at interception (cm)
    'intercept_z': float,            # Z position at interception (cm)
    'time_to_intercept': float,      # Seconds until ball arrives
    'velocity_at_intercept': tuple,  # (Vx, Vy, Vz) at interception
    'current_position': tuple,       # Current (X, Y, Z)
    'current_velocity': tuple,       # Current (Vx, Vy, Vz) with Vy corrected
    'strategy': str                  # 'apex' or 'z_plane'
}
```

---

## Module Reference

### tracking/ball_tracker.py

**Class:** `EnhancedBallTracker`

The core detection engine. Uses MOG2 background subtraction to detect moving objects and contour scoring to identify the ball.

| Method | Description |
|--------|-------------|
| `detect(frame, return_debug=False)` | Detect ball in a BGR frame. Returns standard result dict. |
| `find_best_ball(mask, frame)` | Score contour candidates and return the best one. |
| `reset_background()` | Recreate MOG2 subtractor, reset warmup and tracking state. |
| `is_ready()` | Returns `True` once 300-frame warmup is complete. |
| `get_warmup_progress()` | Returns 0.0 to 1.0 warmup progress. |
| `set_hsv_thresholds(lower, upper)` | No-op (legacy compatibility). |
| `set_lab_thresholds(lower, upper)` | No-op (legacy compatibility). |
| `set_fusion_weight(hsv_weight)` | No-op (legacy compatibility). |

**Alias:** `BallTracker = EnhancedBallTracker` for backward compatibility.

### tracking/stereo_detector.py

**Class:** `StereoDetector`

Detects the ball in both cameras independently (no 3D triangulation). Used by `test_stereo_detection.py`.

| Method | Description |
|--------|-------------|
| `start_cameras(width, height)` | Open and configure both cameras via `configure_camera()`. |
| `stop_cameras()` | Release both camera streams. |
| `detect()` | Capture frames and detect ball in both cameras. |
| `detect_from_frames(left, right)` | Detect from pre-captured frames. |
| `draw_detections(result)` | Annotate frames with detection circles. |
| `reset_background()` | Reset MOG2 on both trackers. |
| `warmup_status()` | Returns `{left_ready, right_ready, left_progress, right_progress}`. |

### tracking/stereo_triangulator.py

**Class:** `StereoTriangulator`

The full stereo pipeline: loads calibration, detects in both cameras, triangulates to 3D. Used by all triangulation and prediction test scripts.

| Method | Description |
|--------|-------------|
| `__init__(calibration_dir, cam_left_id, cam_right_id)` | Load calibration .dat files, create trackers. |
| `start_cameras(width, height)` | Open and configure cameras. |
| `stop_cameras()` | Release cameras. |
| `update()` | Capture, detect, triangulate. Returns full result dict. |
| `triangulate(point_left, point_right)` | DLT triangulation from pixel coordinates. Returns (X,Y,Z). |
| `draw_results(result)` | Annotate frames with 3D position overlay. |
| `load_thresholds(filepath)` | Load JSON thresholds (calls no-op setters). |
| `reset_background()` | Reset MOG2 on both trackers. |
| `warmup_status()` | Returns warmup progress for both trackers. |
| `get_baseline()` | Stereo baseline distance in calibration units. |
| `get_focal_length_px()` | Average focal length in pixels. |

### trajectory/position_buffer.py

**Class:** `PositionBuffer`

Circular buffer (deque) storing `{x, y, z, t}` dicts. Max default size: 10 entries.

| Method | Description |
|--------|-------------|
| `add(x, y, z, timestamp=None)` | Add position (auto-timestamps with `perf_counter`). |
| `get_recent(n=None)` | Get last N positions as list of dicts. |
| `get_as_arrays(n=None)` | Get as numpy arrays: `(positions[N,3], timestamps[N])`. |
| `get_latest()` | Most recent entry dict. |
| `is_ready(min_points)` | True if buffer has >= min_points entries. |
| `get_time_span()` | Seconds between oldest and newest entries. |
| `get_average_dt()` | Average time between consecutive samples. |
| `clear()` | Empty the buffer. |

### trajectory/velocity_estimator.py

**Class:** `VelocityEstimator`

Computes (Vx, Vy, Vz) velocity vector from position history.

| Method | Description |
|--------|-------------|
| `estimate(positions, timestamps)` | Returns `{vx, vy, vz, speed, valid}` dict. |
| `estimate_regression(positions, timestamps)` | Linear regression fit (default, noise-robust). |
| `estimate_simple(positions, timestamps)` | Simple (last-first)/dt (fast but noisy). |
| `estimate_from_buffer(position_buffer)` | Convenience: estimate directly from PositionBuffer. |

### trajectory/physics_model.py

**Class:** `PhysicsModel`

Kinematic prediction with gravity and optional air drag.

| Method | Description |
|--------|-------------|
| `predict_position(position, velocity, dt)` | Position after dt seconds. |
| `predict_velocity(velocity, dt)` | Velocity after dt seconds. |
| `predict_trajectory(position, velocity, duration, dt)` | Full trajectory as list of (X,Y,Z,t). |
| `position_at_z(position, velocity, target_z)` | Position/time when ball reaches target Z. |
| `position_at_apex(position, velocity)` | Position/time at trajectory peak (Vy=0). |
| `time_to_z(position, velocity, target_z)` | Time to reach target Z (analytical, no drag). |

### trajectory/trajectory_predictor.py

**Class:** `TrajectoryPredictor`

The main prediction class that combines all trajectory components.

```python
predictor = TrajectoryPredictor(
    buffer_size=10,           # Circular buffer capacity
    min_points=3,             # Minimum points before predicting
    velocity_method='regression',  # 'simple' or 'regression'
    gravity=981.0,            # cm/s^2
    y_down=True,              # +Y is downward in camera coords
    enable_drag=True          # Include air resistance
)
```

| Method | Description |
|--------|-------------|
| `add_position(x, y, z, timestamp=None)` | Add new triangulated 3D position. Call every frame. |
| `predict(target_z)` | Predict interception point. Returns full result dict. |
| `predict_trajectory(duration, dt)` | Predict full trajectory for visualization. |
| `get_velocity()` | Current velocity estimate `{vx, vy, vz, speed, valid}`. |
| `get_current_position()` | Most recent (X, Y, Z). |
| `is_ready()` | True if enough data to predict. |
| `reset()` | Clear all history. |
| `get_stats()` | Buffer size, time span, velocity info. |

---

## Test Scripts

All scripts are run from the `top_level_control/` directory.

### test_single_camera.py

**Purpose:** Test MOG2 ball detection on a single camera.

```bash
cd top_level_control
python scripts/test_single_camera.py
```

**Controls:**
| Key | Action |
|-----|--------|
| `q` | Quit |
| `d` | Toggle debug view (foreground mask) |
| `b` | Reset background model (re-warmup) |
| `c` | Cycle camera ID (0, 1, 2...) |

**What to verify:**
- Warmup progress bar appears for ~3 seconds
- Green circle tracks the ball after warmup
- Debug view shows clean foreground mask with ball blob
- No false detections on static objects after warmup

### test_stereo_detection.py

**Purpose:** Test ball detection on both cameras simultaneously. No 3D triangulation.

```bash
cd top_level_control
python scripts/test_stereo_detection.py
```

**Controls:**
| Key | Action |
|-----|--------|
| `q` | Quit |
| `l` | Cycle lighting normalization mode (OFF/CLAHE/White Balance/etc.) |
| `d` | Toggle debug view (foreground masks for both cameras) |
| `b` | Reset background model |

**What to verify:**
- Both cameras detect the ball independently
- "BOTH DETECTED" status appears when ball is visible to both cameras
- Detection is consistent across different lighting modes

### test_triangulation.py

**Purpose:** Test stereo calibration accuracy by triangulating 3D positions.

```bash
cd top_level_control
python scripts/test_triangulation.py
```

**Controls:**
| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save current 3D measurement |
| `r` | Reset/clear all saved measurements |
| `d` | Toggle debug view |
| `b` | Reset background model |

**What to verify:**
- Hold ball at known distances (measure with tape measure)
- Z value should match the real-world distance within 5%
- Save multiple measurements at the same distance — the range should be small
- Measurement summary printed on exit

### test_trajectory_prediction.py (End-to-End Pipeline)

**Purpose:** The full pipeline: cameras -> detection -> triangulation -> prediction -> visualization. This is the closest to what the robot controller will use.

```bash
cd top_level_control
python -m trajectory.test_trajectory_prediction
```

**Controls:**
| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Reset predictor (clear position history) |
| `v` | Toggle velocity display |
| `t` | Toggle trajectory visualization |
| `p` | Print current stats |
| `z` | Increase robot Z plane by 10 cm |
| `x` | Decrease robot Z plane by 10 cm |
| `b` | Reset background model + predictor |

**Visualization:**
- Green circle: Detected ball in camera view
- Blue line: Predicted trajectory arc
- Red X + circle: Predicted interception point
- OSD text: `[APEX]` or `[Z-PLANE]` with X, Y, Z coordinates and time in ms

**What to verify:**
- Toss ball upward: OSD shows `[APEX]` with intercept at trajectory peak
- Throw ball flat/downward: OSD shows `[Z-PLANE]` with intercept at robot Z distance
- Velocity values are reasonable (200-800 cm/s for typical throws)
- Time-to-intercept decreases as ball approaches
- Predicted intercept point updates smoothly (no wild jumps)

**Console output format:**
```
Pos:(  X,  Y,  Z) Vel:(Vx, Vy, Vz) [APEX] Intercept:(X, Y, Z) in  XXms
```

---

## Detection Experiments

The `detection_experiments/` folder contains R&D scripts that were used to develop and validate the MOG2 detection approach. They are **not part of the production pipeline** but document the reasoning behind design decisions.

| Script | What It Explores |
|--------|-----------------|
| `step1_ROI.py` | Region of Interest cropping to reduce processing area |
| `step2_background_subtraction.py` | MOG2 parameter tuning (history, varThreshold, learning rates) |
| `step3_contour_filtering.py` | Contour scoring algorithm (area, circularity, proximity, color). **This is the source code that was ported into `ball_tracker.py`.** |
| `step4_KF.py` | Kalman filter for smoothing detections. Explored but **not used** in the final pipeline. |

If you want to understand why specific parameter values were chosen (e.g., why `min_area=150`, why `circularity > 0.45`), read `step3_contour_filtering.py`.

---

## Setup Workflow

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Connect Cameras

- Plug in both Arducam OV9782 cameras via USB
- Note their device IDs (typically 0 and 1 or 0 and 2)
- Update `CAMERA_LEFT_ID` and `CAMERA_RIGHT_ID` in `config/camera_config.py`

### 3. Verify Single Camera

```bash
cd top_level_control
python scripts/test_single_camera.py
```

Wait for warmup, wave ball in front of camera. Press `c` to cycle camera IDs if needed.

### 4. Verify Stereo Detection

```bash
python scripts/test_stereo_detection.py
```

Confirm both cameras detect the ball. Press `d` to verify foreground masks look clean.

### 5. Calibrate Cameras

Print a checkerboard pattern (4x7 inner corners, 3.17 cm squares or your measured size).

```bash
cd camera_calibration
python calib.py calibration_settings.yaml
```

Follow the prompts:
1. Show checkerboard to **left camera** from different angles (15 frames)
2. Show checkerboard to **right camera** from different angles (15 frames)
3. Show checkerboard visible to **both cameras** simultaneously (15 frames)

Check the RMSE values printed at the end. If stereo RMSE > 1.0, redo with better coverage.

### 6. Verify Triangulation

```bash
cd ..
python scripts/test_triangulation.py
```

Hold the ball at a known distance (e.g., 50 cm from cameras, measured with tape). The displayed Z value should match within 5%.

### 7. Run Full Pipeline

```bash
python -m trajectory.test_trajectory_prediction
```

Toss and throw the ball. Verify:
- `[APEX]` appears for upward tosses
- `[Z-PLANE]` appears for flat/descending throws
- Interception coordinates are physically reasonable
- Adjust `robot_z` with `z`/`x` keys to match your robot's reach

---

## Coordinate System

```
        Y (down in image = physically down)
        |
        |
        |
Left ---+-------- Right
Camera  |         Camera
(origin)|
        |
        +----------> X (right)
       /
      /
     Z (depth, toward ball)
```

**Conventions:**
- **Origin:** Left camera (camera 0)
- **+X:** Right (from left camera's perspective)
- **+Y:** Down (standard camera image coordinates)
- **+Z:** Depth (away from cameras, toward the ball/table)
- **Ball going physically up** = Y decreasing = Vy < 0
- **Gravity pulls ball down** = acts in +Y direction = `gravity_sign = +1`
- Units: Same as `checkerboard_box_size_scale` (cm if squares are measured in cm)

---

## Units

All 3D measurements use the same units as `checkerboard_box_size_scale` in `calibration_settings.yaml`:

| If squares are... | Then all outputs are in... |
|-------------------|---------------------------|
| 3.17 cm | Centimeters |
| 31.7 mm | Millimeters |
| 0.0317 m | Meters |

This applies to: positions (X, Y, Z), velocities (cm/s), interception coordinates, baseline, etc.

Typical table tennis table is ~274 cm long. A `robot_z` of 30-80 cm is typical depending on your setup geometry.

---

## Key Parameters Reference

| Parameter | Value | Location | Description |
|-----------|-------|----------|-------------|
| `CAMERA_LEFT_ID` | 0 | `camera_config.py` | USB device index for left camera |
| `CAMERA_RIGHT_ID` | 2 | `camera_config.py` | USB device index for right camera |
| `FRAME_WIDTH` | 1280 | `camera_config.py` | Horizontal resolution |
| `FRAME_HEIGHT` | 800 | `camera_config.py` | Vertical resolution |
| `FPS` | 100 | `camera_config.py` | Target frame rate |
| `FOURCC` | MJPG | `camera_config.py` | Video codec |
| `TRIGGER_MODE` | True | `camera_config.py` | External trigger sync |
| `checkerboard_box_size_scale` | 3.17 | `calibration_settings.yaml` | Square size in cm |
| `checkerboard_rows` | 4 | `calibration_settings.yaml` | Inner corners (rows) |
| `checkerboard_columns` | 7 | `calibration_settings.yaml` | Inner corners (columns) |
| `history` | 300 | `ball_tracker.py` | MOG2 background model history |
| `varThreshold` | 40 | `ball_tracker.py` | MOG2 variance threshold |
| `_warmup_frames` | 300 | `ball_tracker.py` | Frames before detection starts |
| `min_area` | 150 px | `ball_tracker.py` | Minimum contour area |
| `max_area` | 1100 px | `ball_tracker.py` | Maximum contour area |
| `min_circularity` | 0.45 | `ball_tracker.py` | Minimum circularity threshold |
| `search_radius` | 150 px | `ball_tracker.py` | Proximity search radius |
| `buffer_size` | 10 | `trajectory_predictor.py` | Position history buffer size |
| `min_points` | 3 | `trajectory_predictor.py` | Points needed before prediction |
| `gravity` | 981.0 | `physics_model.py` | cm/s^2 |
| `drag_coefficient` | 0.45 | `physics_model.py` | Air drag Cd |
| `robot_z` | 50.0 | `test_trajectory_prediction.py` | Z-plane interception distance (cm) |

---

## Troubleshooting

| Problem | Likely Cause | Solution |
|---------|-------------|----------|
| Ball not detected | Background not built | Wait for warmup (~3s at 100fps). Check progress overlay. |
| Ball not detected after warmup | Ball too small/large | Adjust `min_area`/`max_area` in `ball_tracker.py` for your ball size and camera distance. |
| False detections | Other moving objects | Orange color boost should prefer the ball. Reduce `max_area` to reject large objects. |
| Z depth is wrong | Bad calibration | Redo calibration with better checkerboard coverage. Verify `checkerboard_box_size_scale`. |
| Cameras won't open | Wrong device IDs | Run `test_single_camera.py`, press `c` to find working camera ID. Update `camera_config.py`. |
| Low frame rate (<30 fps) | Wrong codec | Verify MJPG codec is set. Check `configure_camera()` output for actual FPS. |
| Erratic predictions | Noisy triangulation | Check calibration RMSE. Increase `min_points` to require more data before predicting. |
| "Warming up" never finishes | Camera not providing frames | Check USB connection. Verify camera opens with `test_single_camera.py`. |
| Background model confused | Scene changed | Press `b` to reset background model in any test script. |
| Detection breaks after moving camera | Stale background model | Press `b` to reset. The MOG2 model is camera-position-specific. |
| Trajectory shows wrong direction | Coordinate system mismatch | Verify `y_down=True` in predictor config. Check that camera 0 is actually the left camera. |
| Intercept point wildly wrong | Bad velocity estimate | Check that buffer has enough points (`min_points=3`). Verify position data isn't jumping. |

---

## Extending the System

### Adding Robot Communication

The pipeline outputs interception coordinates via `predict()`. To send to a robot:

```python
from trajectory import TrajectoryPredictor
from tracking.stereo_triangulator import StereoTriangulator

triangulator = StereoTriangulator(calibration_dir='camera_calibration/camera_parameters')
triangulator.start_cameras()

predictor = TrajectoryPredictor(gravity=981.0, y_down=True, enable_drag=True)

while True:
    result = triangulator.update()

    if result['found_3d']:
        predictor.add_position(*result['position_3d'])

    pred = predictor.predict(target_z=50)

    if pred['valid']:
        # These are the values to send to the robot controller:
        robot_x = pred['intercept_x']       # cm, horizontal position
        robot_y = pred['intercept_y']       # cm, vertical position
        time_ms = pred['time_to_intercept'] * 1000  # milliseconds until arrival
        strategy = pred['strategy']          # 'apex' or 'z_plane'

        # Example: send via serial
        # serial_port.write(f"{robot_x:.1f},{robot_y:.1f},{time_ms:.0f}\n".encode())
```

### Tuning Detection for Different Balls

If using a different colored ball, modify the orange color boost in `ball_tracker.py`:

```python
# In find_best_ball(), change the HSV range:
if 5 <= hue <= 25 and sat > 80 and val > 80:  # Orange
    score += self.w_color
```

For a white ball: `if sat < 30 and val > 200`
For a yellow ball: `if 25 <= hue <= 35 and sat > 80 and val > 80`

### Adjusting for Different Camera Distances

If the ball appears very small or very large in the frame, adjust the area filters:

```python
# In EnhancedBallTracker.__init__():
self.min_area = 150    # Decrease for distant cameras
self.max_area = 1100   # Increase for close cameras
```

### Improving Prediction Accuracy

1. **Better calibration:** Lower RMSE = less noise in 3D positions = smoother velocity estimates
2. **More buffer points:** Increase `buffer_size` and `min_points` for more stable velocity (at the cost of latency)
3. **Adjust drag:** Set `enable_drag=False` if your playing distance is very short (<50 cm) and drag is negligible
4. **Custom physics:** Extend `PhysicsModel._simulate()` to add spin effects (Magnus force) if needed
