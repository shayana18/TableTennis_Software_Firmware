# Stereo Vision + Trajectory Prediction Pipeline

Complete documentation of the ball tracking and interception prediction system,
from camera calibration through to the final robot intercept command.

---

## 1. System Overview

The system uses two synchronized cameras to detect a table tennis ball in real time,
triangulate its 3D position, and predict where and when it will arrive at the robot's
end of the table.

```
  Cameras (OV9782 x2)
        │
        ▼
  Ball Detection (MOG2 + contour scoring, per camera)
        │
        ▼
  Undistortion (lens correction on detected pixel coords)
        │
        ▼
  Stereo Triangulation (DLT → 3D point in cm)
        │
        ▼
  Outlier Rejection (jump / speed filters)
        │
        ▼
  Position Buffer (circular buffer, last 15 points)
        │
        ▼
  Velocity Estimation (linear regression on buffer)
        │
        ▼
  Vy Midpoint Correction (compensate for gravity bias in regression)
        │
        ▼
  Physics Simulation (gravity + optional drag + bounce)
        │
        ▼
  X-Plane Interception (where ball crosses robot endline)
        │
        ▼
  Camera → Robot Coordinate Transform
        │
        ▼
  Workspace Check (elliptical boundary)
        │
        ▼
  Robot Command (robot_x, robot_y, robot_z in mm + time)
```

### Physical Setup

- Two Arducam OV9782 global-shutter USB cameras mounted on the **side** of the table,
  looking **across** the table width.
- An external trigger signal (Arduino or similar) fires both cameras simultaneously
  via the UVC backlight-compensation register.
- Camera 0 (left) defines the world-space origin. Camera 1 (right) is offset
  primarily in X (~26 cm baseline).

---

## 2. Camera Hardware and Configuration

**Sensor:** Arducam OV9782 — 1MP global shutter, USB UVC
**Resolution:** 640 x 480
**Framerate:** 100 fps (target)
**Codec:** MJPG (required for full framerate over USB)
**Trigger mode:** External hardware trigger via `CAP_PROP_BACKLIGHT = 1`

### Central config: `config/camera_config.py`

This is the **single source of truth** for runtime camera parameters:

| Constant          | Value   |
|-------------------|---------|
| `CAMERA_LEFT_ID`  | 1       |
| `CAMERA_RIGHT_ID` | 2       |
| `FRAME_WIDTH`     | 640     |
| `FRAME_HEIGHT`    | 480     |
| `FPS`             | 100     |
| `FOURCC`          | `MJPG`  |
| `TRIGGER_MODE`    | `True`  |

**Key functions:**

- `configure_camera(cap, width, height, trigger_mode)` — applies codec, resolution,
  fps, manual exposure, and trigger mode to an opened `VideoCapture`. Retries up to
  3 times (ArduCam on Windows/DirectShow sometimes needs a dummy read).
- `load_camera_settings()` — returns a dict of the constants above (backward compat).

### Trigger mode details

Setting `CAP_PROP_BACKLIGHT = 1` puts the OV9782 into external-trigger mode.
Each rising edge on the trigger line captures one frame in both cameras simultaneously.
Manual exposure is forced (`CAP_PROP_AUTO_EXPOSURE = 1` on DirectShow,
`0.25` on V4L2). Exposure value `-6` controls the effective trigger frame rate
(max ~90 fps with MJPG).

### Synchronized capture

All stereo scripts use the **grab/retrieve** pattern:

```python
cap_left.grab()    # both grab from same trigger pulse
cap_right.grab()
_, frame_left  = cap_left.retrieve()
_, frame_right = cap_right.retrieve()
```

This ensures both frames correspond to the same trigger pulse.
`CAP_PROP_BUFFERSIZE = 1` keeps the driver buffer minimal.

**Every** stereo file enforces this pattern — `StereoDetector.read_frames()`,
`StereoTriangulator.update()`, `test_stereo_feed.py`, `test_stereo_detection.py`,
`stereo_detection.py` (experiment), and `test_trajectory_prediction.py` (calibration
capture). Sequential `read()` calls are never used for stereo pairs because they can
return frames from different trigger pulses.

---

## 3. Calibration Pipeline

Calibration is performed in `camera_calibration/` and produces `.dat` parameter files.

### Settings: `calibration_settings.yaml`

```yaml
camera0: 1
camera1: 2
frame_width: 640
frame_height: 480
mono_calibration_frames: 15
stereo_calibration_frames: 15
view_resize: 2
checkerboard_box_size_scale: 3.17    # cm per square
checkerboard_rows: 4
checkerboard_columns: 7
cooldown: 150
```

> **Note:** `camera0/camera1` and `frame_width/frame_height` in the YAML must be
> kept in sync with `camera_config.py` manually. The calibration scripts read from
> the YAML, not from `camera_config.py`.

### Step 1 — Intrinsic Calibration (`calib.py`)

For each camera independently:

1. **Capture frames** — `save_frames_single_camera()` opens the camera and saves
   `mono_calibration_frames` (15) images to `frames/`, with a cooldown of 150
   frames between captures.
2. **Find corners** — `findChessboardCorners()` on a 4x7 checkerboard pattern,
   refined with `cornerSubPix()` (11x11 search window). User can press `s` to skip
   bad detections.
3. **Calibrate** — `cv2.calibrateCamera()` produces the 3x3 intrinsic matrix `K`
   and 5-coefficient distortion vector `[k1, k2, p1, p2, k3]`.
4. **Save** — Written to `camera_parameters/camera{N}_intrinsics.dat` in plain text:
   ```
   intrinsic:
   fx  0   cx
   0   fy  cy
   0   0   1
   distortion:
   k1 k2 p1 p2 k3
   ```

### Step 2 — Stereo Extrinsic Calibration (`calib.py`)

1. **Capture paired frames** — `save_frames_two_cams()` opens both cameras and saves
   `stereo_calibration_frames` (15) synchronized frame pairs to `frames_pair/`.
2. **Stereo calibrate** — `cv2.stereoCalibrate()` with `CALIB_FIX_INTRINSIC` (intrinsics
   are locked). Finds the rotation `R` and translation `T` from camera0 to camera1.
3. **World origin** — Camera 0 is set as the world origin:
   - `R0 = I` (3x3 identity), `T0 = [0, 0, 0]`
   - `R1 = R`, `T1 = T` (from stereoCalibrate)
4. **Save** — Written to `camera_parameters/camera{N}_rot_trans.dat`:
   ```
   R:
   r00 r01 r02
   r10 r11 r12
   r20 r21 r22
   T:
   tx
   ty
   tz
   ```
5. **Visual check** — `check_calibration()` overlays projected 3D coordinate axes
   on the live camera feeds to verify alignment.

### Extrinsic-Only Recalibration (`recalib_extrinsic.py`)

When intrinsics are already good but the cameras have been moved:
- Loads existing `camera{N}_intrinsics.dat`
- Runs only the stereo frame capture + `stereoCalibrate` steps
- Overwrites the `camera{N}_rot_trans.dat` files

### Checkerboard Units

All 3D coordinates are in the units defined by `checkerboard_box_size_scale` (currently
**3.17 cm** per checkerboard square). So `T1 = [-26.4, 0.17, 2.2]` means the right
camera is ~26.4 × 3.17 ≈ 83.7 cm to the right... **actually** the scale is baked into
the object points already, so the output units are **centimeters** directly
(each square = 3.17 cm, and the object points are `objp = scale * mgrid`, so
T is in cm). The baseline `||T1 - T0|| ≈ 26.5 cm`.

---

## 4. Ball Detection Engine

**File:** `tracking/ball_detector.py` — `BallDetector` class
**Single source of truth** for all detection logic. Every module that needs ball
detection imports from here.

### Pipeline

```
BGR frame
    │
    ├──[ROI crop if set]
    │
    ▼
MOG2 foreground mask (learningRate = 0.002 during detect, 0.05 during build_background)
    │
    ▼
Morphological cleanup:
    MORPH_OPEN  3x3 ellipse  — remove noise specks
    MORPH_CLOSE 7x7 ellipse  — fill small holes
    │
    ▼
findContours (RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
    │
    ▼
Filter: area < 5 → skip (too tiny for moments)
    │
    ▼
Compute centroid via moments (m10/m00, m01/m00)
    │
    ▼
Filter: area < min_area (80)  → rejected "SMALL"
Filter: area > max_area (2000) → rejected "BIG"
    │
    ▼
Compute circularity = 4π·area / perimeter²
Filter: circularity < min_circularity (0.35) → rejected "SHAPE"
    │
    ▼
Multi-criteria scoring (see below)
    │
    ▼
Best candidate = highest score
    │
    ▼
Update last_pos for next-frame proximity scoring
```

### MOG2 Parameters

| Parameter      | Value |
|----------------|-------|
| `history`      | 300   |
| `varThreshold` | 40    |
| `detectShadows`| False |

### Scoring Weights

| Criterion          | Weight | Computation |
|--------------------|--------|-------------|
| Circularity        | 0.40   | `circularity * 0.4` |
| Proximity          | 0.30   | `(1 - dist/search_radius) * 0.3` if within 150px of last known pos |
| No history bonus   | 0.15   | Added when `last_pos` is None (first detection) |
| Orange color       | 0.30   | Added if mean HSV of contour region is hue 5-25, sat > 80, val > 80 |

### Detection Defaults

| Parameter          | Default |
|--------------------|---------|
| `min_area`         | 80 px²  |
| `max_area`         | 2000 px²|
| `min_circularity`  | 0.35    |
| `search_radius`    | 150 px  |
| `roi`              | None (full frame) |

### Return Value

`detect()` returns `(best_candidate, all_candidates, rejected, fg_mask)` where each
candidate is a dict with keys: `center`, `area`, `circularity`, `score`, `contour`,
`is_orange`.

### Wrappers

- **`EnhancedBallTracker`** (`tracking/ball_tracker.py`) — thin wrapper that adds
  automatic warmup (300 frames at `learningRate=0.05`) and a simplified result dict
  (`found`, `center`, `radius`, `confidence`, `mask`). Used by `StereoDetector`.
- **`StereoTriangulator`** uses `BallDetector` directly (not `EnhancedBallTracker`)
  and manages its own warmup (120 frames).

> **Important:** `BallDetector.detect()` and `EnhancedBallTracker.detect()` have
> **different return formats**. `BallDetector` returns a 4-tuple where the best
> candidate is `None` or a dict with `area` (not `radius`). `EnhancedBallTracker`
> returns a single dict with `found`, `center`, `radius`. Any code consuming
> detections must match the tracker type it uses.

---

## 5. Stereo Triangulation

**File:** `tracking/stereo_triangulator.py` — `StereoTriangulator` class

### Loading Calibration

On construction, loads all four `.dat` files from the calibration directory and builds
projection matrices:

```
P = K @ [R | T]     (3x4 projection matrix)
```

where `K` is the 3x3 intrinsic matrix and `[R|T]` is the 3x4 extrinsic matrix.

### DLT (Direct Linear Transform)

Given corresponding pixel coordinates `(u_L, v_L)` and `(u_R, v_R)`, the 3D point
is found by solving the overdetermined system:

```
A = [ v_L * P0[2] - P0[1] ]
    [ P0[0] - u_L * P0[2] ]
    [ v_R * P1[2] - P1[1] ]
    [ P1[0] - u_R * P1[2] ]

SVD(A) → X = last row of Vh, normalized by X[3]
```

This gives `(X, Y, Z)` in calibration units (cm).

### Undistortion

Before triangulation, detected pixel coordinates are undistorted:

```python
cv2.undistortPoints(pt, cmtx, dist, P=cmtx)
```

This removes lens distortion so the DLT (which assumes a pinhole model) gets
clean inputs. The `P=cmtx` argument re-projects into the same pixel coordinate
system (not normalized coordinates).

Undistortion is applied in **all** triangulation paths — the main `update()` loop,
click-to-measure mode in `test_stereo_measure.py`, and frozen-frame auto-detection.
With barrel distortion `k1 ≈ -0.42`, skipping undistortion causes significant
position error near frame edges.

### Validation Filters

After triangulation, four sanity checks are applied before accepting a result:

| Filter             | Threshold        | Reject reason         |
|--------------------|------------------|-----------------------|
| Disparity range    | 5-500 px         | `low_disp` / `high_disp` |
| Epipolar consistency| ≤ 50 px y-diff  | `epipolar`            |
| Z range            | 10-500 cm        | `z_negative` / `z_close` / `z_far` |
| Reprojection error | ≤ 15 px          | `reproj`              |

**Reprojection error** is computed by projecting the 3D point back into both images
via the projection matrices and measuring pixel distance from the original detections.

### Update Loop

`update()` is the main per-frame method:

1. `grab()` both cameras (synchronized trigger)
2. `retrieve()` both frames
3. `detect()` ball in both frames (independent `BallDetector` instances)
4. If both detect: undistort pixel coords → triangulate → validate
5. Return result dict with `found_3d`, `position_3d`, `disparity`, `reproj_err`,
   or `reject_reason`

### Warmup

MOG2 needs frames to learn the background. `StereoTriangulator` tracks frame count
and reports readiness after `WARMUP_FRAMES = 120` (~1.5s at 80fps). During warmup,
frames are fed via `build_background()` with fast learning rate (0.05).

---

## 6. Trajectory Prediction Pipeline

**Files:**
- `trajectory/position_buffer.py` — `PositionBuffer`
- `trajectory/velocity_estimator.py` — `VelocityEstimator`
- `trajectory/physics_model.py` — `PhysicsModel`
- `trajectory/trajectory_predictor.py` — `TrajectoryPredictor`

### Data Flow

```
3D position (X, Y, Z) from triangulator
        │
        ▼
  ┌─────────────────┐
  │ Outlier Rejection│  — max jump 40cm, max speed 1500 cm/s
  └────────┬────────┘
           │ accepted
           ▼
  ┌─────────────────┐
  │ Position Buffer  │  — circular deque, max 15 points
  └────────┬────────┘
           │ ≥4 points + ≥35ms span
           ▼
  ┌──────────────────┐
  │ Velocity Estimator│ — linear regression (polyfit degree 1)
  └────────┬─────────┘
           │ (vx, vy, vz) cm/s
           ▼
  ┌───────────────────────┐
  │ Vy Midpoint Correction │ — vy += g * (t_latest - t_mean)
  └────────┬──────────────┘
           │ corrected velocity
           ▼
  ┌──────────────────┐
  │   Physics Model   │ — numerical sim (semi-implicit Euler, dt=0.001s)
  └────────┬─────────┘   with gravity, optional drag, optional bounce
           │
           ▼
     Intercept point at target X
```

### Position Buffer

- **Type:** `collections.deque(maxlen=15)`
- **Entry format:** `{x, y, z, t}` (floats, t from `time.perf_counter()`)
- `is_ready(min_points)` — True when buffer has ≥ `min_points` entries
- `get_time_span()` — seconds from oldest to newest entry
- `get_as_arrays()` — returns `(positions[N,3], timestamps[N])` numpy arrays

### Outlier Rejection

Before adding to the buffer, each new point is checked against the previous:

| Check              | Threshold       |
|--------------------|-----------------|
| Position jump      | 40 cm           |
| Speed              | 1500 cm/s       |
| Time delta         | > 0 (no duplicates) |

If rejected, the point is discarded and `_rejected_count` increments.

### Velocity Estimation

**Default method:** Linear regression (`numpy.polyfit(t, positions, 1)`)

For each axis independently, fits a line to the position-vs-time data.
The slope is the velocity component in cm/s.

**Alternative:** Simple `(last - first) / dt` (noisier, not default).

Velocity is capped at `MAX_VELOCITY = 2000 cm/s` — above this the estimate is
invalidated.

### Vy Midpoint Correction

Linear regression estimates velocity at the **midpoint** of the time window, not at
the latest timestamp. Since gravity continuously accelerates Vy, the midpoint
velocity underestimates the current Vy. The correction is:

```
vy_corrected = vy_regression + g_sign * g * (t_latest - t_mean)
```

This shifts the regression velocity forward in time to match the latest measurement.

### Readiness

Prediction is ready when all three conditions are met:
- Buffer has ≥ `min_points` (4)
- Velocity estimate is valid (speed ≤ 2000 cm/s)
- Time span ≥ `MIN_TIME_SPAN` (0.035s, ~3 frames at 80fps)

---

## 7. Physics Model

**File:** `trajectory/physics_model.py` — `PhysicsModel` class

### Coordinate System

Camera frame (all values in cm):
- **X** — along table length (ball travels on this axis between players, 274 cm)
- **Y** — vertical (positive = **down**, standard camera convention)
- **Z** — across table width / depth from camera (152.5 cm + ~110 cm camera offset)

Gravity acts on Y only: `ay = +981 cm/s²` (downward in camera coords).

### Kinematic Equations (no drag)

```
X(t) = X₀ + Vx·t
Y(t) = Y₀ + Vy·t + ½g·t²
Z(t) = Z₀ + Vz·t
```

### Numerical Simulation (with drag)

When `enable_drag=True`, semi-implicit Euler integration is used:

```
dt = 0.001 s (1ms steps)
speed = ||v||
drag = -k * speed    (k = ρ·Cd·A / 2m)
ax = drag * vx
ay = drag * vy + g
az = drag * vz
v += a * dt
p += v * dt
```

**Physical constants:**

| Constant           | Value          |
|--------------------|----------------|
| Gravity            | 981 cm/s²      |
| Ball mass          | 2.7 g          |
| Ball radius        | 2.0 cm         |
| Air density        | 0.001225 g/cm³ |
| Drag coefficient   | 0.45           |

### Stop Conditions

The simulation stops when the first condition is triggered:
- **X-plane crossing** — ball reaches `target_x` (primary for robot interception — ball travels along X)
- **Z-plane crossing** — ball reaches `target_z` (lateral/depth interception)
- **Apex** — `Vy` changes sign (rising → falling)
- **Max time** — 2.0s (timeout)

All use linear interpolation at the crossing point for sub-step accuracy.

### Bounce Model

When `table_y` is set (camera Y of the table surface), the simulation detects
when the ball crosses the table surface while falling:

```
if prev_y < table_y and y >= table_y and vy > 0:
    # interpolate to exact bounce point
    vy = -vy * COR_vertical       # 0.92
    vx *= COR_horizontal          # 0.88
    vz *= COR_horizontal          # 0.88
    bounce_count++
```

| Bounce Parameter          | Value |
|---------------------------|-------|
| Vertical COR              | 0.92  |
| Horizontal velocity retention | 0.88 |
| Max bounces               | 2     |

### Prediction Strategies

`TrajectoryPredictor.predict()` tries two strategies in order:

1. **Apex** — if ball is rising (`vy < 0` in camera coords) and vertical motion
   dominates or ball isn't heading toward robot. Used for lobs.
2. **X-plane** — primary. Finds where ball crosses `target_x` (robot endline along table length).

---

## 8. Coordinate Transform and Robot Command

### Camera-to-Robot Transform

```
robot_x = (cam_z - cam_z_center) * 10      mm, lateral across table (cam Z = table width)
robot_y = -(cam_y - cam_y_table) * 10       mm, height above table (sign flip)
robot_z = |cam_x - robot_x_cam| * 10        mm, depth from endline toward net (cam X = table length)
```

- `cam_z_center` — camera Z at center of table width (set via `set_table_calibration`)
- `cam_y_table` — camera Y at table surface level
- `robot_x_cam` — camera X at the robot's endline

### Table and Robot Geometry

| Dimension          | Value     |
|--------------------|-----------|
| Table length       | 2740 mm   |
| Table width        | 1525 mm   |
| Net height         | 152.5 mm  |
| Robot reach (X)    | ±680 mm   |
| Robot reach (Z)    | 0-440 mm  |
| Robot reach (Y)    | 0-500 mm  |

### Workspace Check

Elliptical XZ workspace boundary:

```
(robot_x / 680)² + (robot_z / 440)² ≤ 1.0
AND 0 ≤ robot_y ≤ 500
```

### Robot Command Output

`get_robot_command()` returns:

```python
{
    'valid': bool,
    'cam_x', 'cam_y', 'cam_z': float,     # intercept in camera coords (cm)
    'robot_x', 'robot_y', 'robot_z': float, # intercept in robot coords (mm)
    'in_workspace': bool,
    't': float,             # time to intercept (seconds)
    'strategy': str,        # 'x_plane' or 'apex'
    'confidence': float,    # 0.0-1.0
    'buffer_points': int,
    'time_span': float
}
```

### Confidence Score

Weighted combination of four factors:

| Factor         | Weight | Saturates at |
|----------------|--------|--------------|
| Buffer points  | 0.30   | 10 points    |
| Time span      | 0.30   | 150 ms       |
| Accept ratio   | 0.20   | 100%         |
| Velocity quality | 0.20 | 50-1200 cm/s |

---

## 9. Calibration Data Reference

### Camera 0 (Left) — Intrinsics

```
K = [ 903.51   0.00   662.65 ]
    [   0.00  905.24   417.56 ]
    [   0.00    0.00     1.00 ]

dist = [-0.4165, 0.2885, 0.0006, -0.0001, -0.1480]
```

- Focal length: ~904 px (both axes, near-square pixels)
- Principal point: (662.6, 417.6) — right of center (640/2=320), below center (480/2=240)
  → calibrated at 1280x800, values reflect that resolution

### Camera 1 (Right) — Intrinsics

```
K = [ 905.70   0.00   657.42 ]
    [   0.00  905.22   342.15 ]
    [   0.00    0.00     1.00 ]

dist = [-0.4365, 0.3289, -0.0027, -0.0008, -0.1999]
```

### Camera 0 — Extrinsics (World Origin)

```
R0 = I (identity)
T0 = [0, 0, 0]
```

Camera 0 defines the world coordinate system.

### Camera 1 — Extrinsics

```
R1 = [ 0.9918  -0.0115   0.1273 ]
     [ 0.0039   0.9982   0.0601 ]
     [-0.1277  -0.0592   0.9900 ]

T1 = [-26.41, 0.17, 2.21]  (cm)
```

- Baseline: ~26.5 cm (primarily in X)
- Small rotation (~7.3° around Y axis)

---

## 10. File Map

### Configuration

| File | Purpose |
|------|---------|
| `config/camera_config.py` | **Single source of truth** for camera IDs, resolution, fps, trigger mode. `configure_camera()` and `load_camera_settings()` |

### Calibration (`camera_calibration/`)

| File | Purpose |
|------|---------|
| `calibration_settings.yaml` | Checkerboard size, camera IDs, frame counts. Read by `calib.py` and `recalib_extrinsic.py` |
| `calib.py` | Full calibration pipeline: intrinsic per camera → stereo extrinsic → visual check |
| `recalib_extrinsic.py` | Extrinsic-only recalibration. Loads existing intrinsics, runs stereo calibrate |
| `camera_parameters/camera{N}_intrinsics.dat` | Intrinsic matrix K + distortion coefficients |
| `camera_parameters/camera{N}_rot_trans.dat` | Rotation R + translation T (world frame) |
| `frames/` | Single-camera calibration frames (15 per camera) |
| `frames_pair/` | Stereo calibration frame pairs (15 pairs) |

### Detection (`tracking/`)

| File | Purpose |
|------|---------|
| `ball_detector.py` | `BallDetector` — canonical MOG2 + contour scoring detection logic |
| `ball_tracker.py` | `EnhancedBallTracker` — thin wrapper with auto warmup (300 frames) + simplified result dict |
| `stereo_detector.py` | `StereoDetector` — dual-camera 2D detection only (no triangulation). Uses grab/retrieve for sync |
| `stereo_triangulator.py` | `StereoTriangulator` — stereo 3D: calibration loading, DLT, undistortion, validation filters |
| `__init__.py` | Exports `BallDetector`, `EnhancedBallTracker`, `StereoDetector`, `StereoTriangulator` |

### Trajectory Prediction (`trajectory/`)

| File | Purpose |
|------|---------|
| `position_buffer.py` | `PositionBuffer` — circular deque of timestamped 3D positions |
| `velocity_estimator.py` | `VelocityEstimator` — linear regression or simple velocity from position history |
| `physics_model.py` | `PhysicsModel` — kinematics with gravity, drag, bounce, axis-plane interception |
| `trajectory_predictor.py` | `TrajectoryPredictor` — main predictor: outlier rejection, velocity, prediction, robot command |
| `test_trajectory_prediction.py` | Interactive 3D trajectory visualizer with recording, calibration clicks, and error analysis. Uses grab/retrieve for sync |
| `__init__.py` | Exports all trajectory components |

### Test Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `test_single_camera.py` | Live single-camera MOG2 detection viewer |
| `test_stereo_feed.py` | Raw stereo camera feed display (no detection). Uses grab/retrieve for sync |
| `test_stereo_detection.py` | Dual-camera 2D detection with lighting normalization modes. Uses grab/retrieve for sync |
| `test_stereo_measure.py` | 3D measurement tool: auto ball diameter + click-to-measure distance. Undistorts all pixel coords before triangulation |
| `test_triangulation.py` | 3D triangulation viewer with rejection analysis and depth statistics |
| `test_velocity_validation.py` | Trajectory prediction accuracy validation: record tosses, compare predicted vs actual |
| `trigger_setup.py` | One-time hardware trigger initialization (run after USB plug-in) |
| `trigger_verify.py` | Camera sync verification: 4 tests including grab-delta timing (<1ms target) |

### Detection Experiments (`detection_experiments/`)

| File | Purpose |
|------|---------|
| `step1_ROI.py` | Interactive click-to-select region of interest |
| `step2_background_subtraction.py` | MOG2 background subtraction demo with morphological cleanup |
| `step3_contour_filtering.py` | Contour filtering + multi-criteria scoring algorithm development |
| `step4_KF.py` | Kalman filter integration for temporal smoothing and coasting |
| `stereo_detection.py` | Full stereo MOG2 + contour detection on both cameras (ground truth for `BallDetector`). Uses grab/retrieve for sync |

### Other

| File | Purpose |
|------|---------|
| `config/ball_thresholds.json` | HSV/LAB thresholds (legacy, not used by MOG2 detection). `load_thresholds()` calls are no-ops |
| `config/ball_thresholds_stereo.json` | Stereo HSV/LAB thresholds (legacy, not used by MOG2 detection). `load_thresholds()` calls are no-ops |
| `scripts/sync_captures/` | Saved synchronized frame pairs from trigger verification |
