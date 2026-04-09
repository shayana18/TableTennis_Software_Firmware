# Stereo Pipeline — Change Log & Debug Report

## Session: 2026-03-18 (Throw Data Logging + Velocity Validation Rewrite)

### Status: Logging instrumented. Velocity validation rewritten for robot-frame KF. Ready for data collection.

---

### Change #26: Velocity Validation Script Rewrite (Robot Frame + KF)

**File:** `scripts/test_velocity_validation.py` — FULL REWRITE

**What:** Replaced deprecated `TrajectoryPredictor` (camera-frame cm, linear regression) with production `RobotPredictor` (robot-frame mm, Kalman filter).

**Key changes:**
- Imports: `RobotPredictor`, `load_points_based_transform`, `cam_to_robot`, `GRAVITY_Z`
- ThrowAnalyzer: X linear, Y linear, Z quadratic (was X linear, Y quad, Z quad)
- Gravity: only in Z axis, `GRAVITY_EXPECTED = 9810 mm/s^2` (no camera pitch decomposition)
- Bounce detection: Z reversal (robot frame: Z negative = down, rising = dz > 0)
- Forward prediction: inline kinematics `z0 + vz*t + 0.5*GRAVITY_Z*t^2` (no PhysicsModel)
- VelocityChart: Row 1 = X/Y/Z position fits, Row 2 = Vx/Vy constant + Vz linear (gravity slope)
- VelocityValidator: loads points-based transform, feeds `cam_to_robot()` output to `RobotPredictor`
- All units: mm and mm/s throughout

---

### Change #27: Throw Data Logger in Integration Script

**File:** `scripts/test_integration_simple.py`

**What:** Added comprehensive per-throw, per-frame data logging to JSON for offline analysis.

**Output file:** `scripts/throw_data_log.json`

**JSON structure:**
```
{
  "session_start": "...",
  "config": { all predictor params, gravity, drag, workspace bounds },
  "throws": [
    {
      "id": 1,
      "start_wall": "...",
      "end_reason": "auto_home_done|lost_detection|manual_clear|...",
      "summary": { n_frames, n_accepted, n_rejected, duration_s, n_sends },
      "frames": [
        {
          "t": abs_timestamp,
          "dt": ms_since_last_accepted,
          "cam": [cx, cy, cz],        // camera cm
          "rob": [rx, ry, rz],        // robot mm
          "ok": true/false,            // predictor accepted?
          "rej": "reason",             // if rejected
          "vel": [vx, vy, vz],        // KF velocity mm/s
          "kf_pos": [px, py, pz],     // KF estimated position
          "buf": N,                    // buffer size
          "kf_rdy": bool,
          "rdy": bool,                // predictor is_ready()
          "disp": px, "rep": px,      // stereo quality
          "bnc": N,                   // bounce count (if >0)
          "pred": { x, y, z, t, vx, vy, vz, clamp }  // intercept prediction
        }, ...
      ],
      "sends": [
        { "t": ts, "target": {x,y,z}, "t_intercept": s, "vel": [vx,vy,vz],
          "latency_ms": ms, "buf_pts": N, "is_update": bool, "clamped": bool }
      ]
    }, ...
  ]
}
```

**Throw lifecycle:**
- Start: first accepted 3D detection when gate is ON
- End: 30 frames with no 3D detection, OR predictor reset (auto-clear, manual, gate toggle)
- End reasons tracked: `auto_home_done`, `lost_detection`, `target_out_of_workspace`, `planning_failed`, `gate_on/off`, `manual_clear`, `manual_reset`, `bg_reset`, `shutdown`

**What to analyze from the data:**
1. KF velocity convergence: per-frame `vel` vs post-hoc trajectory fit
2. Intercept accuracy: `pred` target vs actual ball trajectory continuation
3. Stereo quality: `disp` and `rep` correlations with position noise
4. Transform correctness: `cam` vs `rob` consistency
5. Timing: `dt` inter-frame intervals, `latency_ms` in sends
6. Rejection patterns: `rej` reasons and their frequency

---

### Next Steps

1. Run `python scripts/test_integration_simple.py --port COM6`
2. Collect 10-20 throws with gate ON
3. Copy `scripts/throw_data_log.json` for analysis
4. Analyze: velocity convergence, prediction accuracy, gravity from Z fits, systematic biases
5. Tune R/Q matrices, drag coefficient, workspace bounds based on findings

---

## Session: 2026-03-15 (Points-Based Transform Robustness)

### Status: All 4 robustness features implemented. Ready for testing.

---

### Change #19: Reproj Error Gate in Points-Based Transform Script

**File:** `scripts/test_find_points_based_transform.py`

**What:** Added automatic quality gate after each stereo click pair is triangulated.

**Behavior:**
- Reproj < 1px → auto-accepted (good click)
- Reproj 1-3px → warning printed, user prompted y/n to accept
- Reproj > 3px → auto-rejected with "Re-click" message

**Why:** Without a gate, bad clicks (wrong spot, mismatched left/right) silently enter the Kabsch fit and corrupt the transform. A single garbage point with 10px reproj can shift the solved rotation by several degrees.

---

### Change #20: Multi-Frame Averaging Mode

**File:** `scripts/test_find_points_based_transform.py`

**What:** Press `a` to enter averaging mode. Click left+right on the same marker multiple times (3-5 recommended). Each click pair is triangulated independently. Press `f` to finalize: averages all samples into one point, reports std dev in robot units.

**Key details:**
- Std dev threshold: 5mm (warns if exceeded — indicates inconsistent clicks)
- After finalizing, prompts for robot XYZ as normal
- Press `r` to cancel averaging mode and discard samples
- On-screen overlay shows `[AVG MODE: N samples]` in orange

**Why:** Single-click noise is ~5mm at typical depths. Averaging 3-5 clicks reduces to ~2-3mm without needing ArUco markers. Especially helpful for manual marker clicks where sub-pixel precision is hard.

---

### Change #21: Leave-One-Out Outlier Detection After Solving

**File:** `scripts/test_find_points_based_transform.py`

**What:** After Kabsch SVD solve, computes per-point residuals and flags outliers.

**Algorithm:**
- Outlier threshold = max(2× median residual, 5.0 robot-units)
- Points exceeding threshold are flagged
- If outliers found AND removing them leaves ≥3 points: prompts user to re-solve excluding outliers
- On confirmation: removes outlier points from stored lists, re-runs Kabsch, prints improved RMSE

**Why:** One bad point pair (e.g., wrong robot coordinate entered, or a click on a different marker) can dominate the fit. Outlier detection catches this automatically after the initial solve and offers a clean re-solve.

---

### Change #22: Summary Table in Transform Output

**File:** `scripts/test_find_points_based_transform.py`

**What:** After solving, prints a numbered table of all point pairs with columns:
- `#` — point index
- `Camera Scaled (X,Y,Z)` — triangulated point in robot units
- `Robot (X,Y,Z)` — user-entered robot coordinates
- `Residual` — Euclidean distance between fitted and actual robot point
- `Flag` — `OUTLIER` if above threshold, blank otherwise

Also prints outlier threshold value when outliers exist.

**Why:** Makes it immediately obvious which pose was bad. Previously only showed a flat array of per-point errors with no context.

---

### Updated Controls

Old: `SPACE freeze | r reset pair | d delete last | q quit`
New: `SPACE freeze | a avg | f finalize | r reset | d del | q quit`

- `a` — enter averaging mode (multi-click same marker)
- `f` — finalize average (compute mean, prompt robot XYZ)
- `r` — reset click pair; also cancels averaging mode if active

---

### Next Steps

1. Recalibrate cameras (still pending from 2026-03-10)
2. Run `python scripts/test_find_points_based_transform.py --required-points 4` to test new features
3. Verify reproj gate rejects intentionally bad clicks
4. Verify outlier detection flags worst point after solving
5. Verify multi-click averaging reports std dev correctly

---

## Session: 2026-03-10 (Integration Script — Robot-Frame Prediction + UART)

### Status: test_integration_simple.py v4 complete. Air drag model + rectangle workspace. Ready for testing.

---

### Background: Why a New Script

`test_integration_day.py` used `TrajectoryPredictor` which predicts in **camera frame** — decomposing gravity along camera axes, then transforming to robot coords. This produced wildly inaccurate intercept predictions (e.g., Z positive when the ball is below the robot base). The camera-frame gravity decomposition amplified errors through the rotation matrix.

**Solution:** Create `test_integration_simple.py` that predicts entirely in **robot frame (mm)** where gravity is simply `(0, 0, -9810)` mm/s². No axis decomposition needed.

---

### Change #15: Created test_integration_simple.py (v1)

**File:** `scripts/test_integration_simple.py` — **NEW**

**What:** End-to-end integration script replacing test_integration_day.py.

**Pipeline:**
1. Triangulate ball in camera frame (cm) via verified `StereoTriangulator`
2. Transform to robot frame (mm) via verified `cam_to_robot()` rotation matrix
3. Buffer robot-frame positions, estimate velocity via linear regression
4. Predict trajectory with gravity: `pos(t) = p0 + v*t + 0.5*(0,0,-9810)*t²`
5. Scan forward in time for first point entering workspace with enough reaction time
6. Send `(x, y, z, time)` to robot via UART — no velocity data sent

**Key design:**
- `RobotPredictor` class: self-contained robot-frame predictor
  - X, Y velocity: linear regression (no gravity)
  - Z velocity: gravity-corrected regression (`z_corrected = z - 0.5*g*dt²`)
  - Workspace: elliptic XY (790×540mm, scaled by SAFE_XY_SCALE) + Z bounds
  - Direction filter: only predict when ball approaches workspace
  - Reachability check: trapezoidal velocity profile estimate
- `cam_to_robot()`: same verified rotation matrix as trajectory_predictor.py
- Camera pose constants duplicated at top of file (not imported, for standalone use)

**Camera pose:**
```
CAM_POSE_X_MM  = 1582.5    (camera to robot's RIGHT)
CAM_POSE_Y_MM  = 1500.0    (camera toward net)
CAM_POSE_Z_MM  = -452.4    (camera below base plate)
CAM_POSE_YAW   = 185°      (looking in -X, 5° toward net)
CAM_POSE_PITCH = 20°       (fixed stand)
CAM_POSE_ROLL  = 0°
```

---

### Test Results: v1 (~30 throws)

- Robot went to 23 intercepts total, but:
  - 7 rejected by STM32 ("TARGET OUT OF WORKSPACE") — deep Z (<-1000) or large XY at extreme Z
  - Only ~2 of 30 throws produced visually accurate intercepts
  - Most predictions were spatially close but not accurate enough for interception
- **Key issue:** Manual 'c' key needed between throws to clear `intercept_sent` flag
  - Many throws missed because the robot was still in "sent" state from previous throw

---

### Change #16: test_integration_simple.py v2 — Auto-Clear + State Machine

**What:** Major update to fix usability and robustness issues.

**Auto-clear on COMPLETED Q:**
- No manual 'c' key needed between throws
- After robot completes intercept → auto-send HOME → auto-clear on HOME completion
- State machine: `_pending_action` tracks flow: `None → 'intercept' → 'homing' → None`

**Tighter workspace bounds:**
```python
Z_MIN = -1000.0   # conservative (robot.h: -1050)
Z_MAX = -731.0    # conservative (robot.h: -721)
SAFE_XY_SCALE = 0.85  # initially; later reduced to 0.80
```

**Direction filter:**
- `_ball_approaching()` — only predict when ball moves toward workspace
- `MIN_APPROACH_VY = -200 mm/s` or Y < 600mm (ball already close)

**Enhanced diagnostics:**
- Per-throw log: target coords, ball position, velocity, buffer count, latency
- Overlay: shows state (MOVING/HOMING/SENT/READY), approach indicator, throw count
- Terminal: real-time position + velocity + intercept + reason ([AWAY], [NO_WS])
- STM32 rejection handling: auto-clear on "TARGET OUT OF WORKSPACE"

---

### Test Results: v2 (6 throws)

- 1 of 6 throws sent to robot (low success rate)
- **Critical bug found:** HOME → COMPLETED Q → HOME infinite loop
  - Each HOME command produces "COMPLETED Q" from STM32
  - Script treated every "COMPLETED Q" as intercept completion → sent another HOME
  - Robot oscillated between HOME commands indefinitely
- Other throws: MIN_POINTS=8 too restrictive (not enough tracking frames), [NO_WS] rejections

---

### Change #17: HOME Loop Fix + Parameter Relaxation

**HOME loop fix — state machine:**
```python
# _pending_action tracks what we're waiting for:
#   None       = idle, ignore COMPLETED Q (startup noise)
#   'intercept' = sent intercept → on COMPLETED Q, send HOME
#   'homing'    = sent HOME → on COMPLETED Q, clear & reset
```

Only HOME is sent after an intercept completion. A "COMPLETED Q" during homing triggers the final clear (not another HOME). "COMPLETED Q" when idle is ignored.

**Parameter relaxation:**
```python
MIN_POINTS:     8 → 6    # fewer frames needed to start predicting
SCAN_DURATION:  1.0 → 1.5  # scan further forward in time
SAFE_XY_SCALE:  0.85 → 0.80  # tighter XY to avoid IK rejections
SAFE_Z_MARGIN:  20 → 15mm    # slightly wider Z range
```

**Rationale:** With MIN_POINTS=8, most throws didn't accumulate enough tracking frames before the ball left the camera FOV. At 100fps with stereo matching, we typically get 6-10 valid frames per throw. MIN_POINTS=6 is the sweet spot — stable enough for regression, fast enough to react.

---

### Test Results: v2 (13 throws, retested after HOME loop fix)

- 13 throws sent to robot — all reached STM32 and robot attempted intercept
- **Accuracy:** Poor — robot went to wrong positions (visually ~10-30cm off)
- **Arc visualization:** Trail appeared inverted/wrong direction
- **Root cause analysis:**
  - Predictions made from only 6 points at Y=1700mm+ (ball still far from robot)
  - No air drag model: horizontal velocity decelerates 15-25% due to drag on ping pong ball
  - At 3-4 m/s over 300-400ms flight time, drag causes 80-350mm position error
  - Arc visualization only showed 0.4s window with no-drag physics — didn't match actual ball path

---

### Test Results: v3 (3 throws, proximity filter + continuous updates)

- Added MAX_PREDICT_Y=1400mm proximity filter and continuous update while STM32 in PLAN
- 3 throws sent — still inaccurate
- Predictions improved slightly (closer predictions) but still ~5-15cm off
- Arc trail still visually incorrect
- Confirmed: air drag is the dominant error source, not a code bug

---

### Change #18: test_integration_simple.py v4 — Air Drag + Rectangle Workspace

**What:** Major physics and workspace overhaul for accurate interception.

**Air drag model (Euler integration):**
```python
DRAG_K = 0.000112  # mm^-1 — Cd=0.40, mass=2.7g, diameter=40mm
# Per step: drag = DRAG_K * speed; a_drag = -drag * v_component
# Euler integration at dt=0.001s for predict, dt=0.01s for arc visualization
```
- Ping pong ball drag is 15-20% of gravity at typical 3-4 m/s speeds
- Over 400ms flight: no-drag model overshoots Y by ~84mm, undershoots Z by ~29mm
- Sanity check: at v=3000mm/s over 300ms, Vy decelerates from -3000 to -2682 (10.6%)

**Rectangle workspace (10% bigger than robot.h):**
```python
WS_HALF_X = 869.0   # mm (790 * 1.1) — firmware handles actual bounds
WS_HALF_Y = 594.0   # mm (540 * 1.1)
Z_MIN     = -1050.0  # mm (robot.h value, no margin)
Z_MAX     = -721.0   # mm (robot.h value, no margin)
```
- Was: elliptic XY with 0.80 safe scale + Z margins
- Now: simple rectangle, slightly oversized — let firmware IK reject if truly unreachable
- Rationale: firmware `check_workspace()` in robot.h handles actual bounds

**Proximity filter:**
- `MAX_PREDICT_Y = 1400mm` — don't predict until ball is within 1400mm of robot in Y
- Prevents wild predictions from far-away detections (was sending from Y=1700+)

**Continuous updates:**
- Track `_stm32_moving` flag: set True on "STATE: MOVE" UART line, cleared on completion
- While `_pending_action == 'intercept'` and `_stm32_moving is False`: keep sending refined predictions
- Once robot starts moving, stop updating (can't change target mid-move)

**Arc visualization fix:**
- Was: 0.4s no-drag parabola (appeared inverted because ball was going UP early in trajectory)
- Now: 1.0s drag-aware Euler integration — arc curves realistically and matches actual prediction
- Same `_step_euler()` used for both prediction and visualization — guaranteed consistency

**Other parameter changes:**
```python
MIN_TIME_SPAN: 0.06 → 0.08s   # need more data for stable velocity
MIN_TIME_HIT:  0.15 → 0.10s   # allow faster reactions
SCAN_DURATION: 1.5s            # scan 1.5s into future
```

---

### Pending: v4 Testing

1. Run `python scripts/test_integration_simple.py --port COM6`
2. Verify drag model improves interception accuracy
3. Tune DRAG_K if predictions still systematically off
4. Verify arc visualization matches actual ball path
5. Check rectangle workspace — expect fewer "TARGET OUT OF WORKSPACE" rejections

---

## Session: 2026-03-10 (Diagnosing High Reprojection Error)

### Problem: 4-5px Reprojection Error in Triangulation Verification

User recalibrated cameras and re-ran `test_triangulation_verify.py` mode 1 (DISTANCE). Got 4.5-5.5px reproj error, compared to 0.2-0.8px with the 2026-03-07 calibration.

### Root Cause: cam0 Intrinsic Calibration is Unstable

**Evidence — comparing current vs previous (working) calibration:**

| Parameter | 2026-03-07 (worked) | Current (broken) | Change |
|-----------|---------------------|-------------------|--------|
| cam0 fx   | 545.15              | 576.13            | +5.7%  |
| cam1 fx   | 531.07              | 532.65            | +0.3%  |
| cam0 k2   | 0.277               | 0.544             | 2x     |
| cam0 k3   | -0.006              | -0.506            | 84x    |
| fx diff   | 2.6%                | 8.2%              | —      |

- cam0 focal length jumped 5.7% while cam1 barely changed. Same lens model (OV9782) — should be <3% apart.
- cam0 k3 exploded from -0.006 to -0.51, k2 doubled. Higher-order distortion terms wildly unstable → **overfitting**.
- When cam0 undistortion is wrong, `_rectify_point()` produces incorrect normalized coordinates.
- Stereo rectification can't make epipolar lines horizontal → y-mismatch after rectification becomes ~8-10px.
- DLT distributes this as ~4-5px reprojection error per camera.

**Mechanism:** DLT reproj ≈ epipolar_error / 2. So 4-5px reproj → ~8-10px epipolar mismatch → rectification failure → bad undistortion → bad intrinsics.

### Changes Made

**1. Added calibration quality diagnostics to test_triangulation_verify.py**
- Startup: prints both cameras' fx, fy, k1, k2, k3 + warns if fx difference >5% or k3 too high
- Startup: tests rectification quality at image center (epipolar error of center points)
- Per-click: prints raw coords, rectified coords, epipolar error (dy), per-camera reproj
- Warns explicitly if epipolar error >3px ("rectification quality is poor, likely bad intrinsics")

**2. Tightened MAX_EPIPOLAR_ERR in stereo_triangulator.py**
- `MAX_EPIPOLAR_ERR`: 35 → 15px (catches bad rectification with informative message in auto-detection path)

### Change: Swapped Camera Order (device 1 ↔ device 2)

After recalibration with good intrinsics, epipolar error dropped to **0.3px** (excellent!), but disparity was negative — cameras were physically swapped. Device 1 (camera0 in calibration) was on the RIGHT, device 2 (camera1) was on the LEFT.

**Fix applied (no recalibration needed):**
1. Swapped camera0_intrinsics.dat ↔ camera1_intrinsics.dat
2. Inverted extrinsics: new camera0 (device 2, LEFT) = origin, new camera1 (device 1, RIGHT) = R^T, -R^T@T
3. Updated camera_config.py: LEFT=2, RIGHT=1
4. Updated calibration_settings.yaml: camera0=2, camera1=1
5. Baseline preserved: 23.41 cm

### Result: 0.5px Reproj — Triangulation Verified!

After camera swap + alpha=-1 fix:
- Reproj: **0.5px** (was 4-5px before recalibration)
- Epipolar dy: sub-pixel
- Rectified focal: ~518px (was 1287px with alpha=0 zoom)
- Sample point: X=-4.36, Y=+31.33, Z=117.95 cm

**Triangulation is now verified and accurate. Moving to camera→robot coordinate transform.**

### Change: Camera→Robot Transform — Measured Offsets Applied

**File:** `trajectory/trajectory_predictor.py`

**Measured camera pose relative to robot base (mm):**
```
CAM_POSE_X_MM  = +1582.5   (camera to robot's RIGHT)
CAM_POSE_Y_MM  = +1240.0   (camera toward net)
CAM_POSE_Z_MM  = -452.4    (camera below base plate)
CAM_POSE_YAW   = 185°      (looking in -X direction, 5° toward net)
CAM_POSE_PITCH = 20°       (fixed stand)
CAM_POSE_ROLL  = 0°
```

**Sanity check** with verified triangulation point (cam: -4.36, +31.33, 117.95 cm):
- Robot: (588.9, 1109.3, -1150.2) mm
- Point is on table surface (Z≈-1150, below workspace) and far toward net (Y=1109, outside ±540 workspace) — physically correct for a table-surface click
- Robot HOME (0,0,-900) → camera (−110, −16, 174) cm — reasonable for 1582mm lateral offset

---

### Previous Fix Required (now resolved): Recalibrate cam0 Intrinsics

The cam0 mono calibration needs to be redone with:
1. **30+ images** covering the full frame (especially corners and edges)
2. **Multiple distances** (20cm, 40cm, 60cm from camera)
3. **Multiple orientations** of the checkerboard (tilted, rotated)
4. Consider using `cv2.CALIB_FIX_K3` flag if k3 keeps being unstable — 5-coefficient model may overfit for this lens

After re-running cam0 intrinsics, re-run stereo calibration (extrinsics), then verify with mode 1.

---

## Session: 2026-03-08 (Velocity Estimation Debugging)

### Key Finding: Test Metrics Were Comparing Against Wrong Reference

The velocity validation test fitted a SINGLE polynomial across the ENTIRE trajectory, but
throws have 80-290ms stereo dead zones where the ball crosses between cameras. The ball
often bounces invisibly during these gaps. A single polynomial across gaps + bounces gives
nonsensical gravity (241, -10, -10, 415, 251 cm/s²) and meaningless "velocity error" metrics.

**Per-arc gravity** (splitting at gaps AND bounces) is actually reasonable: 878-1034 cm/s².
The real-time **Vx estimates are already excellent** (1-8% within-arc error).

### Changes Made

**1. trajectory_predictor.py — Z-axis median filter (kept)**
- 3-point running median on Z before buffering — kills ±3cm stereo depth spikes
- Cleared on gap reset, bounce, and full reset
- Re-seeded with last 2 points' Z on bounce

**2. trajectory_predictor.py — Increased min data for first velocity (kept)**
- `MIN_TIME_SPAN`: 0.035 → 0.08s
- `min_points`: 4 → 6

**3. trajectory_predictor.py — Bounce Vx/Vz preservation (REMOVED)**
- Initial plan: preserve pre-bounce Vx/Vz across bounces, blend after 10 points
- Result: HARMFUL. Pre-bounce arcs often had only 5-6 points (unreliable estimates).
  Throw 5 showed 60 cm/s Vx discontinuity and 125 cm/s Vz discontinuity at blend threshold.
- Reverted completely.

**4. test_velocity_validation.py — Per-arc analysis overhaul**
- Arc segmentation now splits on BOTH bounces AND time gaps (>60ms)
- Convergence table compares RT velocity against per-arc fit (not whole-trajectory)
- Forward prediction computed per-arc (stays within each arc)
- Summary uses per-arc metrics throughout
- Fixed `min_points=4` override → `min_points=6` to match predictor default

**5. test_velocity_validation.py — min_points bug fix**
- Test script hardcoded `min_points=4`, overriding the predictor's default of 6
- All previous test runs had velocity firing at buf=4-5 despite the code change

### Test Results (pre-fix analysis, 5 throws)
- Per-arc gravity: 878-1034 cm/s² (within 5-10% of 981)
- Stereo baseline: 23.2cm (was 42.77cm) — depth precision degraded ~2x
- New calibration: cam0 fx=531, cam1 fx=535, baseline=23.2cm, rotation=10.5°
- Forward prediction errors inflated by cross-arc prediction (predicting across invisible bounces)

### Next Steps
- Re-run test with fixed analysis — expect much lower reported errors
- Consider increasing stereo baseline for better depth precision
- Per-arc gravity ~900 cm/s² (-8%) suggests mild calibration scale issue

---

## Session: 2026-03-07 (Calibration Verification & Debugging)

### Status: ALL TESTS COMPLETE. ~3% lateral scale error identified. Ready to fix or proceed.

### Key Finding: Consistent 3% Lateral Scale Error

**Ruler test:** 50.05 cm object reads 51.50 cm → **+2.9% error** (α=0.972)
**Gravity test:** Good throws read avg 1010 cm/s² vs expected 981 → **+3.0% error**
**Both match** → focal length ~3% underestimated, X and Y measurements read ~3% too large.

**Camera tilt investigation:**
- Mode 1 data showed Y changing from +11 to -44 as depth increased (129→256 cm)
- Initially calculated 24° camera pitch from this Y-vs-Z trend (R²=0.995)
- Predicted gravity should read 894 cm/s² (= 981×cos(24°)) → implied 13% Y-axis error
- **BUT** ruler test showed only 2.9% error, contradicting the 13% prediction
- **Resolution:** The Y-vs-Z trend reflects the camera frame geometry (optical axis pointed downward), but this does NOT require a cos(θ) correction for gravity. The triangulation already works in the camera frame correctly. The gravity quadratic fit extracts acceleration along whatever axis it's measured — and the ~3% discrepancy is fully explained by the lateral scale factor (α=0.972), NOT by tilt.
- **Proof:** cos(θ) = g_measured × α / g_true = 1010 × 0.972 / 981 = 1.0006 ≈ 1.0 → no tilt correction needed.

**Previous session comparison:**
- Old calibration (15 images): g=824 → α=981/824=1.19 (19% overestimation)
- New calibration (30+ images): g=1010 → α=981/1010=0.971 (3% underestimation)
- Massive improvement from 19% → 3%. Additional recalibration could close remaining gap.

---

### Change #1: Created Triangulation Verification Script

**File:** `scripts/test_triangulation_verify.py` — **NEW**

**What:** Interactive 4-mode verification tool for stereo triangulation accuracy.

**Modes:**
1. **DISTANCE** — Click a point on LEFT then RIGHT image → auto-triangulates → prints X, Y, Z coords. User measures with tape to compare. No ground truth prompt needed.
2. **LENGTH** — Click 4 points (A left, A right, B left, B right) → auto-triangulates both endpoints → prints 3D distance + per-axis deltas (dX, dY, dZ). User measures object length to compare.
3. **GRAVITY** — Toss ball, record trajectory, fit Y(t) quadratic → extract measured gravity. Compares against 981 cm/s². The definitive lateral scale test.
4. **DIAMETER** — Continuous per-frame ball diameter logging to terminal. Each unbroken detection sequence is a "pass" with summary stats. Tests consistency (NOT absolute accuracy — Z/fx cancel).

**Features:**
- Trigger sync verification (`CAP_PROP_BACKLIGHT=1` check on both cameras)
- Prominent click visuals: circle + crosshair + center dot + label (A=blue, B=red)
- Yellow border highlights which image to click next
- Auto-triangulate on final click (no manual save needed)
- Results auto-saved, summary printed on quit, JSON export

**Why:** Ball diameter AUTO test (`test_stereo_measure.py`) is self-consistent (Z/fx errors cancel) and cannot detect lateral scale errors. Previous session found ~19% lateral scale error. This tool tests each axis independently using physical references.

---

### Change #2: Bug Fix — MARKER_CROSS Typo

**File:** `scripts/test_triangulation_verify.py` line 642
**What:** `cv2.MARKER_CROSs` → `cv2.MARKER_CROSS`
**Why:** Typo caused crash on first click in Mode 2 (LENGTH).

---

### Change #3: Simplified Mode 1 (DISTANCE)

**What:** Removed ground truth prompt. Now: click LEFT → click RIGHT → auto-prints X, Y, Z box. Press 'r' to reset and click next point.
**Why:** User measures with tape AFTER seeing script output — simpler, faster workflow.

---

### Change #4: Simplified Mode 2 (LENGTH)

**What:** Removed ground truth length prompt, orientation selection, and `_prompt_length_gt()`. Now: click 4 points → auto-prints 3D distance + dX/dY/dZ + per-point coords. Press 'r' to reset.
**Removed:** `LengthResult.ground_truth_len`, `LengthResult.orientation`, `LengthResult.error_cm`, `LengthResult.error_pct`. `delta_x/y/z` now signed (not absolute).
**Why:** User measures with tape AFTER seeing script output. No need to type anything.

---

### Verification Test Results (this session)

#### Current Calibration Intrinsics
```
Camera 0 (LEFT):  fx=545.15  fy=543.16  cx=327.03  cy=213.72  HFOV=60.8°
Camera 1 (RIGHT): fx=531.07  fy=530.69  cx=330.75  cy=226.53  HFOV=62.1°
Distortion cam0:  k1=-0.443  k2=0.277  p1=-0.002  p2=-0.001  k3=-0.006
Distortion cam1:  k1=-0.425  k2=0.265  p1=0.003   p2=-0.000  k3=-0.096
```
OV9782 datasheet: 73° HFOV at 1280×800. At 640×480 center crop ≈ 62-66°. Calibrated values are in range.

#### Test 1: DISTANCE (Mode 1) — 6 points

| # | X(cm) | Y(cm) | Z(cm) | Reproj |
|---|-------|-------|-------|--------|
| 1 | -5.80 | +11.58 | 136.83 | 0.5px |
| 2 | +6.20 | -15.95 | 187.37 | 0.6px |
| 3 | +9.32 | +7.39 | 140.75 | 0.8px |
| 4 | +12.56 | -17.10 | 194.75 | 0.2px |
| 5 | +8.81 | +11.40 | 129.11 | 0.3px |
| 6 | +18.11 | -43.95 | 256.06 | 0.8px |

**Observation:** Y flips from positive to negative at Z ≈ 155cm.
**Explanation:** Camera is tilted downward ~12-15° to see table. At close range, table is below optical axis (Y+). At far range, optical axis dips below table surface (Y-). This is normal camera tilt, NOT a calibration error.
**Reproj errors:** 0.2-0.8px — excellent stereo geometry consistency.

#### Test 2: LENGTH (Mode 2) — 50.05 cm horizontal object

| # | 3D Dist | dX | dY | dZ | Depth | Reproj |
|---|---------|-----|-----|-----|-------|--------|
| 1 | 51.50 | +51.39 | -0.36 | -3.27 | 144 cm | 0.4px |

- **Actual length:** 50.05 cm
- **Error:** +1.45 cm (+2.9%)
- **X-axis scale factor:** α = 50.05 / 51.50 = 0.972
- dY ≈ 0 confirms object was level (horizontal)
- Reproj 0.4px — excellent stereo consistency

#### Test 4: DIAMETER (Mode 4) — 2 major passes

| Pass | Points | Mean Diam | Std | Z Range | Overread vs 4.0cm |
|------|--------|-----------|-----|---------|-------------------|
| 7 | 242 | 4.37 cm | 0.45 | 111-179 cm | +9.3% |
| 8 | 618 | 4.45 cm | 0.30 | 151-212 cm | +11.3% |

**Previous session results (old calibration, for comparison):**

| Pass | Points | Mean Diam | Std | Z Range | Overread |
|------|--------|-----------|-----|---------|----------|
| 1 | 89 | 4.66 cm | 0.58 | 130-229 cm | +16.5% |
| 3 | 58 | 4.54 cm | 0.61 | 100-131 cm | +13.5% |
| 4 | 87 | 4.93 cm | 0.76 | 156-172 cm | +23.3% |

**Improvement:** New calibration gives tighter std (0.30 vs 0.58-0.76) and lower overread (9-11% vs 13-23%).

**Root cause of ~10% overread:** Contour area inflation from MOG2 + MORPH_CLOSE 7×7 kernel. Adds ~1-2px to detected radius. At typical detection radii of 6-9px, this produces 10-25% area overestimate. Depth-dependent trend confirmed: smaller pixel ball at greater depth → same pixel padding → larger % error. Also position-dependent: barrel distortion (k1=-0.44) compresses edges → smaller r_px at frame edges → lower diameter reading.

**Key insight:** Diameter test is CONSISTENCY check only. Z/fx errors cancel in `diam = 2*r_px*Z/fx`. Cannot verify absolute calibration accuracy.

#### Test 3: GRAVITY (Mode 3) — 3 throws

| # | g (cm/s²) | Error % | R² | Points | Duration |
|---|-----------|---------|------|--------|----------|
| 1 | 999 | +1.8% | 0.9824 | 15 | 590ms |
| 2 | 769 | -21.7% | 0.8922 | 31 | 783ms |
| 3 | 1022 | +4.2% | 0.9885 | 19 | 641ms |

- **Throw 2 discarded:** R²=0.89 (noisy tracking, bad throw)
- **Good throws avg:** (999 + 1022) / 2 = **1010 cm/s²** (+3.0% vs 981)
- **Matches ruler error** (+2.9%) — confirms consistent lateral scale factor α=0.972
- **Previous calibration:** g=824 (19% off) → **Now g=1010 (3% off)** — massive improvement

---

### Analysis: Previous vs Current Calibration

**Previous calibration (15 mono images):**
- Focal length overestimated by α≈1.19
- Gravity read 824 cm/s² (16% low) — lateral X,Y shrunk
- Diameter read ~3.3-3.5cm (low) — consistent with α=1.19
- Z measurements correct (stereoCalibrate compensates baseline)

**Current calibration (30+ mono images):**
- Diameter now reads HIGH (4.37-4.45 vs 4.0) — old α=1.19 overestimation is GONE
- If α<1 now (focal length slightly underestimated), diameter reads high: `diam = true/α > true`
- But the ~10% overread is better explained by contour inflation artifact
- Gravity test will confirm whether lateral scale is now accurate

---

## Session: 2026-03-02

### Problem Statement
Velocity estimation validation (`test_velocity_validation.py`) shows:
- **Gravity consistently ~16% low**: mean 824 cm/s² vs expected 981 cm/s²
- **Many stereo frame rejections**: reproj errors 5-10px causing dropped frames
- **Forward prediction errors**: 18.2cm mean, 23cm max (throw #1)

---

### Change #1: Reprojection Threshold (5px → 8px)
**File:** `tracking/stereo_triangulator.py` line 57
**What:** `MAX_REPROJ_ERR = 5` → `MAX_REPROJ_ERR = 8`
**Why:** Many valid detections were rejected with reproj errors of 5.0-10.1px. In throws 1, 3, and 4, we lost 9-10 consecutive frames to reproj failures just above 5px. These gaps trigger buffer resets (GAP_RESET_TIME=80ms) — destroying accumulated trajectory data mid-throw.
**Effect:** Recovers most rejected frames. Trajectory regression averages out the added noise. More data points = better velocity estimates.

---

### Root Cause Investigation: 16% Gravity Error

#### Hypothesis 1: Checkerboard square size wrong — RULED OUT
User confirmed squares are 3.17cm as configured.

#### Hypothesis 2: Ball diameter test shows correct scale — MISLEADING
The AUTO diameter measurement uses `diameter = 2 * r_px * Z / fx`. If focal length is overestimated by factor α:
- Z = (α·f) · (B/α) / disparity = f·B/d → **Z is CORRECT** (errors cancel)
- But diameter = 2·r_px·Z_true / (α·f_true) = true_diam / α → **reads low**

User reported diameter "close to 4 but not exact" — likely ~3.3-3.5cm, consistent with α ≈ 1.15-1.19. **The AUTO test cannot detect scale errors because Z/fx cancels.**

#### Hypothesis 3: Timestamp bias — RULED OUT
`time.perf_counter()` on Windows uses `QueryPerformanceCounter` (sub-microsecond). Processing latency is roughly constant per frame (dominated by 10ms camera trigger interval). Constant offset doesn't affect acceleration measurement. Verified from output: inter-frame intervals of 20-26ms match expected 2-3 raw frames at 100fps.

#### Hypothesis 4: Air drag — RULED OUT as primary cause
Ping pong ball (2.7g, 40mm) at 290 cm/s: drag force ≈ 14% of gravity. BUT for mostly-horizontal throws (Vy << |V|), the Y-component of drag averages near zero. Drag barely affects the quadratic Y(t) fit. If anything, drag during the upward phase makes effective gravity HIGHER, not lower.

#### ROOT CAUSE FOUND: Focal length overestimation → lateral scale error

**Math proof for rectified stereo DLT:**
```
Given: f_cal = α · f_true  (calibrated focal length too high)

stereoCalibrate (CALIB_FIX_INTRINSIC) compensates:
  B_cal = B_true / α  (finds smaller baseline to match pixel observations)

Triangulation results:
  Z = f_rect · B_rect / disparity = (α·f)·(B/α)/d = f·B/d = Z_true  ← CORRECT
  Y = (v - cy) · B_rect / disparity = (v-cy)·(B/α)/d = Y_true / α   ← SHRUNK
  X = (u - cx) · B_rect / disparity = (u-cx)·(B/α)/d = X_true / α   ← SHRUNK

Measured gravity: g_meas = g_true / α
  → 824 = 981 / α  →  α = 1.19
  → Focal length is ~19% overestimated
```

**Supporting evidence:**
- Gravity consistently 13-19% low across all 3 throws (R²=0.999 on Y quadratic)
- Only 15 mono calibration images per camera (insufficient for precise focal length)
- Ball diameter reads ~3.3-3.5cm instead of 4.0cm (consistent with α=1.19)
- Z (depth) measurements are reasonable — scale error is lateral only

---

### Fix Applied: Recalibration with 30+ mono images
User recalibrated intrinsics with more images covering full frame. New intrinsics shown in Session 2026-03-07 above. The lateral scale correction hack (`LATERAL_SCALE = 1.19`) was NOT applied — proper recalibration was done instead.

---

### Test Data (3 throws from this session — OLD calibration)

| Throw | Points | Duration | Measured g | g Error | Fwd Pred Mean | Fwd Pred Max |
|-------|--------|----------|-----------|---------|---------------|-------------|
| 1     | 14     | 497ms    | 798       | -18.6%  | 18.2cm        | 23.0cm      |
| 2     | 4      | 66ms     | 824       | -16.0%  | 0.7cm         | 0.7cm       |
| 3     | 7      | 244ms    | 851       | -13.3%  | 2.8cm         | 4.4cm       |

**Aggregate:** g_mean=824, g_std=22, Vel err: Vx=10.4 Vy=11.6 Vz=9.3 cm/s

---

### Additional Findings (for future work)

1. **Independent left/right detection** — No epipolar constraint during detection matching. Left and right detectors pick best candidates independently, paired by frame order only. Could cause occasional mismatches.

2. **Epipolar threshold too loose** — `MAX_EPIPOLAR_ERR=30px` after rectification should ideally be <5px. The 30px threshold allows mismatched detections through.

3. **Drag not enabled in validation** — `test_velocity_validation.py` creates predictor with `enable_drag=False` and forward prediction also disables drag. For a 2.7g ping pong ball, drag significantly affects horizontal deceleration over longer prediction horizons.

4. **Calibration RMSE files deleted** — `camera0_intrinsics_rmse.dat`, `camera1_intrinsics_rmse.dat`, `stereo_calibration_rmse.dat` were removed from git. No way to assess calibration quality without re-running.

---

---

## Session: 2026-03-07 (Velocity Validation Preparation)

### Status: Script reviewed and enhanced. Ready to run velocity validation tests.

### Change #5: Enhanced Velocity Validation Diagnostics

**File:** `scripts/test_velocity_validation.py`

**What:** Added comprehensive debugging output for velocity accuracy analysis.

**New per-frame output:**
- `dt_ms` column showing inter-frame timing (detects timing issues / dropped frames)
- Velocity now shows corrected Vy (midpoint-adjusted) instead of raw regression Vy

**New per-throw analysis:**
- **Fit residuals (RMS)** — per-axis position residuals from curve fits. High residuals indicate noisy tracking.
- **Velocity convergence table** — shows RT velocity estimate at each frame vs post-hoc truth. Columns: `#pts, t, Vx_RT, Vy_raw, Vy_corr, Vz_RT | eVx, eVy, eVz, e3D`. Shows how velocity accuracy improves as buffer fills.
- **Dual forward prediction** — computes prediction errors with both `g=981` (expected) and `g=measured` (from quadratic fit). Separates 3% scale error from velocity estimation error.
- **Raw data dump** — CSV-style `t, x, y, z` for every point, copy-paste ready for external analysis.

**Startup calibration info:**
- Prints full intrinsics (fx, fy, cx, cy), distortion coefficients, baseline, rectified focal length, and thresholds at script start. Creates a complete debug log header.

**Velocity snapshot tracking:**
- Stores `(t, n_pts, vx, vy_raw, vy_corr, vz)` at every frame where predictor has valid velocity
- Enables convergence analysis: how quickly does the velocity estimate converge to ground truth?

---

### Change #6: Fixed Axis Convention Comments (Corrected TWICE)

**Files:** `trajectory/physics_model.py`, `trajectory/trajectory_predictor.py`

**Physical setup confirmed with user:**
- Cameras at midpoint of table length, ~110cm offset from table edge
- Cameras look ACROSS the table width (optical axis perpendicular to table length)
- X = along table LENGTH (274cm, ball travels player-to-player)
- Y = vertical (down = positive)
- Z = depth = across table WIDTH (152.5cm + ~110cm offset)

**What happened:**
1. `physics_model.py` ORIGINALLY had correct axis comments ("X: Along table length")
2. `trajectory_predictor.py` had WRONG comments ("Z = along table length") — these were backwards
3. In initial review, I incorrectly "fixed" physics_model.py to match trajectory_predictor.py's wrong comments
4. User clarified the actual physical setup — confirmed X=length, Z=width
5. Reverted physics_model.py to original correct comments, fixed trajectory_predictor.py

**Known issue for later:** `trajectory_predictor.py` predict() uses `position_at_z()` for interception.
Since ball travels along X (not Z), this should be `position_at_x()`. Not fixed yet — we're doing velocity validation first.

---

### Change #7: Created Axis Verification Script

**File:** `scripts/test_axis_check.py` — **NEW**

**What:** Simple interactive script to visually confirm axis mapping.
- Shows large X, Y, Z values on camera feed
- Displays delta from reference point (press 'r' to set)
- Shows "dominant movement axis" label in real-time
- Instructions: "Move ball along LENGTH → X changes", etc.
- Run: `python scripts/test_axis_check.py`

---

### Change #8: Camera Pitch Clarification & 20° Fixed Stand

**What:** User built a rigid camera stand with a fixed, definitive 20° pitch angle.

**Why this matters:**
- Previous sessions estimated camera pitch variously as 12-15° (from Mode 1 Y-vs-Z observations) and then 24° (from linear regression of Y vs Z data, R²=0.995)
- The actual pitch was never physically measured — all values were derived from triangulation data
- A rigid stand with known geometry eliminates this guesswork entirely
- 20° pitch is a good compromise: steep enough to see the full table surface, shallow enough to maintain reasonable vertical resolution

**Impact on calibration:**
- All existing calibration data (intrinsics + extrinsics) was captured at the old unknown pitch
- Full recalibration (mono intrinsics + stereo extrinsics) needed with the new stand
- The 3% lateral scale error (α=0.972) may improve with fresh calibration at the known geometry

---

### Note: Z-Axis Behavior During Real Play

**Clarification from user:** Ball is thrown like a normal ping pong serve/rally (not rolled along the table).

**Key observations:**
- Ball bounces during flight — bounces send the ball across the table width
- Between bounces on straight throws along table length: Z stays approximately constant
- During bounces: Z DOES change as the ball deflects across the table width
- This is expected physical behavior and the tracking pipeline handles it correctly

**Implication for trajectory prediction:**
- `predict()` currently uses `position_at_z()` for interception — this is wrong regardless
- Ball primarily travels along X (table length), so `position_at_x()` is the correct interception function
- Z variation from bounces is a secondary effect the predictor should handle naturally

---

### Change #9: Fixed Axis Conventions Across Entire Codebase

**Files modified:**
- `trajectory/trajectory_predictor.py` — **CRITICAL FUNCTIONAL FIX**
- `trajectory/test_trajectory_prediction.py`
- `trajectory/__init__.py`
- `PIPELINE.md`

**Correct axis convention (now enforced everywhere):**
- **X = along table LENGTH** (274 cm, ball travels player-to-player)
- **Y = vertical** (down = positive, camera convention)
- **Z = across table WIDTH / depth** (152.5 cm + ~110 cm camera offset)

**What was wrong:**
1. `predict()` used `position_at_z()` for interception — ball travels along X, not Z
2. `cam_to_robot()` had X/Z mapping swapped: `cam_x → robot_x` and `cam_z → robot_z` — but camera X is table LENGTH (robot_z = from endline) and camera Z is table WIDTH (robot_x = lateral)
3. Parameters named `robot_z_cam`, `target_z`, `cam_x_center` were all referencing the wrong axis
4. 3D trajectory viewer had axis labels "X(width)" and "Z(length)" — backwards
5. PIPELINE.md coordinate system section had X/Z swapped
6. PIPELINE.md said cameras look "along the table length" — they look "across the table width"

**Functional fixes applied:**
- `predict(target_z)` → `predict(target_x)` — now uses `position_at_x()` for interception
- `robot_z_cam` → `robot_x_cam` — camera X coordinate of robot endline
- `_cam_x_center` → `_cam_z_center` — camera Z at center of table width
- `cam_to_robot()`:
  - `robot_x = (cam_z - z_center) * 10` — cam Z (width) → robot X (lateral)
  - `robot_z = |cam_x - x_end| * 10` — cam X (length) → robot Z (from endline)
- Strategy string: `'z_plane'` → `'x_plane'`
- 3D viewer: `_draw_z_plane()` → `_draw_x_plane()`, axis labels fixed

**Note:** `physics_model.py` correctly keeps both `position_at_x()` and `position_at_z()` as generic utility methods. Only the caller (`trajectory_predictor.py`) needed fixing. Camera pitch (20°) is NOT hardcoded anywhere — it's encoded in calibration extrinsic matrices at runtime.

---

## Session: 2026-03-08 (Camera-to-Robot Transform & End-to-End Pipeline)

### Status: Transform pipeline implemented. Verification script ready. Unit audit complete.

---

### Change #10: Rotation-Based Camera-to-Robot Transform

**File:** `trajectory/trajectory_predictor.py`

**What:** Replaced simple axis-swap `cam_to_robot()` with a proper rotation matrix transform that accounts for the camera's 20° pitch.

**Why:** Simple axis swap (cam_z→robot_x, cam_x→robot_y, cam_y→robot_z) ignores the 20° camera pitch. At typical depths (150cm), the pitch mixes Y and Z by ~34cm (sin(20°)×100cm), causing ~100mm cross-talk error in robot coords.

**Implementation:**
- `_build_transform(pos_mm, yaw, pitch, roll)` — builds R = R_euler(ZYX) @ R_optical
- `R_optical` converts OpenCV camera axes to standard frame: cam_z→+X, cam_x→+Y, cam_y→-Z
- `R_euler` applies yaw/pitch/roll in robot base frame
- `cam_to_robot(cx, cy, cz)`: `p_robot = R @ (p_cam * 10) + t` (cm→mm)
- `robot_to_cam(rx, ry, rz)`: `p_cam = R^T @ (p_robot - t) / 10` (mm→cm)
- `set_camera_pose(x, y, z, yaw, pitch, roll)` — update at runtime without restart

**Camera pose constants (lines 60-123):**
- Extensive measurement guide in comments (what to measure, reference frames, sign conventions)
- Current values: pos=(1848.5, 1330, 440.074)mm, yaw=5°, pitch=20°, roll=0°

---

### Change #11: Workspace Constants from robot.h

**File:** `trajectory/trajectory_predictor.py`

**What:** Set workspace to rectangular bounds from robot.h:
```
ROBOT_LIMIT_X = (-500, 500)    mm  — across table width
ROBOT_LIMIT_Y = (-350, 350)    mm  — along table length
ROBOT_LIMIT_Z = (-1100, -700)  mm  — vertical (down = more negative)
ROBOT_HOME    = (0, 0, -900)   mm
MAX_CART_VEL  = 4000 mm/s
MAX_CART_ACC  = 20000 mm/s²
```

**Also added:** `check_reachable(rx, ry, rz, time_available)` — trapezoidal velocity profile estimate for whether robot can reach position in time.

---

### Change #12: Robot Coord Display in test_trajectory_prediction.py

**File:** `trajectory/test_trajectory_prediction.py`

**What:**
- Terminal output: added RobX/RobY/RobZ/WS columns per frame
- Status panel: shows robot target XYZ (mm), IN WORKSPACE / OUT OF RANGE, time to intercept, strategy
- 3D view: rectangular workspace wireframe + HOME marker (via `robot_to_cam()` inverse)
- Post-throw summary: robot intercept coords, workspace status
- `get_robot_command()` integration in main loop

---

### Change #13: Robot Transform Verification Script

**File:** `scripts/test_robot_transform_verify.py` — **NEW**

**What:** Click-to-verify camera→robot coordinate transform. Same triangulation flow as `test_triangulation_verify.py` Mode 1 (DISTANCE), but outputs robot coords (mm) instead of raw camera coords.

**Flow:**
1. SPACE to freeze frame
2. Click point on LEFT image, then SAME point on RIGHT image
3. Triangulates 3D position in camera coords (cm)
4. Transforms to robot coords (mm) via rotation matrix
5. Displays both coordinate sets + workspace status
6. Compare robot XYZ against tape-measure ground truth

**Features:**
- `u` key: reload CAM_POSE_* constants via `importlib.reload()` — tune transform without restarting
- Re-transforms all existing measurements on reload
- `p` key: print summary table of all measurements
- Trigger sync verification at startup

**Verification workflow:** Place object at known position relative to robot base. Measure with tape. Click in script. Compare. If off, adjust CAM_POSE_* values and press 'u' to reload. Iterate.

---

### Change #14: Fixed Docstring Workspace Values

**File:** `scripts/test_robot_transform_verify.py` (lines 28-30)

**What:** Docstring referenced stale elliptic workspace values (±790mm, ±540mm, -721 to -1000mm). Updated to match actual rectangular constants (±500mm, ±350mm, -700 to -1100mm).

---

### Unit Audit: Triangulation → Transform Pipeline

**Verified the full unit chain is consistent:**

| Stage | Input | Output | Units |
|-------|-------|--------|-------|
| Calibration | `checkerboard_box_size_scale = 3.18` | — | cm |
| `triangulate()` | rectified pixel pairs | `(X, Y, Z)` | cm (calibration units) |
| `detect_and_triangulate()` | stereo frames | `position_3d` tuple | cm |
| `cam_to_robot()` | `(cx, cy, cz)` cm | `(rx, ry, rz)` mm | cm→mm (×10 internally) |
| `robot_to_cam()` | `(rx, ry, rz)` mm | `(cx, cy, cz)` cm | mm→cm (÷10 internally) |
| `check_workspace()` | `(rx, ry, rz)` mm | bool | mm |
| `add_position()` | `(x, y, z)` cm | — | cm (stored as-is) |
| `predict()` | — | `target_x` in cm | cm |

**Conclusion:** All units are correctly aligned. Triangulation outputs cm, predictor works in cm internally, `cam_to_robot()` handles the cm→mm conversion. No changes needed.

---

---

## Session: 2026-03-15 (Integration Pipeline v2 + Velocity in UART)

### Integration Pipeline v2 Snapshot

**File:** `scripts/test_integration_simple.py`

**Status:** Working end-to-end. Predicts ball trajectory, sends intercept point to STM32. Direction is correct, accuracy sufficient for further testing. Labeled as **v2** for future reference.

**Key characteristics of v2:**
- Stereo triangulation → points-based Kabsch transform (R, t, scale=10.0) → robot frame
- RobotPredictor: regression velocity with gravity correction, Euler integration with air drag
- Workspace: elliptic cylinder (790x540mm, Z=-1050 to -720mm)
- Auto-homing after intercept completion, state machine UART flow
- Capture-time fix applied (timestamp from grab() not after processing)

---

### Change #23: Timestamp Bias Fix

**File:** `tracking/stereo_triangulator.py`, `scripts/test_integration_simple.py`

**What:** Moved `time.perf_counter()` call from after `update()` returns to right after the two `grab()` calls inside `update()`. Frame timestamp is now stored as `result['capture_time']`.

**Why:** The old placement timestamped ~10-20ms late (after detect + rectify + triangulate), causing the intercept `time` field to overestimate latency.

---

### Change #24: Ball Velocity in UART Message

**Files:**
- `motor_control/datatypes/shared_types.h` — `TARGET_MSG_FLOAT_COUNT` 7→10, added `TARGET_MSG_VX/VY/VZ` enum fields, added `vec3 ball_velocity` to `target_t`
- `motor_control/datatypes/mailbox.c` — parse velocity from float array
- `motor_control/app/robot.h` — added `vec3 ball_vel` to `robot_target_t`
- `motor_control/app/robot.c` — copy velocity in `robot_set_target_from_mail()`
- `top_level_control/comm_function/transmit_over_uart.py` — 10-float struct, velocity params
- `top_level_control/scripts/test_integration_simple.py` — pass `vx/vy/vz` from intercept dict

**Message format (10 floats, little-endian):**
```
[type, x_mm, y_mm, z_mm, vx_mm_s, vy_mm_s, vz_mm_s, intercept_time_s, time_sent_s, timestamp_s]
```

**Velocity source:** Ball velocity at the predicted intercept point, from Euler integration (includes gravity + air drag). Already in robot frame (mm/s) since prediction runs entirely in robot frame.

---

## Next Steps (updated 2026-03-15)

1. **Flash updated firmware** — TARGET_MSG_FLOAT_COUNT changed from 7→10, must flash both sides together
2. **Test velocity data** — Verify STM32 receives correct velocity values (print `ball_vel` on STM32 side)
3. **Use velocity for strike planning** — Ball velocity enables paddle angle/speed computation for return shots
4. **Continue accuracy testing** — Refine prediction with real throw data


## Session: 2026-03-18 (3D Kalman Filter State Estimation)

### Status: Implemented. Legacy fallback preserved. Needs FilterPy install + R/Q tuning before live use.

---

### Change #23: Added 3D Ball State Estimator

**File:** `trajectory/ball_state_estimation.py` — **NEW**

**What:** Added `BallStateEstimator3D`, a robot-frame linear Kalman filter for ball state estimation.

**Model:**
- State: `(px, py, pz, vx, vy, vz)`
- Measurement: `(px, py, pz)`
- `dt = t_k - t_(k-1)` from consecutive timestamped robot-frame measurements
- Gravity handled as known input on `z`

**Behavior:**
- First valid point initializes the filter
- Large time gaps reinitialize the filter
- `is_ready()` requires multiple fused updates before the estimate is trusted

**Why:** Replaces raw-point velocity fitting with a proper current-state estimate, giving smoother position/velocity for downstream trajectory prediction.

---

### Change #24: Integrated KF Into RobotPredictor With Safe Fallback

**File:** `trajectory/robot_predictor.py`

**What:** `RobotPredictor` now uses `BallStateEstimator3D` for current state estimation when available, while keeping the existing forward physics and workspace scan intact.

**Logic changes:**
- `add_position()` still does the same raw timestamp / jump / speed gating
- Accepted robot-frame points now update the KF instead of directly driving regression velocity
- `predict_intercept()` now seeds from the current estimated state, not the last raw point
- Observed bounces reset the estimator because bounce is a state discontinuity
- If `filterpy` is unavailable, predictor falls back to the previous regression-based estimator

**Why:** This is the least intrusive upgrade path. Measurement intake, bounce detection, and forward prediction stay familiar, while only the current-state estimate changes.

**Note:** `R` and `Q` are conservative starter values. If the KF looks noisy or laggy, tune them from replayed throws before trusting live hits.

---

### Change #25: Added Live KF vs Legacy Trajectory Comparison Tool

**File:** `scripts/test_trajectory.py` — **NEW**

**What:** Added a simple live comparison script using the same main modules as `test_integration_simple.py`:
- `StereoTriangulator`
- `cam_to_robot()`
- `RobotPredictor`
- `BallStateEstimator3D`

**Behavior:**
- Blue line: trajectory from KF-estimated current state
- Red line: trajectory from legacy regression + raw-position start state
- White points: measured robot-frame positions

**Why:** Gives a direct visual check that the KF path is actually improving trajectory stability before using it in the hit pipeline.

---

### Support Updates

**Files:**
- `trajectory/__init__.py`
- `requirements.txt`

**What:**
- Exported `BallStateEstimator3D` from `trajectory`
- Added `scipy` and `filterpy` to requirements

**Why:** Keeps the estimator importable from the shared trajectory package and makes the dependency explicit.

---
