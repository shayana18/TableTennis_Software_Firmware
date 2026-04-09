# Trajectory Prediction — Version History

Tracks the evolution of each subsystem in `trajectory/robot_predictor.py` and related modules. Each section shows what changed, why, and what results were observed.

---

## 1. Velocity Estimation

### V0: Simple Difference (deprecated `velocity_estimator.py`)

**Method:** `velocity = (last_position - first_position) / total_time`

**Code:** `VelocityEstimator.estimate_simple()` in `trajectory/velocity_estimator.py`

**Limitations:**
- Only uses 2 points — extremely noisy
- Single outlier at either endpoint corrupts the entire estimate
- No gravity awareness — Y-axis velocity systematically biased by freefall acceleration
- No drag awareness

**Status:** Deprecated. Only used by deprecated `TrajectoryPredictor` as a fallback.

---

### V1: Linear Regression (deprecated `velocity_estimator.py`)

**Method:** `np.polyfit(t, positions[:, axis], 1)` per axis — fits a line through all buffered positions.

**Code:** `VelocityEstimator.estimate_regression()` in `trajectory/velocity_estimator.py`

**Improvements over V0:**
- Uses all N buffered points, not just first/last
- Least-squares fit averages out measurement noise
- Configurable buffer size

**Limitations:**
- No gravity correction — Y (vertical) velocity is biased by `0.5 * g * dt`
- No drag correction — horizontal velocities underestimated by ~10-15%
- Operated in camera frame (cm), requiring axis decomposition for gravity

**Status:** Deprecated. Kept for backward compatibility with analysis scripts.

---

### V2: Gravity-Corrected Regression (`robot_predictor.py`, 2026-03-10)

**Method:** Least-squares regression in robot frame. Before fitting Z, subtract the known gravitational displacement:

```
z_corrected = z_measured - 0.5 * GRAVITY_Z * dt²
vz = lstsq(A, z_corrected)
```

X and Y are fit with plain linear regression (no gravity component).

**Code:** `RobotPredictor._estimate_velocity()` in `trajectory/robot_predictor.py`

**Key parameters:**
- `MIN_POINTS = 6` (need enough frames for stable regression)
- `MIN_TIME_SPAN = 0.08s` (80ms of data, ~8 frames at 100fps)
- `BUFFER_SIZE = 15` (sliding window)
- `MAX_SPEED = 15000 mm/s` (reject absurd velocities)

**Improvements over V1:**
- Works in robot frame — gravity is simply `(0, 0, -9810)` mm/s², no axis decomposition needed
- Gravity correction gives accurate vertical velocity
- Integrated into `RobotPredictor` class (no separate estimator module)

**Limitations:**
- X/Y treated as constant velocity — ignores air drag deceleration
- At typical speeds (3000-4000 mm/s), drag causes ~2800 mm/s² deceleration
- Over an 80-150ms regression window, this underestimates velocity by ~10-15%
- Results in ~150-200mm position error in forward prediction

**Test results (v2/v3 integration testing, 2026-03-10):**
- 13 throws sent to robot, all reached STM32
- Accuracy: poor — robot ~10-30cm off from actual ball path
- Root cause: no air drag in velocity estimation or prediction
- Per-arc Vx errors: 1-8% (acceptable within single arcs)

---

### V3: Drag-Aware Two-Pass Regression (`robot_predictor.py`, 2026-03-17)

**Method:** Two-pass least-squares. Pass 1 is identical to V2 (gravity-only). Pass 2 subtracts the drag-induced position offset using the rough velocity from Pass 1, then re-fits:

```
# Pass 1: standard (gravity-only on Z)
vx_rough = lstsq(A, xs)
vy_rough = lstsq(A, ys)
vz_rough = lstsq(A, zs - 0.5*g*dt²)

# Pass 2: subtract drag offsets
speed0 = |v_rough|
drag_factor = 0.5 * DRAG_K * speed0
xs_corrected = xs + drag_factor * vx_rough * dt²
ys_corrected = ys + drag_factor * vy_rough * dt²
zs_corrected = zs_grav + drag_factor * vz_rough * dt²
vx, vy, vz = lstsq(A, corrected)
```

**Physics:** Drag deceleration is `a_drag = -DRAG_K * |v| * v_component`. Over time dt from reference, this displaces positions by `≈ 0.5 * DRAG_K * |v| * v_component * dt²`. This has the same quadratic form as gravity, so we can subtract it before regression.

**Why one pass suffices:** The correction is ~10% of displacement. Error in the correction from using rough velocity is ~10% of 10% = 1% — well below measurement noise.

**Key constants:**
- `DRAG_K = 0.000112 mm⁻¹` (from Cd=0.40, ρ_air=1.2 kg/m³, A=π×20², m=2.7g)

**Improvements over V2:**
- Recovers the ~10-15% velocity underestimate from drag
- All three axes corrected (not just Z)
- Negligible computational cost — 3 extra lstsq calls on a 6-15 point matrix

**Expected impact:**
- ~150-200mm reduction in prediction position error
- More accurate intercept timing

**Status:** Current (2026-03-17). Pending real-robot validation.

---

## 2. Trajectory Forward Simulation

### V0: Closed-Form Kinematics (deprecated `physics_model.py`)

**Method:** Analytical equations: `pos(t) = p0 + v*t + 0.5*g*t²`

**Code:** `PhysicsModel.predict_position()` in `trajectory/physics_model.py`

**Limitations:**
- No air drag — accurate only for short predictions (~100ms)
- Camera frame — required gravity decomposition along camera pitch angle
- Gravity decomposition: `g_y = 981 * cos(pitch)`, `g_z = -981 * sin(pitch)` — errors amplified through rotation matrix

**Status:** Deprecated for real-time. Kept for post-hoc analysis.

---

### V1: Euler Integration with Gravity Only (`test_integration_simple.py` v1-v3, 2026-03-10)

**Method:** Forward Euler in robot frame:

```
v += (0, 0, GRAVITY_Z) * dt
pos += v * dt
```

Scan 1.5s forward at 5ms steps. No drag.

**Test results (v1, ~30 throws):**
- 23 intercepts sent, 7 rejected by STM32 ("TARGET OUT OF WORKSPACE")
- Only ~2 of 30 throws had visually accurate intercepts
- Most predictions ~10-30cm off

**Test results (v3, 3 throws, added proximity filter):**
- Still ~5-15cm off
- Confirmed: air drag is the dominant error source

---

### V2: Euler Integration with Gravity + Air Drag (`test_integration_simple.py` v4, 2026-03-10)

**Method:** Semi-implicit Euler with quadratic drag:

```
speed = |v|
drag = DRAG_K * speed
a = (-drag*vx, -drag*vy, GRAVITY_Z - drag*vz)
v += a * dt
pos += v * dt
```

**Code:** `RobotPredictor._step_euler()` in `trajectory/robot_predictor.py`

**Key parameters:**
- `SCAN_DURATION = 1.5s`
- `SCAN_DT = 0.005s` (5ms steps)
- `MIN_TIME_HIT = 0.10s` (minimum robot reaction time)
- `DRAG_K = 0.000112 mm⁻¹`
- `GRAVITY_Z = -9810.0 mm/s²`

**Drag significance:** At v=3000 mm/s, drag acceleration = DRAG_K × 3000² = 1008 mm/s². That's ~10% of gravity. Over 400ms flight: no-drag model overshoots Y by ~84mm, undershoots Z by ~29mm.

**Status:** Current (since 2026-03-10). The forward simulation itself hasn't changed — what changed in V3 is how velocity is estimated before entering the simulation.

---

### V3: Euler Integration with Bounce Reflection (2026-03-17)

**Method:** Same Euler integration as V2, plus bounce detection at table surface:

```
# After each Euler step:
if z < Z_TABLE_SURFACE and z_prev >= Z_TABLE_SURFACE:
    frac = (Z_TABLE_SURFACE - z_prev) / (z - z_prev)
    # Position at bounce point
    xb, yb, zb = interpolate(prev, curr, frac)
    # Reflect velocity
    vz = -vz * RESTITUTION_COEFF   # 0.85
    vx *= FRICTION_COEFF            # 0.95
    vy *= FRICTION_COEFF            # 0.95
    # Complete remaining timestep
    step_euler(xb, yb, zb, vx, vy, vz, dt * (1 - frac))
```

**Code:** `RobotPredictor._apply_bounce()` in `trajectory/robot_predictor.py`

**Key constants:**
- `Z_TABLE_SURFACE = -1150.0 mm` (table top in robot frame)
- `RESTITUTION_COEFF = 0.85` (vz damping — ping pong on MDF)
- `FRICTION_COEFF = 0.95` (vx/vy damping — tangential friction)
- `MAX_BOUNCES = 2` per prediction scan

**What this fixes:**
- Previously: ball passed through table surface in simulation, trajectory lost
- Now: ball bounces realistically and can re-enter workspace post-bounce
- Enables predicting interception of bounced shots

**Performance:** 1 extra comparison per scan step when no bounce occurs. Negligible.

**Status:** Current (2026-03-17). Pending real-robot validation.

---

## 3. Workspace Model

### V0: Rectangular Bounds (deprecated `trajectory_predictor.py`)

```
ROBOT_LIMIT_X = (-500, 500) mm
ROBOT_LIMIT_Y = (-350, 350) mm
ROBOT_LIMIT_Z = (-1100, -700) mm
```

Simple axis-aligned box. Did not match actual firmware IK geometry.

---

### V1: Oversized Rectangle (`test_integration_simple.py` v4)

```
WS_HALF_X = 869 mm (790 * 1.1)
WS_HALF_Y = 594 mm (540 * 1.1)
Z_MIN = -1050, Z_MAX = -721
```

10% oversized — let firmware IK reject if truly unreachable. Reduced "TARGET OUT OF WORKSPACE" rejections but still didn't match actual elliptic workspace.

---

### V2: Elliptic Cylinder with Safety Margin (`workspace.py`, current)

```
ELLIPSE_A = 790 * 0.9 = 711.0 mm  (X semi-axis, 10% margin)
ELLIPSE_B = 540 * 0.9 = 486.0 mm  (Y semi-axis, 10% margin)
Z_MIN = -1050 mm (25mm margin from robot.h -1025)
Z_MAX = -720 mm  (10mm margin from robot.h -731)
```

**Code:** `in_workspace()` and `clamp_to_workspace()` in `trajectory/workspace.py`

Matches firmware `check_workspace()` elliptic check from `robot.h`, with 10% safety margin to avoid IK rejections near the boundary.

**Clamp fallback:** If no trajectory point enters workspace, clamp the nearest point (up to `MAX_CLAMP_DIST = 350 mm`). This catches balls that just miss the workspace edge.

**Status:** Current (since 2026-03-15).

---

## 4. Observed Bounce Detection (in input data)

### V0: No Bounce Detection

Pre-bounce and post-bounce data mixed in the same buffer. If ball bounced (Z reversed), the regression fit nonsensical velocities across the discontinuity.

**Evidence (2026-03-08 session):** Bounce Vx/Vz preservation was attempted — preserving pre-bounce velocity and blending after 10 points. Result: HARMFUL. Pre-bounce arcs had only 5-6 points (unreliable). Throw 5 showed 60 cm/s Vx discontinuity and 125 cm/s Vz discontinuity at blend threshold. Reverted completely.

---

### V1: Bounce Detection via Z Reversal (2026-03-17)

**Method:** Track minimum Z since reset. If ball was falling (Z decreasing) and now rising for 2+ consecutive frames, and has fallen at least 50mm from initial Z — declare bounce.

**Code:** `RobotPredictor._detect_observed_bounce()` and `_handle_observed_bounce()` in `robot_predictor.py`

**On bounce detection:**
1. Save last 2 positions
2. Clear entire buffer
3. Reset velocity estimate
4. Re-seed buffer with saved points
5. Increment `_bounce_count`

**Key parameters:**
- `MIN_BOUNCE_FALL_Z = 50.0 mm` — minimum descent before accepting bounce (avoids false triggers from noise)
- `BOUNCE_RISE_FRAMES = 2` — consecutive rising frames needed to confirm

**Why clear buffer instead of blend:**
- The 2026-03-08 session proved that blending pre/post-bounce data is harmful
- Clean reset with 2 seed points gives the regression a fresh start on the post-bounce arc
- 4 more frames (~40ms) and the velocity estimate is back online

**Performance:** 1 comparison per frame (Z vs Z_prev). Negligible.

**Status:** Current (2026-03-17). Pending real-robot validation.

---

## 5. Coordinate Transforms

### V0: Simple Axis Swap (deprecated `trajectory_predictor.py`, pre-2026-03-08)

```
robot_x = (cam_z - z_center) * 10
robot_y = |cam_x - x_end| * 10
robot_z = -(cam_y) * 10
```

**Problem:** Ignores 20° camera pitch. At typical depths (150cm), pitch mixes Y and Z by ~34cm, causing ~100mm cross-talk error in robot coords.

---

### V1: Rotation Matrix Transform (deprecated `trajectory_predictor.py`, 2026-03-08)

**Method:** Full rotation matrix: `R = R_euler(ZYX) @ R_optical`
- `R_optical`: converts OpenCV axes to standard frame (cam_z→+X, cam_x→+Y, cam_y→-Z)
- `R_euler`: applies yaw/pitch/roll in robot frame
- `cam_to_robot(cx, cy, cz) = R @ (p_cam * 10) + t`

**Problem:** Still operated in camera frame for prediction, then transformed. Gravity decomposition along tilted camera axes amplified errors.

---

### V2: Points-Based Kabsch Transform (current `comm_function/points_based_transform.py`, 2026-03-15)

**Method:** Kabsch SVD alignment from 4+ manually-clicked stereo points with known robot coordinates. Solves for R, t, and scale in one step.

**Used by:** `test_integration_simple.py` — triangulate in camera frame (cm), transform to robot frame (mm) via `cam_to_robot(R, t, scale, cx, cy, cz)`, then predict entirely in robot frame.

**Advantage:** No manual measurement of camera pose needed. Directly calibrated from physical reference points.

**Status:** Current.

---

## 6. Integration Test Results Timeline

| Date | Version | Throws | Sent to Robot | Accuracy | Key Issue |
|------|---------|--------|---------------|----------|-----------|
| 2026-03-10 | v1 | ~30 | 23 (7 rejected) | ~2 accurate | Manual clear needed between throws |
| 2026-03-10 | v2 | 6 | 1 | Poor | HOME→COMPLETED Q infinite loop |
| 2026-03-10 | v2 (fix) | 13 | 13 | ~10-30cm off | No drag model |
| 2026-03-10 | v3 | 3 | 3 | ~5-15cm off | No drag model (confirmed) |
| 2026-03-10 | v4 | — | — | Pending | Added drag model + rect workspace |
| 2026-03-15 | v5 | — | — | Pending | Elliptic workspace + points-based transform |
| 2026-03-17 | v6 | — | — | Pending | Drag-aware velocity + bounce detection |

---

## 7. Calibration History & Scale Accuracy

| Date | Images | cam0 fx | cam1 fx | Baseline | Gravity Measured | Scale Error |
|------|--------|---------|---------|----------|-----------------|-------------|
| Pre-03-07 | 15 | ~575? | ~530? | ~42.8cm | 824 cm/s² | 19% (α=1.19) |
| 2026-03-07 | 30+ | 545.15 | 531.07 | 42.77cm | 1010 cm/s² | 3% (α=0.972) |
| 2026-03-08 | — | 576.13 | 532.65 | — | — | BAD (k3 overfitting) |
| 2026-03-10 | 30+ | 512.8 | 524.0 | 23.4cm | — | Pending verification |

---

## 8. Key Constants Reference (current values)

### Physics
| Constant | Value | Source |
|----------|-------|--------|
| `GRAVITY_Z` | -9810.0 mm/s² | Standard gravity |
| `DRAG_K` | 0.000112 mm⁻¹ | Cd=0.40, ρ=1.2e-9 g/mm³, A=π×20², m=2.7g |
| `Z_TABLE_SURFACE` | -1150.0 mm | Measured table height in robot frame |
| `RESTITUTION_COEFF` | 0.85 | Ping pong ball on MDF table |
| `FRICTION_COEFF` | 0.95 | Tangential damping on bounce |
| `MAX_BOUNCES` | 2 | Per prediction scan |

### Predictor Tuning
| Constant | Value | Rationale |
|----------|-------|-----------|
| `BUFFER_SIZE` | 15 | Sliding window for regression |
| `MIN_POINTS` | 6 | Was 8 → too restrictive at 100fps |
| `MIN_TIME_SPAN` | 0.08s | Was 0.06 → need 80ms for stable fit |
| `MAX_SPEED` | 15000 mm/s | Reject absurd velocities |
| `MAX_JUMP` | 400 mm | Reject position outliers |
| `GAP_RESET` | 0.12s | Reset buffer after >120ms gap |
| `SCAN_DURATION` | 1.5s | Was 1.0 → scan further forward |
| `SCAN_DT` | 0.005s | 5ms prediction steps |
| `MIN_TIME_HIT` | 0.10s | Was 0.15 → allow faster reaction |
| `MAX_PREDICT_Y` | 1400 mm | Don't predict when ball too far |

### Workspace
| Constant | Value | Rationale |
|----------|-------|-----------|
| `ELLIPSE_A` | 711.0 mm | 790 × 0.9 (10% safety margin) |
| `ELLIPSE_B` | 486.0 mm | 540 × 0.9 (10% safety margin) |
| `Z_MIN` | -1050 mm | 25mm margin from robot.h -1025 |
| `Z_MAX` | -720 mm | 10mm margin from robot.h -731 |
| `MAX_CLAMP_DIST` | 350 mm | Max distance for clamp fallback |
