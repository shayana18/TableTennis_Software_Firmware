# Pipeline Review: `test_integration_simple.py`

**Date:** 2026-03-15
**Script version:** v5 (ellipse workspace, clamp-to-workspace fallback)

---

## 1. High-Level Pipeline

The script chains six stages into a real-time loop running at up to 100 fps:

```
┌─────────────────────────────────────────────────────────┐
│ 1. STEREO CAPTURE + DETECTION (StereoTriangulator)      │
│    grab()/retrieve() both cameras → BallDetector (MOG2) │
│    → undistort + rectify → DLT triangulate → (cm)       │
│                                                         │
│ 2. COORDINATE TRANSFORM (points-based Kabsch SVD)       │
│    p_robot_mm = R @ (p_cam_cm × scale) + t              │
│                                                         │
│ 3. POSITION BUFFERING + OUTLIER REJECTION               │
│    RobotPredictor.add_position() → deque(maxlen=15)     │
│    Rejects: speed > 15 m/s, jump > 400 mm, gap > 120ms │
│                                                         │
│ 4. VELOCITY ESTIMATION (least-squares regression)       │
│    Requires ≥ 6 points spanning ≥ 80 ms                 │
│    Z-axis: gravity-corrected regression                 │
│                                                         │
│ 5. TRAJECTORY PREDICTION (Euler integration)            │
│    Gravity (9810 mm/s²) + quadratic air drag            │
│    Scan 1.5 s forward in 5 ms steps                     │
│    Return first workspace entry or clamped fallback     │
│                                                         │
│ 6. UART TRANSMISSION                                    │
│    Binary packet: [type, x, y, z, t_intercept, t_sent,  │
│    t_frame] → STM32  (throttle: 30 ms min interval)    │
└─────────────────────────────────────────────────────────┘
```

### State machine (UART flow)

```
IDLE  ──send intercept──►  PENDING('intercept')
                             │
                      STATE: MOVE (robot executing)
                             │
                      COMPLETED Q
                             │
                    ──send HOME──►  PENDING('homing')
                             │
                      COMPLETED Q
                             │
                    ──clear──►  IDLE (ready for next throw)
```

---

## 2. What's Correct

### 2.1 Stereo Triangulation
- **Stereo rectification** applied correctly via `cv2.stereoRectify` before triangulation; epipolar lines become horizontal so DLT only needs x-disparity.
- **Four-layer validation** (disparity range, epipolar error, Z range, reprojection error) is well-structured and catches bad geometry.
- **Epipolar threshold** (15 px) was tightened from 35 px to surface bad rectification early — good defensive choice.
- **`grab()`/`retrieve()` pattern** is correct for hardware-triggered synchronization.
- **BallDetector** is used as a shared instance (not re-implemented), maintaining single source of truth.

### 2.2 Coordinate Transform
- **Points-based (Kabsch SVD) transform** is mathematically sound: rigid rotation + translation, no scale ambiguity if calibration points were measured in matching units.
- The `cam_scale` multiplier handles unit conversion (cm → mm) cleanly: `R @ (p_cm * scale) + t`.
- **Inverse transform** in `draw_intercept_marker()` correctly uses `R.T` (rotation transpose = inverse for orthonormal R).

### 2.3 Velocity Estimation
- **Gravity-corrected regression** on the Z axis (`z_corrected = z - 0.5*g*dt²`) is the right approach — removes gravity bias from the linear fit so the extracted `vz` is the launch velocity, not the time-averaged velocity.
- **Reference time** is anchored to the most recent sample (`t_ref = pts[-1][3]`), which keeps the `dt` values small and numerically stable.

### 2.4 Trajectory Prediction
- **Euler integration with drag** is appropriate for a ping-pong ball. The drag coefficient (`k = 0.000112 mm⁻¹`) matches the standard aerodynamic model (Cd=0.40, ρ_air=1.2 kg/m³, m=2.7 g, d=40 mm).
- **Workspace check** uses the firmware's own ellipse equation `(x/A)² + (y/B)² ≤ 1` with 5% safety margin — matching the IK rejection boundary avoids STM32 rejections.
- **Clamp-to-workspace fallback** with a max-distance guard (350 mm) is a reasonable degraded-mode strategy.
- **MIN_TIME_HIT = 100 ms** ensures the robot has enough planning time.

### 2.5 UART Protocol
- **Binary struct** (7 floats, 28 bytes) is compact and unambiguous.
- **Latency compensation** (`t_adjusted = intercept_time - latency`) accounts for processing delay.
- **30 ms TX throttle** prevents flooding the STM32 serial buffer.
- **Significant-change gating** (80 mm delta) for updates avoids jitter-induced re-planning.

### 2.6 Robustness
- **Auto-clear after each throw cycle** (intercept → COMPLETED Q → HOME → COMPLETED Q → ready) eliminates manual 'c' presses.
- **Graceful handling** of STM32 rejections (`TARGET OUT OF WORKSPACE`, `PLANNING FAILED`) resets state correctly.
- **Intercept JSON log** (`intercept_log.json`) saves every throw for post-session analysis.

---

## 3. Potential Errors and Mismatches

### 3.1 CRITICAL: `cam_to_robot()` uses *different* transform than `StereoTriangulator.cam_to_robot()`

The script imports `load_points_based_transform()` and uses its own standalone `cam_to_robot()` function (line 65):

```python
def cam_to_robot(R, t, scale, cam_x, cam_y, cam_z):
    p = R @ (np.array([cam_x, cam_y, cam_z]) * scale) + t
    return float(p[0]), float(p[1]), float(p[2])
```

Meanwhile, `StereoTriangulator` has its own built-in `cam_to_robot()` method (referenced in MEMORY.md) that uses a **rotation-matrix approach** with Euler angles (yaw=185°, pitch=20°, roll=0°) and explicit R_optical axes swap.

**Risk:** These two transforms may not agree. The points-based transform is data-driven (Kabsch SVD from measured point pairs), while the Euler-angle approach in `StereoTriangulator` is model-driven. If the points-based transform JSON was calibrated independently, any drift between the two methods means `draw_intercept_marker()` (which inverts the points-based R) will be correct, but predictions could be off if the Kabsch fit was done with stale or inaccurate point pairs.

**Recommendation:** Verify the residual RMSE of the points-based transform JSON. If RMSE > 10 mm, recollect calibration points. Consider printing the RMSE at startup.

### 3.2 GRAVITY SIGN CONVENTION

```python
GRAVITY_Z = -9810.0  # mm/s², robot Z is vertical, negative = down
```

This means the robot frame has **Z pointing up** (negative = down). In `_step_euler()`:

```python
az = GRAVITY_Z - drag * vz   # gravity pulls Z more negative
```

And in `_estimate_velocity()`:

```python
zs_corrected = zs - 0.5 * GRAVITY_Z * dt * dt
# Since GRAVITY_Z is negative and dt² is positive,
# this ADDS 0.5 * 9810 * dt² to zs → removes downward drift
```

**This is correct**, but the comment says "robot Z is vertical, negative = down" while the workspace Z range is:

```python
Z_MIN = -1050.0 + 25  = -1025  (lower, more negative)
Z_MAX = -721.0  - 10  = -731   (upper, less negative)
```

So the entire workspace is at negative Z, and the ball falls toward more-negative Z. This is self-consistent: gravity pulls Z more negative, workspace is in the negative-Z region. **No error, but the sign convention is non-standard and worth documenting explicitly.**

### 3.3 `frame_ts` IS SET AFTER `update()`, NOT AT CAPTURE TIME

```python
result = self.triangulator.update()    # line 706 — includes grab + retrieve + detect + triangulate
frame_ts = time.perf_counter()         # line 707 — timestamp AFTER all processing
```

The `frame_ts` is used to:
1. Timestamp the ball position fed to `add_position()` (line 723)
2. Compute latency for UART time adjustment (line 588)

**Problem:** If `update()` takes 10-30 ms (MOG2 + contour + triangulation), the timestamp is systematically late. At 5 m/s ball speed, 20 ms of timestamp error = 100 mm position error in velocity estimation. Since regression fits velocity over ~80 ms of data, the *relative* timing between consecutive `frame_ts` values is approximately correct (each includes similar processing delay), so **velocity magnitude is mostly unaffected**. However, the **absolute `time` field in the intercept** will overestimate latency, causing `t_adjusted` to be too small by ~one frame period.

**Recommendation:** Move `frame_ts = time.perf_counter()` to immediately after `grab()` returns (or better, inside `update()` right after the synchronized grab). This is a 10-20 ms systematic bias on intercept timing.

### 3.4 VELOCITY REGRESSION DOES NOT ACCOUNT FOR DRAG

`_estimate_velocity()` corrects for gravity on the Z axis but assumes constant velocity on X and Y:

```python
vx = float(np.linalg.lstsq(A, xs, rcond=None)[0][0])   # linear fit: x = vx*dt + x0
vy = float(np.linalg.lstsq(A, ys, rcond=None)[0][0])
```

For a ping-pong ball at 5 m/s, drag deceleration ≈ `0.000112 × 5000 × 5000 / 5000 = 0.56 mm/s² per ms` — i.e., ~2.8 m/s² total drag deceleration. Over the 80 ms regression window, drag reduces speed by ~220 mm/s (4.4% of initial). The linear fit averages this out, producing a velocity estimate biased toward the *mean* velocity over the window rather than the *current* velocity.

**Impact:** The predicted intercept position could be off by a few cm over a 0.5 s prediction. The Euler integration forward does include drag, so it partially compensates. **Low priority** but could improve accuracy with a quadratic fit.

### 3.5 APPROACH DIRECTION FILTER AXIS

```python
MIN_APPROACH_VY = -200.0     # mm/s
APPROACH_Y_THRESHOLD = 600.0 # mm

def _ball_approaching(self):
    y_now = self.positions[-1][1]    # robot Y
    vy = self.velocity[1]            # robot vy
    if abs(y_now) < self.APPROACH_Y_THRESHOLD:
        return True
    return vy < self.MIN_APPROACH_VY
```

This checks the **Y axis** of the robot frame. According to MEMORY.md's axis mapping:
- Robot Y = along table length (cam X) — the direction ball travels player-to-player.

So `vy < -200 mm/s` means "ball is moving toward negative-Y" — this assumes the ball approaches the robot from **positive Y**. If the robot is at Y=0 and the thrower is at positive-Y, this is correct. If the geometry is reversed, the filter blocks all valid throws.

**Recommendation:** Validate this filter with actual throw data. A ball approaching at `vy = +3000 mm/s` (toward positive-Y) would be rejected if the robot is on the positive-Y side. Check the sign by looking at recorded `vel_y_mm_s` values in `intercept_log.json`.

### 3.6 `MAX_PREDICT_Y` IS ONE-SIDED

```python
if y_now > self.MAX_PREDICT_Y:   # line 226
    return None
```

Only checks `y_now > 1400`, not `y_now < -1400`. This means the proximity filter is **asymmetric** — it only rejects balls far in the positive-Y direction. If the robot frame has the thrower at positive-Y, this is correct. But if the ball somehow ends up at Y = -2000 (behind the robot), it would *not* be filtered out.

**Low risk** — the workspace check at intercept time would catch this anyway.

### 3.7 CLAMP-TO-WORKSPACE GEOMETRY BUG

```python
def clamp_to_workspace(x, y, z):
    z_c = max(Z_MIN, min(Z_MAX, z))
    r = math.sqrt((x / ELLIPSE_A) ** 2 + (y / ELLIPSE_B) ** 2)
    if r > 1.0:
        x_c = x / r
        y_c = y / r
    else:
        x_c, y_c = x, y
    dist = math.sqrt((x - x_c) ** 2 + (y - y_c) ** 2 + (z - z_c) ** 2)
    return x_c, y_c, z_c, dist
```

The XY clamping divides by `r = sqrt((x/A)² + (y/B)²)`. This projects the point along the line from the origin to the ellipse boundary. But for an **ellipse** (A ≠ B), this does **not** give the nearest point on the ellipse. The nearest point on an ellipse to an external point is found by solving a quartic or using iterative methods.

**Example:** For a point at (1000, 0, -900) with A=750.5, B=513:
- `r = 1000/750.5 = 1.332`
- `x_c = 1000/1.332 = 750.5`, `y_c = 0` — **this IS the nearest point** (on the major axis).

But for (500, 500, -900):
- `r = sqrt((500/750.5)² + (500/513)²) = sqrt(0.444 + 0.950) = 1.180`
- `x_c = 500/1.180 = 424`, `y_c = 500/1.180 = 424`
- The true nearest point on the ellipse is different (would require solving the orthogonal projection).

**Impact:** The clamped position is approximate, not exact. The error is typically <20 mm for points within the 350 mm clamp-dist budget. **Low priority** — the robot's motion planner will interpolate to a reachable position anyway.

### 3.8 EULER INTEGRATION STEP SIZE

`SCAN_DT = 0.005` (5 ms) over `SCAN_DURATION = 1.5 s` = 300 integration steps. For Euler integration of drag + gravity:

- **Gravity** is constant, so any step size is exact.
- **Drag** is velocity-dependent and non-linear. The relative error per step is approximately `O(DRAG_K × speed × dt)`. At 5 m/s: `0.000112 × 5000 × 0.005 ≈ 0.003` — 0.3% per step, accumulating to ~1% total over 0.5 s.

**Acceptable.** RK4 would be more accurate but Euler is fine at 5 ms steps for this speed regime.

### 3.9 NO BOUNCE DETECTION

The predictor scans the trajectory forward but does not model table bounces. If the ball bounces off the table surface, the actual trajectory diverges from the predicted parabola. The `in_workspace()` check may find a point that the ball never actually reaches.

**Mitigation:** For throws aimed at the robot's workspace (below the table surface), bounces are unlikely. But for low, fast throws that graze the table edge, this could produce phantom intercepts.

### 3.10 `intercept_sent = True` AT STARTUP

```python
self.intercept_sent = True    # line 508 — set in send_home_and_wait()
```

After homing, `intercept_sent` is set `True`. When the user presses 'g' to enable the gate:

```python
elif key == ord("g"):
    self.run_gate = not self.run_gate
    if self.run_gate:
        self.predictor.reset()
        self.intercept_sent = False    # line 789
```

This correctly clears it. No bug here, but the initial `True` state on line 508 is defensive (prevents accidental sends before gate is toggled). **Correct behavior.**

---

## 4. Recommendations

### High Priority

| # | Issue | Action |
|---|-------|--------|
| 1 | **Timestamp bias** (§3.3) | Move `frame_ts` to right after `grab()` succeeds, or add a `capture_timestamp` to the `update()` return dict. Eliminates ~10-20 ms systematic bias on intercept timing. |
| 2 | **Verify points-based transform** (§3.1) | Print the Kabsch RMSE at startup from the saved JSON. If RMSE > 10 mm, recollect calibration points. Cross-check by triangulating a known static point and comparing cam_to_robot output against measured robot coordinates. |
| 3 | **Validate approach direction** (§3.5) | Record 5+ throws, inspect `vel_y_mm_s` in `intercept_log.json`. Confirm negative-vy means approaching the robot. If sign is wrong, flip `MIN_APPROACH_VY` to `+200`. |

### Medium Priority

| # | Issue | Action |
|---|-------|--------|
| 4 | **Drag-aware velocity fit** (§3.4) | Replace linear regression with quadratic on X and Y, or use a 2-point finite-difference at the most recent pair after regression smoothing. |
| 5 | **Bounce detection** (§3.9) | Add a Z-floor check during Euler scan. If `z < Z_TABLE_SURFACE`, reflect `vz` with a restitution coefficient (~0.85 for ping-pong on MDF). |
| 6 | **Log the points-based transform RMSE** in intercept_log.json so post-analysis can correlate transform quality with intercept accuracy. |

### Low Priority / Polish

| # | Issue | Action |
|---|-------|--------|
| 7 | **Ellipse clamp** (§3.7) | Replace radial projection with iterative nearest-point-on-ellipse (Newton's method, 3-4 iterations). Marginal gain for edge cases. |
| 8 | **Symmetric proximity filter** (§3.6) | Change `y_now > MAX_PREDICT_Y` to `abs(y_now) > MAX_PREDICT_Y` if bi-directional filtering is desired. |
| 9 | **Document Z sign convention** (§3.2) | Add a one-line comment: `# Robot frame: Z up (positive), workspace at Z ∈ [-1025, -731] (below base plate)`. |

### Structural Observations

- **RobotPredictor is defined inline** in the integration script (~200 lines). Consider extracting it to `tracking/robot_predictor.py` if it will be reused by other scripts (e.g., `test_integration_day.py`).
- **No unit tests** for `RobotPredictor`, `in_workspace()`, or `clamp_to_workspace()`. These are pure functions and easy to test.
- **The points-based transform and the Euler-angle `cam_to_robot` in StereoTriangulator are redundant systems.** Long-term, pick one and retire the other to avoid divergence.

---

## 5. Unit Flow Verification

| Stage | Input Units | Output Units | Conversion |
|-------|-------------|--------------|------------|
| Calibration (checkerboard) | px | cm (via `box_size_scale=3.18`) | Baked into intrinsics/extrinsics |
| `StereoTriangulator.update()` | stereo frames | `position_3d` in **cm** | DLT in calibration units |
| `cam_to_robot()` | cm | **mm** | `R @ (p_cm × scale) + t`, scale = cm→mm factor |
| `RobotPredictor` buffer | mm | mm | Identity |
| `_estimate_velocity()` | mm, s | mm/s | Regression slope |
| `GRAVITY_Z` | — | mm/s² | Constant: −9810 |
| `DRAG_K` | — | mm⁻¹ | Constant: 0.000112 |
| `_step_euler()` | mm, mm/s | mm, mm/s | Euler integration |
| UART packet | mm, s | mm, s | Direct passthrough |

**Chain is unit-consistent.** The only risk is if `cam_scale` in the points-based transform JSON doesn't match the expected cm→mm = 10.0 factor. Verify `cam_scale` value at startup.

---

## 6. Timing Budget (per frame)

| Phase | Typical | Notes |
|-------|---------|-------|
| `grab()` × 2 | ~0 ms | Returns immediately if trigger-synced |
| `retrieve()` × 2 | ~2 ms | MJPG decode |
| `BallDetector.detect()` × 2 | ~5-8 ms | MOG2 + morphology + contour |
| Rectify + triangulate | ~1 ms | Small matrix ops |
| `cam_to_robot()` | ~0 ms | 3×3 matmul |
| `add_position()` + velocity | ~0.5 ms | Regression (15 points) |
| `predict_intercept()` | ~1.5 ms | 300 Euler steps |
| UART send | ~0.5 ms | 28 bytes @ 115200 baud |
| Visualization + resize | ~3-5 ms | `cv2.resize` × 2 + text overlay |
| **Total** | **~15-20 ms** | Leaves headroom for 100 fps capture |

The bottleneck is detection (MOG2), not prediction or UART. Frame capture is trigger-paced at 100 fps (10 ms period), so the pipeline has ~5-8 ms of slack per frame.
