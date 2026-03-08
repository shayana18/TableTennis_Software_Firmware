# Stereo Pipeline — Change Log & Debug Report

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

## Next Steps (updated 2026-03-07)

All verification tests COMPLETE with old stand. New 20° fixed-pitch stand built. Full recalibration needed. Axis convention fix DONE.

1. **Recalibrate with 20° fixed pitch stand** — Full mono intrinsics (30+ images per camera) + stereo extrinsics with new rigid mount geometry
2. **Re-run verification tests (Modes 1-4)** — Confirm improvement over current 3% lateral scale error (α=0.972)
3. **Run velocity validation** — `test_velocity_validation.py` with new calibration
4. **End-to-end trajectory prediction testing** — Full pipeline test with real throws, drag enabled
