# Pipeline Review & Optimization Roadmap

## Current Performance (March 23, 2026)
- ~25% of shots: spot on XYZ
- ~60% of shots: semi-accurate, some axes off
- ~15% of shots: wrong coordinates
- Mean 3D prediction error: ~434mm
- Best achievable: <100mm (proven possible)

---

## STEP-BY-STEP OPTIMIZATION ROADMAP

Each step has acceptance criteria. Do NOT move to the next step until the current step passes.

---

### STEP 1: STEREO CALIBRATION (Foundation — Everything Depends on This)

**Current state:**
- Reproj error: 69% of frames <1px (GOOD)
- BUT: reproj error is 75% worse on right side of frame vs left
- 42% of frames fail stereo (POOR acceptance rate)
- 62% of epipolar failures are >50px (severe mismatches)
- Reproj error increases with distance from camera (r=0.36, highly significant)

**What to do:**
1. Recalibrate with 40+ stereo image pairs covering the ENTIRE frame
   - Specifically: get 10+ images where the checkerboard is in the RIGHT half of frame
   - Get 10+ images at CLOSE range (60-100cm) and 10+ at FAR range (200-260cm)
   - Get images at frame edges and corners, not just center
2. Consider adding k3 distortion term if not already used
3. After calibration, verify:

**Acceptance criteria before moving on:**
- [ ] Stereo reproj error < 0.5px RMS (currently ~1.06px)
- [ ] Reproj error consistent across frame (right side within 20% of left side)
- [ ] Epipolar error < 2px for 95% of frame positions
- [ ] Run `test_triangulation_verify.py` Mode 4 (diameter): ball diameter should read 3.8-4.2cm consistently at near, mid, and far range

---

### STEP 2: CAMERA-TO-ROBOT TRANSFORM (Critical for XY Accuracy)

**Current state:**
- Using combined_transform.json (XY from points-based, Z from measurement-based)
- Static XY accuracy: ~43mm RMS (from 15-point optimization)
- Z accuracy: adjusted with offset, approximate
- Reproj filter in integration script: 100px (WAY too loose)

**What to do:**
1. After Step 1 calibration is verified, re-run `test_find_points_based_transform.py`
2. Collect 15-20 point correspondences spanning the FULL workspace:
   - 5 points at Y≈700mm (near robot)
   - 5 points at Y≈1400mm (mid table)
   - 5 points at Y≈2100mm (far end)
   - At each Y: measure center (X≈0) and both sides (X≈±500mm)
   - Include points at DIFFERENT Z heights (table surface AND ~200mm above table)
3. Run the optimization (Kabsch SVD fit)
4. Verify with 5 NEW points not used in the fit

**Acceptance criteria before moving on:**
- [ ] XY RMS error < 30mm on verification points
- [ ] Z RMS error < 30mm on verification points (requires points at multiple Z heights)
- [ ] No systematic bias > 15mm on any axis
- [ ] Scale factor is consistent (should be 9.5-10.5)

---

### STEP 3: REPROJ FILTER TIGHTENING (Prevents Bad Data from Entering KF)

**Current state:**
- Integration script reproj filter: 100px (effectively disabled)
- Stereo triangulator internal filter: 8px
- Points with 2-8px reproj are borderline quality

**What to do:**
1. Tighten reproj filter in `test_integration_simple.py` from 100px to 5px
2. Run 10 throws and check:
   - What % of frames get rejected by the new filter?
   - Does the buffer still fill to 10 points within the tracking window?
   - If acceptance drops below 40%, relax to 8px

**Acceptance criteria before moving on:**
- [ ] Reproj filter set to 5-8px
- [ ] Buffer still reaches MIN_SEND_BUFFER (10 pts) for >80% of throws
- [ ] Frame acceptance rate > 40%

**File:** `test_integration_simple.py` line 714: change `100` to `5`

---

### STEP 4: KF TUNING VERIFICATION (Velocity Accuracy)

**Current state:**
- accel_std=1500, meas_std_xy=40, meas_std_z=80, fading_factor=1.03
- VY matches raw data within ~9% (GOOD)
- VZ diverges significantly from raw data (BAD — the #1 prediction error source)
- KF VZ is systematically too positive (predicts ball rises faster than reality)

**What to do:**
1. With clean data from Steps 1-3, run 10 throws and analyze `throw_data_log.json`
2. For each throw with sends, compare:
   - KF VZ vs raw buffer VZ (should be within 20%)
   - KF VY vs raw buffer VY (should be within 10%)
   - KF VX vs raw buffer VX (should be within 15%)
3. If VZ still diverges >30% from raw:
   - Increase meas_std_z to 120 (trust gravity model more)
   - OR decrease accel_std to 1000 (smoother, less reactive)
4. If VY diverges >15%:
   - Increase meas_std_xy to 50

**Acceptance criteria before moving on:**
- [ ] KF VY within 15% of raw buffer VY for >80% of throws
- [ ] KF VZ within 30% of raw buffer VZ for >80% of throws
- [ ] No velocity axis flipping sign vs raw data

**Files:** `trajectory/ball_state_estimation.py` (meas_std_z, accel_std)

---

### STEP 5: PREDICTION VALIDATION (Forward Sim Accuracy)

**Current state:**
- predict_intercept() uses Euler integration with gravity + drag
- DRAG_K = 0.000124 (Cd=0.445)
- Forward sim starts from KF smoothed state (may lag real ball)
- Workspace: ellipse A=790, B=540, Z=-950 to -760
- Table surface: Z=-1150, restitution=0.85

**What to do:**
1. With good KF velocity from Step 4, run 10 throws
2. For each send, compare:
   - Predicted target vs where ball actually went (from last accepted frames)
   - Break error down by axis: dX, dY, dZ
3. If Z predictions are consistently off:
   - Test DRAG_K=0 vs DRAG_K=0.000124 — compare which gives lower Z error
   - Check if Z_TABLE_SURFACE=-1150 matches reality (measure actual table Z)
   - Check if RESTITUTION_COEFF=0.85 is correct (does ball bounce as high as predicted?)
4. If Y predictions are off by a consistent amount:
   - Check if timing is correct (t_intercept vs actual arrival time)
   - Verify latency calculation is accurate
5. Consider using last RAW position + KF velocity as seed (instead of KF smoothed position)

**Acceptance criteria before moving on:**
- [ ] Mean 3D prediction error < 100mm
- [ ] No axis with mean error > 60mm
- [ ] No axis with std error > 80mm
- [ ] <20% of predictions clamped (direct workspace hits)

**Files:** `trajectory/robot_predictor.py` (predict_intercept, _get_prediction_state), `trajectory/workspace.py` (DRAG_K, Z_TABLE_SURFACE, RESTITUTION_COEFF)

---

### STEP 6: SEND TIMING & GATING (When to Send)

**Current state:**
- MIN_SEND_BUFFER = 10 points
- VZ > 0 gate blocks sends while ball rising post-bounce
- MIN_TIME_HIT = 0.10s (ignores first 100ms of forward sim)
- MAX_PREDICT_Y = 1400mm
- Update threshold: 80mm change required

**What to do:**
1. Verify MIN_SEND_BUFFER=10 is sufficient (should have 250ms+ of data)
2. Check if VZ gate is still needed after KF tuning (Step 4)
3. Consider reducing MIN_TIME_HIT from 0.10s to 0.05s if ball is often close to workspace at send time
4. Monitor update sends — are updates improving or worsening the prediction?

**Acceptance criteria:**
- [ ] First send happens with >250ms of tracking data
- [ ] Time from send to ball arrival > 200ms (robot has time to move)
- [ ] Updates improve prediction (error decreases after update)

---

## HIGH PRIORITY ISSUES FOUND

### ISSUE 1: REPROJ FILTER IS 100px — EFFECTIVELY DISABLED (CRITICAL)

The reproj filter in `test_integration_simple.py` line 714 is set to 100px. The stereo triangulator already rejects at 8px internally. So points with 8-100px reproj pass through — these are terrible measurements.

**Impact:** Noisy points corrupt KF velocity, especially VZ. This is likely the single biggest contributor to the 75% failure rate.

**Fix:** Change `100` to `5` in line 714. This alone could significantly improve results.

### ISSUE 2: 42% STEREO FAILURE RATE (HIGH)

Nearly half of all frames fail stereo matching. This means the KF often has gaps in data, causing velocity drift between measurements. The 62% of epipolar failures being >50px suggests the cameras see different things — possibly motion blur at high speed, or calibration degradation at frame edges.

**Impact:** Fewer accepted frames → less data for KF → noisier velocity → worse prediction.

**Fix:** Better calibration coverage (Step 1), especially at frame edges and close range.

### ISSUE 3: VZ DIVERGENCE FROM RAW DATA (HIGH)

The KF VZ estimate systematically differs from what the raw position data shows. This is the #1 axis error in predictions. The KF overestimates how fast the ball rises (VZ too positive), causing predictions to put the intercept point too deep in Z.

**Impact:** Z prediction error of 93mm mean, up to 400+mm on bad throws.

**Fix:** After clean data from Steps 1-3, increase meas_std_z further (80→120) so KF trusts gravity model and doesn't chase Z noise. Or decrease accel_std (1500→1000).

### ISSUE 4: POSITION-DEPENDENT CALIBRATION ERROR (MODERATE)

Reproj error is 75% worse on the right side of the frame vs left. Reproj also increases with distance. This means triangulation quality varies across the working volume — points far away or on the right side of frame are less accurate.

**Impact:** Throws from certain directions/distances are systematically less accurate.

**Fix:** Recalibrate with better edge/corner coverage (Step 1).

### ISSUE 5: NO Z-HEIGHT VARIATION IN TRANSFORM FIT (MODERATE)

The points-based transform was fitted with all points at Z=-1150.74mm (table surface only). This means the Z axis of the transform is poorly constrained. The combined_transform.json uses a different Z row to compensate, but this is a workaround.

**Impact:** Z accuracy degrades for points not at table surface (i.e., the entire ball flight above the table).

**Fix:** Include points at multiple Z heights when refitting the transform (Step 2).

---

## CURRENT PARAMETER REFERENCE

| Component | Parameter | Current Value | Notes |
|-----------|-----------|---------------|-------|
| **KF** | accel_std | 1500 | Process noise |
| | meas_std_xy | 40 | XY measurement noise |
| | meas_std_z | 80 | Z measurement noise |
| | fading_factor | 1.03 | Memory decay |
| | min_updates | 8 | Readiness threshold |
| **Predictor** | BUFFER_SIZE | 10 | Position buffer |
| | MIN_SEND_BUFFER | 10 | Send gate |
| | MAX_PREDICT_Y | 1400 | Proximity filter |
| | SCAN_DURATION | 1.5s | Forward sim window |
| | MIN_TIME_HIT | 0.10s | Min reaction time |
| **Workspace** | Ellipse | 790×540mm | XY bounds |
| | Z range | -950 to -760mm | Vertical bounds |
| | DRAG_K | 0.000124 | Air drag |
| | RESTITUTION | 0.85 | Bounce damping |
| | TABLE_Z | -1150mm | Table surface |
| **Stereo** | MAX_REPROJ_ERR | 8px | Internal filter |
| | MAX_EPIPOLAR_ERR | 15px | Rectification check |
| **Integration** | reproj filter | 100px (!!) | WAY too loose |
| | tx_interval | 30ms | Send rate limit |
| | update threshold | 80mm | Min change for update |
| **Transform** | File | combined_transform.json | XY from points, Z from measurement |
| | Scale | 9.772049 | cm→mm |

---

## EXPECTED OUTCOMES

| After Step | Expected Error | What Improves |
|------------|---------------|---------------|
| Step 1 | — | Clean stereo data, <1px reproj, >70% acceptance |
| Step 2 | — | Accurate XYZ transform, <30mm static error |
| Step 3 | ~300mm → ~200mm | Bad measurements stop entering KF |
| Step 4 | ~200mm → ~120mm | VZ accuracy improves, velocity aligns with raw data |
| Step 5 | ~120mm → ~80mm | Forward sim matches reality, fewer clamped predictions |
| Step 6 | ~80mm → ~60mm | Optimal send timing, updates improve accuracy |
