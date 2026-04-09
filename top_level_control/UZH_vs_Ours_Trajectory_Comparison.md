# Trajectory Prediction: UZH Event-Based vs Our Stereo Vision Pipeline

**Comprehensive side-by-side comparison of trajectory prediction logic**

Repo compared: [uzh-rpg/event_based_ping_pong_ball_trajectory_prediction](https://github.com/uzh-rpg/event_based_ping_pong_ball_trajectory_prediction)
Paper: "Egocentric Event-Based Vision for Ping Pong Ball Trajectory Prediction" (CVPRW 2025)

> Camera/sensor differences (event camera vs stereo ArduCam) are out of scope.
> This document focuses purely on the trajectory estimation and prediction logic.

---

## 1. Architecture Overview

| Aspect | UZH | Ours |
|--------|-----|------|
| **Primary goal** | Offline trajectory prediction + evaluation | Real-time robot interception |
| **Pipeline** | Detect -> Regression -> EKF bootstrap -> Multi-model prediction | Detect -> Triangulate -> Transform -> KF online -> Forward scan |
| **Prediction models** | 5 models: ODE physics, EKF+ODE, DCGM (deep generative), LSTM, ProMP | 1 model: Euler forward scan with gravity + drag |
| **Output** | Trajectory distribution (mean + covariance) | Single intercept point (x, y, z, time) |
| **Uncertainty** | Full covariance matrix from empirical sampling (30 samples) | None (deterministic prediction) |
| **Frame** | World frame (meters, camera 0 = origin) | Robot frame (mm, delta robot base = origin) |
| **Coordinate system** | Z = vertical (up = positive), gravity = [0, 0, -9.8] | Z = vertical (down = negative), gravity_z = -9810 mm/s^2 |

### Data Flow Comparison

**UZH:**
```
2D detections + radii
  -> Parabola smoothing (image space)
  -> Depth from radius: Z = fx * r_ball / r_pixel
  -> Monotonic depth constraint (scipy.optimize)
  -> World-frame 3D positions
  -> Velocity via finite differences
  -> EKF bootstrap (smooth initial state)
  -> Multi-model prediction (ODE / DCGM / LSTM / ProMP)
  -> Trajectory distribution (mean + covariance per timestep)
```

**Ours:**
```
Stereo 2D detections (left + right cameras)
  -> Stereo rectification + DLT triangulation
  -> Epipolar + reprojection validation
  -> Camera-frame 3D positions (cm)
  -> Kabsch SVD rigid transform -> Robot-frame (mm)
  -> Online Kalman filter (gravity as control input)
  -> Euler forward scan with gravity + drag
  -> First workspace entry = intercept point
```

**Key difference**: UZH reconstructs 3D from monocular depth (radius-based), then runs batch prediction. We use stereo triangulation for direct 3D, then run online KF + forward scan.

---

## 2. State Estimation (Kalman Filtering)

### 2.1 Filter Type & State Vector

| Property | UZH EKF | Our KF |
|----------|---------|--------|
| **Type** | Extended Kalman Filter (scipy ODE solver) | Linear Kalman Filter (FilterPy) |
| **State vector** | CV: [x,y,z,vx,vy,vz] (6D) or CA: [x,y,z,vx,vy,vz,ax,ay,az] (9D) | [x,y,z,vx,vy,vz] (6D) |
| **Measurement** | Position + Velocity (6D) | Position only (3D) |
| **Role** | Bootstrap-only (smooth initial state, then hand off) | Online (runs continuously, provides state for prediction) |
| **Dynamics in prediction step** | Full ODE: gravity + drag + Magnus (scipy.integrate.solve_ivp) | Linear F matrix + gravity as control input u |
| **Nonlinearity** | Yes (drag/Magnus make dynamics nonlinear -> EKF) | No (linear KF, drag handled only in forward scan) |

### 2.2 Process Model

**UZH (EKF predict):**
```python
# Solves full nonlinear ODE:
def ball_dynamics_cv(t, state, omega, dynamics):
    Fd = -0.5 * rho * Cd * A * |v| * v / m     # Drag
    Fm = Cl * cross(omega, v) / m               # Magnus (spin)
    a = Fd + Fm + g                             # Total
    return [vx, vy, vz, ax, ay, az]

# Prediction uses scipy ODE solver:
sol = solve_ivp(ball_dynamics_cv, [0, dt], state, t_eval=[dt])

# Jacobian is linearized (constant F, not true EKF Jacobian):
F_cv = [[I_3, dt*I_3],
        [0,   I_3   ]]
```

**Ours (KF predict):**
```python
# Linear state transition:
F = [[1, 0, 0, dt, 0,  0],
     [0, 1, 0, 0,  dt, 0],
     [0, 0, 1, 0,  0,  dt],
     [0, 0, 0, 1,  0,  0],
     [0, 0, 0, 0,  1,  0],
     [0, 0, 0, 0,  0,  1]]

# Gravity as control input:
B = [0, 0, 0.5*dt^2, 0, 0, dt]^T
u = gravity_z  # -9810 mm/s^2

# x_pred = F @ x + B @ u
```

**Analysis**: UZH models drag + Magnus inside the EKF prediction step, making it nonlinear (hence "Extended" KF). However, they use a linearized Jacobian that doesn't account for these forces in covariance propagation -- the Jacobian is the same constant-velocity form as ours. This is a simplification. Our approach is honest about being linear: gravity handled via control input, drag deferred to the forward scan.

### 2.3 Measurement Model

| Property | UZH | Ours |
|----------|-----|------|
| **H matrix** | 6x6 identity (CV) or 6x9 (CA) -- observes pos + vel | 3x6 -- observes position only |
| **Measures velocity?** | Yes (from finite differences of 3D positions) | No (velocity inferred by KF from position stream) |
| **R matrix** | sigma_R * I (scalar, from config) | diag([25^2, 25^2, 45^2]) mm^2 |

**Analysis**: UZH feeds both position AND velocity as measurements, which gives the filter more information per update but relies on the quality of finite-difference velocity estimates (noisy). We feed position only and let the KF infer velocity from the measurement stream, which is more principled -- velocity emerges from the filter dynamics rather than being measured with noise amplification from differentiation.

### 2.4 Noise Tuning

| Parameter | UZH | Ours |
|-----------|-----|------|
| **Process noise Q** | sigma_Q * I (scalar, uniform all states) | Block-diagonal, position-velocity coupled per axis: `[[0.25*dt^4, 0.5*dt^3], [0.5*dt^3, dt^2]] * accel_std^2` |
| **Measurement noise R** | sigma_R * I (scalar) | Axis-specific: XY=25mm, Z=45mm |
| **Initial P** | sigma_P * I (scalar) | diag([50^2, 50^2, 50^2, 2000^2, 2000^2, 2000^2]) |
| **Fading memory** | None | 1.03x covariance inflation per step |

**Analysis**: Our Q construction is more physically motivated -- it models uncertain acceleration over dt using the standard piecewise-constant white noise model. UZH uses a flat scalar diagonal which doesn't capture position-velocity correlation. Our separate XY vs Z measurement noise reflects stereo triangulation reality (Z/depth is noisier). The fading memory factor (1.03) is a significant advantage for handling state transitions (e.g., approaching bounce).

### 2.5 Bounce Handling in KF

| Aspect | UZH | Ours |
|--------|-----|------|
| **Detection** | Inside EKF predict: if z <= table_height and vz < 0 | Observed in raw data: Z reversal for 3+ frames after 150mm+ fall |
| **Response** | Reverse vz, scale velocity by bounce_factor, set has_bounced flag | Reset KF, keep last 2 points, require 6 more updates |
| **Limitation** | Only 1 bounce (flag-based, no reset) | Up to MAX_BOUNCES (counter-based), full KF restart each time |

**Analysis**: UZH handles bounce analytically inside the filter prediction -- elegant but only handles one bounce and doesn't restart the covariance. Our approach is more robust: we observe the bounce in data, then restart the KF for the new arc. This is safer because the ball state changes discontinuously at bounce, and resetting P avoids the filter trusting pre-bounce velocity estimates.

---

## 3. Physics Models

### 3.1 Forces Modeled

| Force | UZH | Ours |
|-------|-----|------|
| **Gravity** | [0, 0, -9.8] m/s^2 (world Z up) | -9810 mm/s^2 (robot Z down) |
| **Air drag** | F_d = -0.5 * rho * Cd * A * \|v\| * v | a_drag = -DRAG_K * \|v\| * v |
| **Magnus (spin)** | F_m = Cl * (omega x v) * A * rho | Not modeled |
| **Bounce** | vz reversal + bounce_factor (per-component tuple) | vz *= -RESTITUTION (0.85), vxy *= FRICTION (0.95) |

### 3.2 Drag Implementation

**UZH:**
```python
# Full aerodynamic form (configurable per-ball):
mass = 0.0027 kg
radius = 0.02 m
A = pi * radius^2
Fd = -0.5 * rho * Cd * A * |v|^2 * (v/|v|) / mass
# rho ~ 1.225 kg/m^3, Cd ~ 0.47
```

**Ours:**
```python
# Precomputed drag constant:
DRAG_K = 0.000124  # mm^-1
# Cd=0.445, rho=1.2e-9 g/mm^3, A=pi*20^2 mm^2, m=0.0027 kg
# k = 0.5 * Cd * rho * A / m
a_drag = -DRAG_K * |v| * v  # mm/s^2
```

**Analysis**: Mathematically identical. We precompute k = 0.5*Cd*rho*A/m into a single constant (DRAG_K), while UZH computes it from individual parameters at runtime. Same physics, different packaging.

### 3.3 Magnus Force (Spin)

UZH models Magnus force: `Fm = Cl * (omega x v) * A * rho`, where omega is the ball's angular velocity. This can significantly deflect the trajectory for spin serves. **We do not model spin at all.** This is a gap, but Magnus coefficient and omega are very hard to measure without specialized sensors, and UZH notes that omega is typically set to [0,0,0] in their configs unless specifically measured.

### 3.4 Integration Method

| Property | UZH | Ours |
|----------|-----|------|
| **ODE solver** | `scipy.integrate.solve_ivp` (adaptive Runge-Kutta) in EKF; explicit Euler in trajectory prediction | Semi-implicit Euler |
| **Timestep** | 1/180 s (~5.5ms) for diff_eq model | 5ms (SCAN_DT) for forward scan, 1ms for physics_model |
| **Accuracy** | Higher (adaptive solver adjusts step size) | Lower (fixed-step Euler) but adequate for 1.5s horizon |

**Analysis**: UZH's use of scipy's adaptive ODE solver in the EKF is more mathematically rigorous, but also slower. For their offline batch processing this is fine. Our fixed-step semi-implicit Euler is fast enough for real-time (300 steps at 5ms = 1.5s lookahead in <1ms wall time) and the error over a 1.5s prediction is negligible for a ping pong ball trajectory.

### 3.5 Bounce Physics

**UZH (in ODE trajectory prediction):**
```python
if z <= table_height and vz < 0:
    vz *= -1
    v *= bounce_factor  # tuple, can be different per axis
    has_bounced = True
    # Position reflection:
    z = 2 * table_height - z  # (in diff_eq.py)
```

**Ours:**
```python
if z < Z_TABLE_SURFACE and z_prev >= Z_TABLE_SURFACE:
    # Interpolate exact crossing fraction
    frac = (Z_TABLE_SURFACE - z_prev) / (z - z_prev)
    xb, yb = interpolate(frac)
    zb = Z_TABLE_SURFACE
    vz = -vz * RESTITUTION_COEFF  # 0.85
    vx *= FRICTION_COEFF           # 0.95
    vy *= FRICTION_COEFF
    # Complete remaining timestep from bounce point
    x,y,z = step_euler(xb, yb, zb, vx, vy, vz, dt*(1-frac))
```

**Analysis**: Our bounce implementation is more accurate. We:
1. Detect the exact crossing point via linear interpolation
2. Separate normal (restitution) and tangential (friction) coefficients
3. Simulate the remaining timestep after bounce

UZH simply reverses vz and scales the full velocity by a single factor (or tuple), without interpolating the crossing point or completing the remaining timestep. Their diff_eq model does position reflection (`z = 2*z_table - z`), which partially compensates.

---

## 4. Prediction Approaches (UZH has 5, we have 1)

### 4.1 Our Approach: Euler Forward Scan

```
Given: (x, y, z, vx, vy, vz) from KF state
For t = 0 to 1.5s in 5ms steps:
    Step Euler with gravity + drag
    Apply bounce if crossing table surface
    If point is inside workspace ellipse -> return intercept
If no hit: clamp nearest point to workspace boundary
```

This is a single deterministic trajectory. No uncertainty, no distribution.

### 4.2 UZH Approach 1: ODE Physics Model

Same physics as ours (gravity + drag + Magnus), but:
- Uses explicit Euler (slightly different: `position += v_prev * dt` instead of `v_new * dt`)
- Simulates until bounce, then extends for a configurable time
- Returns full trajectory array (not scanning for workspace entry)
- **No workspace concept** -- they predict the full path, evaluation happens separately

### 4.3 UZH Approach 2: EKF Bootstrap + ODE

1. Run EKF forward through all measurements to get smoothed (p_0, v_0)
2. Feed smoothed initial state to ODE model
3. **Key insight**: EKF provides a better initial state estimate than raw measurements, especially for velocity

We do something similar: our KF runs online and provides the state for forward scan. The difference is UZH does it in batch (all measurements at once) while we do it incrementally.

### 4.4 UZH Approach 3: DCGM (Deep Conditional Generative Model)

A **learned** trajectory predictor:
- Encoder compresses observed trajectory to latent distribution
- Decoder generates future trajectory samples from latent + context
- 30 samples -> empirical mean + covariance
- Trained on recorded trajectories (TensorFlow/Keras)

**We have nothing equivalent.** This is the biggest capability gap.

### 4.5 UZH Approach 4: LSTM

- Autoregressive sequence-to-sequence model
- Feeds predictions back as input for next step
- Also produces distributional output via sampling

**We have nothing equivalent.**

### 4.6 UZH Approach 5: ProMP (Probabilistic Movement Primitives)

- Basis function expansion (polynomial or RBF)
- Bayesian linear regression over basis weights
- EM-based training
- Analytical posterior for fast prediction

**We have nothing equivalent.**

### 4.7 UZH Approach 6: BallTrajectory (Physics with Bayesian Init)

(`traj_pred/ball_model/diff_eq.py`)

This is the most interesting physics model:
```python
# Fits initial state distribution via Bayesian linear regression:
BallInitState.fit(time, observations)  # polynomial features

# Samples N initial states from posterior:
pos, vel = init_state.sample(30)  # 30 samples

# Simulates each sample forward:
for n in range(30):
    trajectory[n] = get_traj_sample(pos[n], vel[n], ...)

# Computes empirical distribution:
means, covs = empirical_traj_dist(trajectories)
```

**Key insight**: Instead of a single (p_0, v_0), they sample from the posterior distribution over initial states and propagate each sample through physics. This naturally captures uncertainty in the initial state and propagates it through the trajectory.

**Comparison to ours**: We use a single KF state (point estimate) and propagate deterministically. This gives us no uncertainty information. The Bayesian init approach would be a natural extension of our pipeline -- we already have the KF covariance (P matrix) which encodes uncertainty over the state, but we don't use it for prediction.

---

## 5. Initial State Estimation

### 5.1 Velocity Estimation

**UZH:**
```python
# Finite differences of smoothed 3D positions:
dt = diff(timestamps)
v_measurements = diff(p_measurements, axis=0) / dt
v_0 = mean(v_measurements)  # average over all velocity estimates
```
Plus: Monotonic depth constraint forces physically realistic depth velocity (ball approaching at -1.5 to -6 m/s).

**Ours (KF path):**
```python
# Kalman filter infers velocity from position stream:
# No explicit velocity computation -- emerges from filter dynamics
velocity = state_estimator.get_velocity()  # KF state[3:6]
```

**Ours (legacy fallback):**
```python
# Regression with gravity correction:
z_grav = z - 0.5 * g * t^2       # remove known gravity effect
v = lstsq(A, z_grav)             # linear regression
# Plus iterative drag correction
```

**Analysis**: Our KF approach is superior to UZH's finite differences for velocity estimation. Finite differences amplify measurement noise (differentiation is an ill-conditioned operation). The KF jointly estimates position and velocity with proper noise modeling. UZH partially compensates by feeding velocity as a measurement into their EKF, but this still starts from noisy derivatives.

### 5.2 3D Position Recovery

**UZH (monocular depth from radius):**
```python
Z = fx * ball_radius / measured_radius  # depth from apparent size
# Then: monotonic constraint to force physically realistic depth profile
# Then: unproject (u,v,Z) to world 3D via calibration chain
```

**Ours (stereo triangulation):**
```python
# DLT triangulation from rectified stereo correspondences:
A = [...4 rows from P_rect0, P_rect1...]
U, S, Vh = SVD(A)
X_3D = Vh[-1][:3] / Vh[-1][3]  # homogeneous -> 3D
X_cam0 = R_rect0.T @ X_3D      # rectified -> original camera frame
```

**Analysis**: Stereo triangulation gives direct 3D without assumptions about ball size. UZH's monocular approach requires knowing the exact ball radius and has depth accuracy proportional to 1/r_pixel (noisy for small apparent sizes). However, their monotonic depth constraint is clever -- it regularizes noisy depth estimates by enforcing that the ball approaches at a physically plausible speed.

---

## 6. Outlier Rejection & Robustness

| Check | UZH | Ours |
|-------|-----|------|
| **Spatial bounds** | Implicit (gaze crop limits search area) | Explicit: y in [-500, 3000], \|x\| < 2000, z in [-2000, 0] |
| **Speed limit** | Via depth velocity constraints (-1.5 to -6 m/s) | MAX_SPEED = 15000 mm/s, MAX_JUMP = 400 mm |
| **Time gap** | Implicit (IMU batch boundaries) | GAP_RESET = 0.12s -> full buffer reset |
| **Stale data** | N/A (offline) | STALE_TIMEOUT = 0.15s |
| **Epipolar check** | N/A (monocular) | MAX_EPIPOLAR_ERR = 15px |
| **Reprojection check** | N/A | MAX_REPROJ_ERR = 8px |
| **Monotonic depth** | Yes (scipy optimization constraint) | No |
| **Circularity filter** | Yes (convex hull metric) | Yes (4*pi*A/p^2 >= 0.35) |
| **Min observation time** | Implicit (batch size) | MIN_TIME_SPAN = 0.06s |
| **Min data points** | >= 2 measurements for regression | MIN_POINTS = 8 (legacy) or min_updates = 4 (KF) |

**Analysis**: We have significantly more runtime rejection logic, which makes sense for real-time operation where bad data must be caught immediately. UZH operates in batch mode where they can inspect the full measurement sequence. Their monotonic depth constraint is an elegant form of outlier rejection that we lack.

---

## 7. Key Differences Summary

### Things UZH Does That We Don't

1. **Multiple prediction models** (DCGM, LSTM, ProMP, BallTrajectory) -- we only have physics-based forward scan
2. **Uncertainty quantification** -- they produce trajectory distributions (mean + covariance), we produce a single point
3. **Magnus force / spin modeling** -- they model ball spin effects, we assume zero spin
4. **Bayesian initial state estimation** -- they sample from posterior over (p_0, v_0), we use point estimate
5. **Monotonic depth constraint** -- clever regularization of monocular depth estimates
6. **Constant Acceleration model** (9D state) -- we only have 6D
7. **ODE solver (scipy.integrate.solve_ivp)** -- adaptive step size, higher accuracy
8. **Motion compensation** -- gyroscope-based event warping (sensor-specific, not applicable to us)

### Things We Do That UZH Doesn't

1. **Online Kalman filter** -- continuous state updates, not batch bootstrap
2. **Gravity as control input** -- cleaner separation of known physics from unknown perturbations
3. **Fading memory** (1.03x covariance inflation) -- helps transition between trajectory arcs
4. **Observed bounce detection** -- data-driven Z reversal detection with hysteresis (3 frames, 150mm fall)
5. **KF restart on bounce** -- clean slate for post-bounce arc
6. **Precise bounce interpolation** -- fraction-based crossing point + remaining timestep completion
7. **Workspace-aware prediction** -- scans for workspace entry, clamps fallback
8. **Direction filter** -- only predicts when ball is approaching
9. **Separate XY vs Z measurement noise** -- reflects stereo depth being noisier than lateral position
10. **Real-time constraints** -- stale timeout, speed limits, jump rejection

### Things Both Do Similarly

1. **Quadratic drag**: F_d = -k * |v| * v (identical formula, same Cd ~ 0.45)
2. **Bounce with restitution** (0.7-0.85 range)
3. **Euler integration** for trajectory prediction
4. **6D state vector** [x,y,z,vx,vy,vz] as primary model
5. **Gravity modeling** (9.8 m/s^2)
6. **Contour circularity** for ball detection filtering

---

## 8. Potential Improvements Inspired by UZH

### High Impact, Moderate Effort

1. **Uncertainty propagation from KF covariance**: We already have P (the KF covariance). We could sample N initial states from N(x_kf, P_kf), propagate each through our Euler scan, and report the spread. This would tell us *how confident* the intercept prediction is -- useful for deciding whether to commit to a swing or stay at home.

2. **Bayesian initial velocity estimation**: Instead of using the KF point estimate directly, sample from the velocity posterior and run multiple forward scans. If the trajectories diverge widely, the prediction is unreliable. This is essentially what UZH's BallTrajectory model does.

### Medium Impact, Low Effort

3. **Constant Acceleration model (9D KF)**: Add acceleration to the state vector. This could help during the initial observation window when the ball is decelerating (drag) or when the KF hasn't converged yet. However, 9D requires more updates to converge -- tradeoff with our MIN_POINTS.

4. **Velocity measurement in KF**: Feed finite-difference velocity as an additional measurement (like UZH does). This would help the KF converge faster. Risk: amplified noise from differentiation. Mitigation: set R_velocity much larger than R_position.

### Lower Priority / Research

5. **Learned trajectory predictor (DCGM/LSTM)**: Would require collecting training data from our setup. Significant effort but could capture effects (spin, table friction) that physics models miss. The DCGM approach is particularly interesting because it's generative -- it can model the full distribution of possible trajectories.

6. **Magnus force modeling**: Would require estimating ball spin, which is extremely hard from vision alone. Low priority unless we observe systematic lateral deflection in our trajectories.

---

## 9. Constants Comparison

| Constant | UZH | Ours | Match? |
|----------|-----|------|--------|
| Gravity | 9.8 m/s^2 | 9810 mm/s^2 | Yes (unit difference) |
| Ball mass | 2.7 g | 2.7 g | Yes |
| Ball radius | 20 mm | 20 mm | Yes |
| Cd (drag) | 0.47 | 0.445 | Close (~5% difference) |
| Air density | 1.225 kg/m^3 | 1.225e-9 g/mm^3 | Yes (unit difference) |
| Bounce restitution | 0.6-0.8 (configurable tuple) | 0.85 | Ours is slightly higher |
| Table height | 0.76 m (standard) | -1150 mm (robot frame) | N/A (different reference) |
| Integration dt | 5.5 ms (1/180 Hz) | 5 ms | Close |

---

## 10. Conclusion

Our pipeline is optimized for **real-time single-point interception** with strong engineering for robustness (outlier rejection, bounce detection, workspace clamping). UZH's pipeline is designed for **offline trajectory evaluation** with rich uncertainty quantification and multiple prediction strategies.

The most actionable insight from UZH is **using the KF covariance for uncertainty propagation**. We already compute P but throw it away at prediction time. Sampling from N(x, P) and running multiple forward scans would give us confidence bounds on the intercept point at near-zero implementation cost -- no new models needed, just a loop around our existing `_step_euler` logic.
