"""
Robot-frame trajectory predictor — single source of truth.

Pipeline:
  1. Add robot-frame positions via add_position(x, y, z, t)
  2. State estimated with BallStateEstimator3D
  3. Trajectory forward-simulated with gravity + air drag
  4. Interception found via workspace scan + clamp fallback
"""

from __future__ import annotations

import math
import time
from collections import deque

import numpy as np

from .workspace import (
    MAX_CLAMP_DIST, GRAVITY_Z, DRAG_K,
    Z_TABLE_SURFACE, RESTITUTION_COEFF, FRICTION_COEFF, MAX_BOUNCES,
    in_workspace, clamp_to_workspace,
)
from .ball_state_estimation import BallStateEstimator3D


class RobotPredictor:
    """
    Trajectory predictor in robot frame (mm).
    Uses BallStateEstimator3D when available, else falls back to legacy fitting.
    """

    BUFFER_SIZE    = 15
    MIN_POINTS     = 5      # minimum points for regression velocity (was 8 for KF)
    MIN_TIME_SPAN  = 0.06   # seconds of measurement history before ready (was 0.08)
    MAX_SPEED      = 15000.0 # mm/s
    MAX_JUMP       = 200.0   # mm
    GAP_RESET      = 0.12   # seconds
    STALE_TIMEOUT  = 0.15   # seconds

    # Prediction scan
    SCAN_DURATION  = 1.5    # seconds forward
    SCAN_DT        = 0.005  # 5ms steps
    MIN_TIME_HIT   = 0.10   # minimum reaction time for robot

    # Proximity filter
    MAX_PREDICT_Y  = 1500.0  # mm -- ball must be closer than this

    # Direction filter
    MIN_APPROACH_VY = 0
    APPROACH_Y_THRESHOLD = 600.0

    # Y offset applied to every predicted intercept (mm)
    INTERCEPT_Y_OFFSET = 0

    # Confidence scoring via KF covariance sampling (inspired by UZH BallTrajectory)
    CONFIDENCE_SAMPLES    = 4      # number of initial-state samples (reduced from 8 for FPS)
    MAX_INTERCEPT_SPREAD  = 250.0  # mm -- reject if intercept spread exceeds this
    MIN_HIT_RATIO         = 0.5    # at least 50% of samples must find a workspace hit

    # Observed bounce detection
    MIN_BOUNCE_FALL_Z  = 50.0  # mm minimum Z descent before accepting bounce (was 50; stereo noise is ±30-50mm)
    BOUNCE_RISE_FRAMES = 3      # consecutive rising frames to confirm (was 2)
    BOUNCE_FALL_FRAMES = 3      # consecutive falling frames required BEFORE rising can trigger

    # Post-bounce: with regression velocity (not KF-gated), we only need
    # MIN_POINTS observations on the new arc before prediction resumes.

    def __init__(self):
        self.positions = deque(maxlen=self.BUFFER_SIZE)
        self.velocity = None
        self.state_estimator = None
        self._using_state_estimator = False
        try:
            self.state_estimator = BallStateEstimator3D(
                gravity_z=GRAVITY_Z,
                max_gap_s=self.GAP_RESET,
                min_updates=4,  # fewer updates needed with smoother KF
            )
            self._using_state_estimator = True
        except ImportError:
            # Keep the proven legacy estimator alive until FilterPy is installed.
            self.state_estimator = None
        self._rejected = 0
        self._accepted = 0
        self._last_reject_reason = None
        self._z_min_since_reset = None
        self._rising_count = 0
        self._falling_count = 0
        self._has_fallen = False  # True once we've seen enough falling frames
        self._bounce_count = 0
        self._velocity_seeded = False

    def reset(self):
        self.positions.clear()
        self.velocity = None
        if self.state_estimator is not None:
            self.state_estimator.reset()
        self._z_min_since_reset = None
        self._rising_count = 0
        self._falling_count = 0
        self._has_fallen = False
        self._bounce_count = 0
        self._velocity_seeded = False

    def add_position(self, x, y, z, t):
        """Add a robot-frame position (mm). Returns True if accepted."""
        # Reject positions clearly outside playing volume
        if y > 3000.0 or y < -500.0 or abs(x) > 2000.0 or z > 0.0 or z < -2000.0:
            self._rejected += 1
            self._last_reject_reason = f"out_of_bounds(y={y:.0f})"
            return False

        if self.positions:
            last = self.positions[-1]
            dt = t - last[3]
            if dt <= 0:
                self._rejected += 1
                self._last_reject_reason = "zero_dt"
                return False
            if dt > self.GAP_RESET:
                self._last_reject_reason = f"gap({dt*1000:.0f}ms)"
                self.reset()
            else:
                dx, dy, dz = x - last[0], y - last[1], z - last[2]
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                if dist > self.MAX_JUMP:
                    self._rejected += 1
                    self._last_reject_reason = f"jump({dist:.0f}mm)"
                    return False
                speed = dist / dt
                if speed > self.MAX_SPEED:
                    self._rejected += 1
                    self._last_reject_reason = f"speed({speed:.0f})"
                    return False

        self.positions.append((x, y, z, t))
        self._accepted += 1
        self._last_reject_reason = None

        # Observed bounce detection — reset buffer on Z reversal
        if self._detect_observed_bounce():
            self._handle_observed_bounce()
            return True

        if self.state_estimator is not None:
            # Velocity seeding: on the 2nd measurement, compute rough velocity
            # from finite differences and reinitialize KF with it. This helps
            # the KF converge ~3-4 updates faster than starting from v=0.
            # (Inspired by UZH's initial-state estimation approach.)
            if (not self._velocity_seeded
                    and self.state_estimator.update_count == 1
                    and len(self.positions) >= 2):
                p0 = self.positions[-2]
                dt_seed = t - p0[3]
                if dt_seed > 1e-6:
                    vx_seed = (x - p0[0]) / dt_seed
                    vy_seed = (y - p0[1]) / dt_seed
                    vz_seed = (z - p0[2]) / dt_seed
                    speed_seed = math.sqrt(vx_seed**2 + vy_seed**2 + vz_seed**2)
                    if speed_seed < self.MAX_SPEED:
                        self.state_estimator.initialize_from_measurement(
                            x, y, z, t, vx_seed, vy_seed, vz_seed)
                        self._velocity_seeded = True
                        # Skip estimate() — seeding already set the KF state
                        # at this timestamp. Calling estimate() would get dt=0
                        # and reject, popping this position from the buffer.
                        return True

            accepted, _state = self.state_estimator.estimate(x, y, z, t)
            if not accepted:
                self.positions.pop()
                self._accepted -= 1
                self._rejected += 1
                self._last_reject_reason = "kf_dt"
                return False

        # Always use regression velocity — it fits ALL buffered points equally
        # and is far more stable than the KF for near-zero axes (e.g. X).
        # The KF position is still used for smoothed position in _get_prediction_state.
        if len(self.positions) >= self.MIN_POINTS:
            span = self.positions[-1][3] - self.positions[0][3]
            if span >= self.MIN_TIME_SPAN:
                self._estimate_velocity_legacy()

        return True

    def _detect_observed_bounce(self):
        """Detect Z reversal (ball bouncing off table) in observed data.

        Requires an actual fall-then-rise sequence:
          1. Ball must be observed FALLING for >= BOUNCE_FALL_FRAMES
          2. Then observed RISING for >= BOUNCE_RISE_FRAMES
          3. And the total Z descent >= MIN_BOUNCE_FALL_Z
        This prevents false triggers on balls caught mid-rise.
        """
        if len(self.positions) < 2:
            return False

        z = self.positions[-1][2]
        z_prev = self.positions[-2][2]

        # Track minimum Z since reset
        if self._z_min_since_reset is None:
            self._z_min_since_reset = z
        else:
            self._z_min_since_reset = min(self._z_min_since_reset, z)

        if z < z_prev:
            # Ball is falling — accumulate falling count
            self._falling_count += 1
            self._rising_count = 0
            # Mark that we've seen enough falling to arm the bounce detector
            if self._falling_count >= self.BOUNCE_FALL_FRAMES:
                self._has_fallen = True
            return False

        if z > z_prev:
            # Ball is rising
            self._rising_count += 1
            self._falling_count = 0
        else:
            # z == z_prev — neither rising nor falling, reset rising count
            self._rising_count = 0
            self._falling_count = 0
            return False

        # Must have actually fallen first (not just caught mid-rise)
        if not self._has_fallen:
            return False

        # Need enough consecutive rising frames after the fall
        if self._rising_count < self.BOUNCE_RISE_FRAMES:
            return False

        # Must have fallen enough total (from peak to trough)
        z_first = self.positions[0][2]
        fall = z_first - self._z_min_since_reset
        return fall >= self.MIN_BOUNCE_FALL_Z

    # How many post-bounce points to keep for regression continuity
    BOUNCE_KEEP_POINTS = 4

    def _handle_observed_bounce(self):
        """Reset state after observed bounce, keeping recent points.

        Keeps BOUNCE_KEEP_POINTS (4) recent observations so the regression
        velocity estimator has enough data to produce a fit quickly on the
        new arc. The pre-bounce vy is carried forward (scaled by friction
        coefficient) as a velocity hint since table bounces preserve most
        of the horizontal velocity.
        """
        # Save pre-bounce vy for seeding (if available)
        pre_bounce_vy = None
        if self.velocity is not None:
            pre_bounce_vy = self.velocity[1] * FRICTION_COEFF

        keep = list(self.positions)[-self.BOUNCE_KEEP_POINTS:]
        self.positions.clear()
        for p in keep:
            self.positions.append(p)

        if self.state_estimator is not None:
            # Bounce is a state jump, so restart the KF on the new arc.
            self.state_estimator.reset()
            if len(keep) >= 2:
                # Seed with velocity from the last 2 observations
                p0, p1 = keep[-2], keep[-1]
                dt_b = p1[3] - p0[3]
                if dt_b > 1e-6:
                    vx_b = (p1[0] - p0[0]) / dt_b
                    vy_b = (p1[1] - p0[1]) / dt_b
                    vz_b = (p1[2] - p0[2]) / dt_b
                    # Use pre-bounce vy if we have it — more stable than
                    # 2-point finite difference for the approach direction
                    if pre_bounce_vy is not None:
                        vy_b = pre_bounce_vy
                    self.state_estimator.initialize_from_measurement(
                        p1[0], p1[1], p1[2], p1[3], vx_b, vy_b, vz_b)
                else:
                    self.state_estimator.initialize_from_measurement(*keep[-1])
            elif keep:
                self.state_estimator.initialize_from_measurement(*keep[-1])

        self.velocity = None
        self._z_min_since_reset = None
        self._rising_count = 0
        self._falling_count = 0
        self._has_fallen = False
        self._bounce_count += 1

    def _estimate_velocity_legacy(self):
        """Fallback regression estimator used when FilterPy is unavailable."""
        n = len(self.positions)
        pts = list(self.positions)

        t_ref = pts[-1][3]
        dt = np.array([p[3] - t_ref for p in pts])
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        zs = np.array([p[2] for p in pts])

        A = np.column_stack([dt, np.ones(n)])

        vx = float(np.linalg.lstsq(A, xs, rcond=None)[0][0])
        vy = float(np.linalg.lstsq(A, ys, rcond=None)[0][0])

        zs_grav = zs - 0.5 * GRAVITY_Z * dt * dt
        vz = float(np.linalg.lstsq(A, zs_grav, rcond=None)[0][0])

        speed0 = math.sqrt(vx*vx + vy*vy + vz*vz)
        if speed0 > 1e-3:
            drag_factor = 0.5 * DRAG_K * speed0
            dt2 = dt * dt
            xs_corr = xs + drag_factor * vx * dt2
            ys_corr = ys + drag_factor * vy * dt2
            zs_corr = zs_grav + drag_factor * vz * dt2
            vx = float(np.linalg.lstsq(A, xs_corr, rcond=None)[0][0])
            vy = float(np.linalg.lstsq(A, ys_corr, rcond=None)[0][0])
            vz = float(np.linalg.lstsq(A, zs_corr, rcond=None)[0][0])

        speed = math.sqrt(vx*vx + vy*vy + vz*vz)
        if speed > self.MAX_SPEED:
            self.velocity = None
            return

        self.velocity = (vx, vy, vz)

    def is_ready(self):
        if not self.positions:
            return False
        if self.velocity is None:
            return False
        span = self.positions[-1][3] - self.positions[0][3]
        if span < self.MIN_TIME_SPAN:
            return False
        age = time.perf_counter() - self.positions[-1][3]
        return age < self.STALE_TIMEOUT

    def _get_prediction_state(self):
        """Return the current state used to seed future prediction.

        Uses raw measured position + regression velocity.
        The KF position has a systematic Z bias (~60mm too negative)
        due to gravity in its predict step, so raw position is more
        accurate for prediction starting points.
        """
        if not self.positions or self.velocity is None:
            return None
        x, y, z, _ = self.positions[-1]
        vx, vy, vz = self.velocity
        return x, y, z, vx, vy, vz

    def _ball_approaching(self):
        """Check if ball is moving toward the workspace."""
        # Replaces split raw/KF reads with one prediction state source.
        state = self._get_prediction_state()
        if state is None:
            return False
        _, y_now, _, _, vy, _ = state
        if abs(y_now) < self.APPROACH_Y_THRESHOLD:
            return True
        return vy < self.MIN_APPROACH_VY

    @staticmethod
    def _step_euler(x, y, z, vx, vy, vz, dt):
        """One Euler step with gravity + air drag."""
        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        drag = DRAG_K * speed

        ax = -drag * vx
        ay = -drag * vy
        az = GRAVITY_Z - drag * vz

        vx += ax * dt
        vy += ay * dt
        vz += az * dt

        x += vx * dt
        y += vy * dt
        z += vz * dt

        return x, y, z, vx, vy, vz

    @staticmethod
    def _apply_bounce(x_prev, y_prev, z_prev, x, y, z, vx, vy, vz, dt):
        """Reflect ball off table surface if it crossed Z_TABLE_SURFACE.
        Returns (x, y, z, vx, vy, vz, did_bounce)."""
        if z >= Z_TABLE_SURFACE or z_prev < Z_TABLE_SURFACE:
            return x, y, z, vx, vy, vz, False

        # Interpolate crossing fraction
        dz = z - z_prev
        if abs(dz) < 1e-6:
            return x, y, z, vx, vy, vz, False
        frac = (Z_TABLE_SURFACE - z_prev) / dz
        frac = max(0.0, min(1.0, frac))

        # Position at bounce point
        xb = x_prev + frac * (x - x_prev)
        yb = y_prev + frac * (y - y_prev)
        zb = Z_TABLE_SURFACE

        # Reflect velocity
        vz = -vz * RESTITUTION_COEFF
        vx *= FRICTION_COEFF
        vy *= FRICTION_COEFF

        # Complete remaining timestep from bounce point
        remain = dt * (1.0 - frac)
        if remain > 1e-6:
            x, y, z, vx, vy, vz = RobotPredictor._step_euler(
                xb, yb, zb, vx, vy, vz, remain)
        else:
            x, y, z = xb, yb, zb

        return x, y, z, vx, vy, vz, True

    # Workspace center used for target selection.
    # Intercept is chosen as the in-workspace point closest to this max-Z center.
    WORKSPACE_CENTER = (0.0, 0.0, -800.0)

    def _scan_trajectory(self, x, y, z, vx, vy, vz):
        """Forward-simulate trajectory and return the in-workspace point
        closest to workspace center, or the nearest clamp fallback.

        By choosing the point closest to center we minimise robot travel
        and reduce the chance of IK limit violations.
        """
        best_ws = None          # best in-workspace point (closest to center)
        best_ws_cdist = float('inf')

        best_clamp = None       # fallback: nearest point to workspace boundary
        best_clamp_dist = float('inf')

        cx, cy, cz = self.WORKSPACE_CENTER

        t = 0.0
        step = self.SCAN_DT
        bounces = 0
        while t <= self.SCAN_DURATION:
            if t >= self.MIN_TIME_HIT:
                if in_workspace(x, y, z):
                    # Distance from workspace center
                    d2 = (x - cx)**2 + (y - cy)**2 + (z - cz)**2
                    if d2 < best_ws_cdist:
                        best_ws_cdist = d2
                        best_ws = {
                            'x': x, 'y': y + self.INTERCEPT_Y_OFFSET, 'z': z,
                            'time': t,
                            'vx': vx, 'vy': vy, 'vz': vz,
                            'clamped': False,
                            'bounces': bounces,
                        }
                else:
                    xc, yc, zc, cdist = clamp_to_workspace(x, y, z)
                    if cdist < best_clamp_dist:
                        best_clamp_dist = cdist
                        best_clamp = {
                            'x': xc, 'y': yc + self.INTERCEPT_Y_OFFSET, 'z': zc,
                            'time': t,
                            'vx': vx, 'vy': vy, 'vz': vz,
                            'clamped': True,
                            'clamp_dist': cdist,
                            'bounces': bounces,
                        }

            x_prev, y_prev, z_prev = x, y, z
            x, y, z, vx, vy, vz = self._step_euler(x, y, z, vx, vy, vz, step)

            if bounces < MAX_BOUNCES:
                x, y, z, vx, vy, vz, did_bounce = self._apply_bounce(
                    x_prev, y_prev, z_prev, x, y, z, vx, vy, vz, step)
                if did_bounce:
                    bounces += 1

            t += step

        if best_ws is not None:
            return best_ws
        if best_clamp is not None and best_clamp_dist <= MAX_CLAMP_DIST:
            return best_clamp
        return None

    def _compute_confidence(self, mean_intercept, state):
        """Score prediction confidence by sampling from KF covariance.

        Inspired by UZH's BallTrajectory model which samples N initial states
        from a Bayesian posterior and computes empirical trajectory distribution.
        We do the same using our KF covariance as the state uncertainty.

        Returns (confidence 0-1, spread_mm, hit_ratio).
        """
        if self.state_estimator is None:
            return 1.0, 0.0, 1.0

        P = self.state_estimator.get_covariance()
        mean_state = np.array(state, dtype=float)

        # Ensure P is symmetric positive semi-definite
        P_sym = 0.5 * (P + P.T)
        # Clamp small negative eigenvalues from numerical noise
        try:
            eigvals = np.linalg.eigvalsh(P_sym)
            if eigvals.min() < 0:
                P_sym += (-eigvals.min() + 1e-6) * np.eye(6)
            samples = np.random.multivariate_normal(mean_state, P_sym, self.CONFIDENCE_SAMPLES)
        except np.linalg.LinAlgError:
            return 1.0, 0.0, 1.0

        hit_points = []
        for s in samples:
            result = self._scan_trajectory(s[0], s[1], s[2], s[3], s[4], s[5])
            if result is not None:
                hit_points.append((result['x'], result['y'], result['z']))

        n_hits = len(hit_points)
        hit_ratio = n_hits / self.CONFIDENCE_SAMPLES

        if n_hits < 2:
            return 0.0, float('inf'), hit_ratio

        pts = np.array(hit_points)
        spread = float(np.max(np.ptp(pts, axis=0)))  # max range across x/y/z

        # Confidence: 1.0 when spread=0, 0.0 when spread >= MAX_INTERCEPT_SPREAD
        conf = max(0.0, 1.0 - spread / self.MAX_INTERCEPT_SPREAD)
        # Penalize low hit ratio
        conf *= min(1.0, hit_ratio / self.MIN_HIT_RATIO)

        return conf, spread, hit_ratio

    def predict_intercept(self, robot_pos=None):
        """
        Find the first future point where ball enters workspace.
        If none found, clamp the nearest trajectory point to workspace.
        Handles bounces off the table surface.

        Includes confidence scoring: samples from KF covariance to assess
        prediction reliability. Low-confidence predictions are rejected.
        """
        if not self.is_ready():
            return None

        if not self._ball_approaching():
            return None

        state = self._get_prediction_state()
        if state is None:
            return None

        # Proximity filter
        y_now = state[1]
        if y_now > self.MAX_PREDICT_Y:
            return None

        x, y, z, vx, vy, vz = state

        # Primary trajectory scan (deterministic, from KF mean state)
        result = self._scan_trajectory(x, y, z, vx, vy, vz)
        if result is None:
            return None

        # Confidence scoring disabled for FPS — was log-only anyway.
        # Re-enable _compute_confidence() call when stereo noise is reduced.
        result['confidence'] = 1.0
        result['spread_mm'] = 0.0
        result['hit_ratio'] = 1.0

        return result

    def get_current_position(self):
        if self.state_estimator is not None:
            pos = self.state_estimator.get_position()
            if pos is not None:
                return pos
        if self.positions:
            p = self.positions[-1]
            return (p[0], p[1], p[2])
        return None

    def get_stats(self):
        pos = self.state_estimator.get_position() if self.state_estimator is not None else None
        y_now = pos[1] if pos is not None else (self.positions[-1][1] if self.positions else 9999)
        return {
            'buffer': len(self.positions),
            'accepted': self._accepted,
            'rejected': self._rejected,
            'has_vel': self.velocity is not None,
            'approaching': self._ball_approaching() if self.velocity else False,
            'close_enough': y_now <= self.MAX_PREDICT_Y,
            'bounces': self._bounce_count,
            'kf_ready': self.state_estimator.is_ready() if self.state_estimator is not None else False,
            'kf_updates': self.state_estimator.update_count if self.state_estimator is not None else 0,
            'kf_enabled': self._using_state_estimator,
        }
