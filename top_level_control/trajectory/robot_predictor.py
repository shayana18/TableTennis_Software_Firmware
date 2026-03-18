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
    MIN_POINTS     = 6      # minimum updates before ready
    MIN_TIME_SPAN  = 0.08   # seconds of measurement history before ready
    MAX_SPEED      = 15000.0 # mm/s
    MAX_JUMP       = 400.0   # mm
    GAP_RESET      = 0.12   # seconds
    STALE_TIMEOUT  = 0.15   # seconds

    # Prediction scan
    SCAN_DURATION  = 1.5    # seconds forward
    SCAN_DT        = 0.005  # 5ms steps
    MIN_TIME_HIT   = 0.10   # minimum reaction time for robot

    # Proximity filter
    MAX_PREDICT_Y  = 1400.0  # mm -- ball must be closer than this

    # Direction filter
    MIN_APPROACH_VY = -200.0
    APPROACH_Y_THRESHOLD = 600.0

    # Observed bounce detection
    MIN_BOUNCE_FALL_Z  = 50.0   # mm minimum Z descent before accepting bounce
    BOUNCE_RISE_FRAMES = 2      # consecutive rising frames to confirm

    def __init__(self):
        self.positions = deque(maxlen=self.BUFFER_SIZE)
        self.velocity = None
        self.state_estimator = None
        self._using_state_estimator = False
        try:
            self.state_estimator = BallStateEstimator3D(
                gravity_z=GRAVITY_Z,
                max_gap_s=self.GAP_RESET,
                min_updates=self.MIN_POINTS,
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
        self._bounce_count = 0

    def reset(self):
        self.positions.clear()
        self.velocity = None
        if self.state_estimator is not None:
            self.state_estimator.reset()
        self._z_min_since_reset = None
        self._rising_count = 0
        self._bounce_count = 0

    def add_position(self, x, y, z, t):
        """Add a robot-frame position (mm). Returns True if accepted."""
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
            # Replaces regression-based velocity estimation with KF state updates.
            accepted, _state = self.state_estimator.estimate(x, y, z, t)
            if not accepted:
                self.positions.pop()
                self._accepted -= 1
                self._rejected += 1
                self._last_reject_reason = "kf_dt"
                return False

            self.velocity = (
                self.state_estimator.get_velocity()
                if self.state_estimator.is_ready()
                else None
            )
        elif len(self.positions) >= self.MIN_POINTS:
            span = self.positions[-1][3] - self.positions[0][3]
            if span >= self.MIN_TIME_SPAN:
                self._estimate_velocity_legacy()

        return True

    def _detect_observed_bounce(self):
        """Detect Z reversal (ball bouncing off table) in observed data."""
        if len(self.positions) < 3:
            return False

        z = self.positions[-1][2]

        # Track minimum Z since reset
        if self._z_min_since_reset is None:
            self._z_min_since_reset = z
        else:
            self._z_min_since_reset = min(self._z_min_since_reset, z)

        # Check if ball is rising
        z_prev = self.positions[-2][2]
        if z > z_prev:
            self._rising_count += 1
        else:
            self._rising_count = 0
            return False

        # Need enough consecutive rising frames
        if self._rising_count < self.BOUNCE_RISE_FRAMES:
            return False

        # Must have fallen enough from first buffered Z
        z_first = self.positions[0][2]
        fall = z_first - self._z_min_since_reset
        return fall >= self.MIN_BOUNCE_FALL_Z

    def _handle_observed_bounce(self):
        """Reset buffer after observed bounce, keeping last 2 points."""
        keep = list(self.positions)[-2:]
        self.positions.clear()
        for p in keep:
            self.positions.append(p)

        if self.state_estimator is not None:
            # Bounce is a state jump, so restart the KF on the new arc.
            self.state_estimator.reset()
            if self.positions:
                last = self.positions[-1]
                self.state_estimator.initialize_from_measurement(*last)

        self.velocity = None
        self._z_min_since_reset = None
        self._rising_count = 0
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
        if self.state_estimator is not None and not self.state_estimator.is_ready():
            return False
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
        """Return the current state used to seed future prediction."""
        if self.state_estimator is not None:
            state = self.state_estimator.get_state()
            if state is not None:
                return state
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

    def predict_intercept(self, robot_pos=None):
        """
        Find the first future point where ball enters workspace.
        If none found, clamp the nearest trajectory point to workspace.
        Handles bounces off the table surface.
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

        # Replaces raw last-sample start with the current prediction state.
        x, y, z, vx, vy, vz = state

        # Track closest-to-workspace point for fallback
        best_clamp = None
        best_clamp_dist = float('inf')

        t = 0.0
        step = self.SCAN_DT
        bounces = 0
        while t <= self.SCAN_DURATION:
            if t >= self.MIN_TIME_HIT:
                if in_workspace(x, y, z):
                    return {
                        'x': x, 'y': y, 'z': z,
                        'time': t,
                        'vx': vx, 'vy': vy, 'vz': vz,
                        'clamped': False,
                        'bounces': bounces,
                    }

                # Track closest point for fallback
                xc, yc, zc, cdist = clamp_to_workspace(x, y, z)
                if cdist < best_clamp_dist:
                    best_clamp_dist = cdist
                    best_clamp = {
                        'x': xc, 'y': yc, 'z': zc,
                        'time': t,
                        'vx': vx, 'vy': vy, 'vz': vz,
                        'clamped': True,
                        'clamp_dist': cdist,
                        'bounces': bounces,
                    }

            x_prev, y_prev, z_prev = x, y, z
            x, y, z, vx, vy, vz = self._step_euler(x, y, z, vx, vy, vz, step)

            # Bounce off table surface
            if bounces < MAX_BOUNCES:
                x, y, z, vx, vy, vz, did_bounce = self._apply_bounce(
                    x_prev, y_prev, z_prev, x, y, z, vx, vy, vz, step)
                if did_bounce:
                    bounces += 1

            t += step

        # Fallback: clamp nearest point if within MAX_CLAMP_DIST
        if best_clamp is not None and best_clamp_dist <= MAX_CLAMP_DIST:
            return best_clamp

        return None

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
