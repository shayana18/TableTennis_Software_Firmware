"""
Robot-frame trajectory predictor — single source of truth.

Extracted verbatim from test_integration_simple.py (v5), which is proven
in real robot testing. All prediction operates in robot frame (mm).

Pipeline:
  1. Add robot-frame positions via add_position(x, y, z, t)
  2. Velocity estimated via least-squares regression with gravity correction
  3. Trajectory forward-simulated with gravity + air drag (Euler integration)
  4. Interception found via workspace scan + clamp fallback
"""

from __future__ import annotations

import math
import time
from collections import deque

import numpy as np

from .workspace import (
    ELLIPSE_A, ELLIPSE_B, Z_MIN, Z_MAX, MAX_CLAMP_DIST,
    ROBOT_HOME, GRAVITY_Z, DRAG_K,
    in_workspace, clamp_to_workspace,
)


class RobotPredictor:
    """
    Trajectory predictor in robot frame (mm).
    Uses Euler integration with gravity + air drag.
    """

    BUFFER_SIZE    = 15
    MIN_POINTS     = 6      # minimum for regression
    MIN_TIME_SPAN  = 0.08   # seconds -- require 80ms of data (~8 frames)
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

    def __init__(self):
        self.positions = deque(maxlen=self.BUFFER_SIZE)
        self.velocity = None
        self._rejected = 0
        self._accepted = 0
        self._last_reject_reason = None

    def reset(self):
        self.positions.clear()
        self.velocity = None

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

        if len(self.positions) >= self.MIN_POINTS:
            span = self.positions[-1][3] - self.positions[0][3]
            if span >= self.MIN_TIME_SPAN:
                self._estimate_velocity()

        return True

    def _estimate_velocity(self):
        """Estimate velocity via least-squares regression."""
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

        # Z with gravity: z - 0.5*g*dt^2 = z0 + vz*dt
        zs_corrected = zs - 0.5 * GRAVITY_Z * dt * dt
        vz = float(np.linalg.lstsq(A, zs_corrected, rcond=None)[0][0])

        speed = math.sqrt(vx*vx + vy*vy + vz*vz)
        if speed > self.MAX_SPEED:
            self.velocity = None
            return

        self.velocity = (vx, vy, vz)

    def is_ready(self):
        if self.velocity is None:
            return False
        if not self.positions:
            return False
        age = time.perf_counter() - self.positions[-1][3]
        return age < self.STALE_TIMEOUT

    def _ball_approaching(self):
        """Check if ball is moving toward the workspace."""
        if not self.positions or self.velocity is None:
            return False
        y_now = self.positions[-1][1]
        vy = self.velocity[1]
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

    def predict_intercept(self, robot_pos=None):
        """
        Find the first future point where ball enters workspace.
        If none found, clamp the nearest trajectory point to workspace.
        """
        if not self.is_ready():
            return None

        if not self._ball_approaching():
            return None

        # Proximity filter
        y_now = self.positions[-1][1]
        if y_now > self.MAX_PREDICT_Y:
            return None

        x, y, z, _ = self.positions[-1]
        vx, vy, vz = self.velocity

        rp = robot_pos if robot_pos is not None else ROBOT_HOME

        # Track closest-to-workspace point for fallback
        best_clamp = None
        best_clamp_dist = float('inf')

        t = 0.0
        step = self.SCAN_DT
        while t <= self.SCAN_DURATION:
            if t >= self.MIN_TIME_HIT:
                if in_workspace(x, y, z):
                    return {
                        'x': x, 'y': y, 'z': z,
                        'time': t,
                        'vx': vx, 'vy': vy, 'vz': vz,
                        'clamped': False,
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
                    }

            x, y, z, vx, vy, vz = self._step_euler(x, y, z, vx, vy, vz, step)
            t += step

        # Fallback: clamp nearest point if within MAX_CLAMP_DIST
        if best_clamp is not None and best_clamp_dist <= MAX_CLAMP_DIST:
            return best_clamp

        return None

    def get_current_position(self):
        if not self.positions:
            return None
        p = self.positions[-1]
        return (p[0], p[1], p[2])

    def get_stats(self):
        y_now = self.positions[-1][1] if self.positions else 9999
        return {
            'buffer': len(self.positions),
            'accepted': self._accepted,
            'rejected': self._rejected,
            'has_vel': self.velocity is not None,
            'approaching': self._ball_approaching() if self.velocity else False,
            'close_enough': y_now <= self.MAX_PREDICT_Y,
        }
