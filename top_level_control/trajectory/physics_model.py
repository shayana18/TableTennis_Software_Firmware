"""
Physics Model - Ball Trajectory Physics

Implements physics equations for table tennis ball trajectory:
  - Gravity (primary effect on Y axis)
  - Air drag (optional, decelerates all axes)
  - Axis-plane interception (stop when ball reaches target X, Y, or Z)
  - Apex detection (highest point of arc)

Coordinate system (camera frame):
    X: Along table length (ball travels between players on this axis)
    Y: Vertical (positive = down in camera coords)
    Z: Across table width (depth from camera)

    Gravity acts on Y only (no X/Z acceleration without drag).

Part of trajectory prediction pipeline:
  position_buffer.py  →  velocity_estimator.py  →  trajectory_predictor.py
                                                          ↓
                                                   physics_model.py
"""

import math
import numpy as np


# ================================================================
# Physical constants
# ================================================================

GRAVITY_CM_S2 = 981.0       # cm/s²  (9.81 m/s²)
GRAVITY_MM_S2 = 9810.0      # mm/s²

# Table tennis ball (CGS for cm/s system)
BALL_MASS_G = 2.7
BALL_RADIUS_CM = 2.0
AIR_DENSITY_G_CM3 = 0.001225
DEFAULT_DRAG_CD = 0.45


class PhysicsModel:
    """
    Physics model for ball trajectory prediction.

    IMPORTANT — Coordinate convention:
        Camera Y is typically DOWN (positive = downward).
        Gravity acts in +Y direction (pulls ball down = increasing Y).
        Set y_down=True for standard camera coordinates.

    Usage:
        model = PhysicsModel(gravity=981, y_down=True)

        # Predict where ball will be at camera X = 120 cm:
        result = model.position_at_x(pos, vel, target_x=120)

        # Predict apex (highest point):
        result = model.position_at_apex(pos, vel)
    """

    def __init__(self, gravity=GRAVITY_CM_S2, y_down=True,
                 enable_drag=False, drag_coefficient=DEFAULT_DRAG_CD):
        self.gravity = gravity
        self.y_down = y_down
        self.enable_drag = enable_drag
        self.drag_coefficient = drag_coefficient

        # +1 if Y increases downward (camera), -1 if Y increases upward
        self.gravity_sign = 1.0 if y_down else -1.0

        # Precompute drag constant: k = (ρ · Cd · A) / (2 · m)
        cross_section = math.pi * BALL_RADIUS_CM ** 2
        self.drag_k = (AIR_DENSITY_G_CM3 * drag_coefficient * cross_section) / (2.0 * BALL_MASS_G)

    # ================================================================
    # NUMERICAL SIMULATION (with drag)
    # ================================================================

    def _simulate(self, position, velocity, max_time=2.0,
                  stop_at_x=None, stop_at_z=None, stop_at_apex=False,
                  record_trajectory=False, record_dt=0.01,
                  sim_dt=0.001):
        """
        Numerical integration with air drag (semi-implicit Euler).

        Stop conditions (first one triggered wins):
            stop_at_x: Stop when X crosses this value
            stop_at_z: Stop when Z crosses this value
            stop_at_apex: Stop when Vy changes sign (rising → falling)

        Returns:
            dict with 'position', 'time', 'velocity', 'valid',
            and optionally 'trajectory' list of (X, Y, Z, t).
        """
        x, y, z = float(position[0]), float(position[1]), float(position[2])
        vx, vy, vz = float(velocity[0]), float(velocity[1]), float(velocity[2])
        k = self.drag_k
        g_acc = self.gravity_sign * self.gravity

        trajectory = [] if record_trajectory else None
        next_record_t = 0.0
        t = 0.0

        prev_x, prev_y, prev_z = x, y, z
        prev_vx, prev_vy, prev_vz = vx, vy, vz

        steps = int(max_time / sim_dt)

        for _ in range(steps):
            if record_trajectory and t >= next_record_t:
                trajectory.append((x, y, z, t))
                next_record_t += record_dt

            # Save previous state
            prev_x, prev_y, prev_z = x, y, z
            prev_vx, prev_vy, prev_vz = vx, vy, vz
            prev_vy_sign = vy
            prev_t = t

            # Drag deceleration
            if self.enable_drag:
                speed = math.sqrt(vx*vx + vy*vy + vz*vz)
                drag = -k * speed
                ax = drag * vx
                ay = drag * vy + g_acc
                az = drag * vz
            else:
                ax = 0.0
                ay = g_acc
                az = 0.0

            # Semi-implicit Euler
            vx += ax * sim_dt
            vy += ay * sim_dt
            vz += az * sim_dt
            x += vx * sim_dt
            y += vy * sim_dt
            z += vz * sim_dt
            t += sim_dt

            # --- Stop condition: X-plane crossing ---
            if stop_at_x is not None:
                d_prev = prev_x - stop_at_x
                d_curr = x - stop_at_x
                if d_prev * d_curr <= 0 and abs(d_prev - d_curr) > 1e-12:
                    frac = abs(d_prev) / abs(d_prev - d_curr)
                    result = self._interpolate_stop(
                        prev_x, prev_y, prev_z, prev_vx, prev_vy, prev_vz,
                        x, y, z, vx, vy, vz, prev_t, sim_dt, frac, trajectory)
                    result['position'] = (stop_at_x,
                                          result['position'][1],
                                          result['position'][2])
                    return result

            # --- Stop condition: Z-plane crossing ---
            if stop_at_z is not None:
                d_prev = prev_z - stop_at_z
                d_curr = z - stop_at_z
                if d_prev * d_curr <= 0 and abs(d_prev - d_curr) > 1e-12:
                    frac = abs(d_prev) / abs(d_prev - d_curr)
                    result = self._interpolate_stop(
                        prev_x, prev_y, prev_z, prev_vx, prev_vy, prev_vz,
                        x, y, z, vx, vy, vz, prev_t, sim_dt, frac, trajectory)
                    result['position'] = (result['position'][0],
                                          result['position'][1],
                                          stop_at_z)
                    return result

            # --- Stop condition: Apex (Vy sign change) ---
            if stop_at_apex and prev_vy_sign * vy <= 0 and abs(prev_vy_sign) > 1e-9:
                denom = abs(prev_vy_sign) + abs(vy)
                frac = abs(prev_vy_sign) / denom if denom > 1e-12 else 0.5
                result = self._interpolate_stop(
                    prev_x, prev_y, prev_z, prev_vx, prev_vy, prev_vz,
                    x, y, z, vx, vy, vz, prev_t, sim_dt, frac, trajectory)
                result['velocity'] = (result['velocity'][0],
                                      0.0,
                                      result['velocity'][2])
                return result

        # Ran out of time
        if record_trajectory:
            trajectory.append((x, y, z, t))

        valid = (stop_at_x is None and stop_at_z is None and not stop_at_apex)
        return {
            'position': (x, y, z),
            'time': t,
            'velocity': (vx, vy, vz),
            'valid': valid,
            'trajectory': trajectory
        }

    def _interpolate_stop(self, px, py, pz, pvx, pvy, pvz,
                          cx, cy, cz, cvx, cvy, cvz,
                          prev_t, sim_dt, frac, trajectory):
        """Linear interpolation at a stop-condition crossing."""
        ix = px + frac * (cx - px)
        iy = py + frac * (cy - py)
        iz = pz + frac * (cz - pz)
        ivx = pvx + frac * (cvx - pvx)
        ivy = pvy + frac * (cvy - pvy)
        ivz = pvz + frac * (cvz - pvz)
        it = prev_t + frac * sim_dt

        if trajectory is not None:
            trajectory.append((ix, iy, iz, it))

        return {
            'position': (ix, iy, iz),
            'time': it,
            'velocity': (ivx, ivy, ivz),
            'valid': True,
            'trajectory': trajectory
        }

    # ================================================================
    # CLOSED-FORM KINEMATICS (no drag)
    # ================================================================

    def predict_position(self, position, velocity, dt):
        """
        Predict position after time dt.

        X(t) = X₀ + Vx·t           (no X acceleration)
        Y(t) = Y₀ + Vy·t + ½g·t²  (gravity on Y)
        Z(t) = Z₀ + Vz·t           (no Z acceleration)
        """
        if self.enable_drag:
            result = self._simulate(position, velocity, max_time=dt)
            return result['position']

        x0, y0, z0 = position
        vx, vy, vz = velocity
        g = self.gravity_sign * self.gravity

        return (x0 + vx * dt,
                y0 + vy * dt + 0.5 * g * dt * dt,
                z0 + vz * dt)

    def predict_velocity(self, velocity, dt):
        """
        Predict velocity after time dt.

        Vx unchanged, Vy += g·t, Vz unchanged.
        """
        if self.enable_drag:
            result = self._simulate((0, 0, 0), velocity, max_time=dt)
            return result['velocity']

        vx, vy, vz = velocity
        return (vx,
                vy + self.gravity_sign * self.gravity * dt,
                vz)

    def predict_trajectory(self, position, velocity, duration, dt=0.001):
        """Predict full trajectory as list of (X, Y, Z, t)."""
        if self.enable_drag:
            result = self._simulate(position, velocity, max_time=duration,
                                    record_trajectory=True, record_dt=dt)
            return result['trajectory']

        trajectory = []
        t = 0.0
        while t <= duration:
            pos = self.predict_position(position, velocity, t)
            trajectory.append((pos[0], pos[1], pos[2], t))
            t += dt
        return trajectory

    # ================================================================
    # AXIS-PLANE INTERCEPTION
    # ================================================================

    def time_to_x(self, position, velocity, target_x):
        """Time for ball to reach target X. Returns None if unreachable."""
        vx = velocity[0]
        if abs(vx) < 1e-6:
            return None
        t = (target_x - position[0]) / vx
        return t if t > 0 else None

    def time_to_z(self, position, velocity, target_z):
        """Time for ball to reach target Z. Returns None if unreachable."""
        vz = velocity[2]
        if abs(vz) < 1e-6:
            return None
        t = (target_z - position[2]) / vz
        return t if t > 0 else None

    def position_at_x(self, position, velocity, target_x):
        """
        Predict ball state when it reaches target_x (along table length).

        This is the PRIMARY interception method for table tennis —
        the ball travels along X between players.

        Returns:
            dict with 'position', 'time', 'velocity', 'valid'
        """
        if self.enable_drag:
            return self._simulate(position, velocity, stop_at_x=target_x)

        fail = {'position': None, 'time': None, 'velocity': None, 'valid': False}
        t = self.time_to_x(position, velocity, target_x)
        if t is None:
            return fail

        return {
            'position': self.predict_position(position, velocity, t),
            'time': t,
            'velocity': self.predict_velocity(velocity, t),
            'valid': True
        }

    def position_at_z(self, position, velocity, target_z):
        """
        Predict ball state when it reaches target_z (across table width).

        Returns:
            dict with 'position', 'time', 'velocity', 'valid'
        """
        if self.enable_drag:
            return self._simulate(position, velocity, stop_at_z=target_z)

        fail = {'position': None, 'time': None, 'velocity': None, 'valid': False}
        t = self.time_to_z(position, velocity, target_z)
        if t is None:
            return fail

        return {
            'position': self.predict_position(position, velocity, t),
            'time': t,
            'velocity': self.predict_velocity(velocity, t),
            'valid': True
        }

    def position_at_apex(self, position, velocity):
        """
        Predict ball state at trajectory apex (where Vy = 0).

        Apex exists only when ball is rising (Vy opposes gravity direction).
        Without drag: t_apex = -Vy / (g_sign * g)

        Returns:
            dict with 'position', 'time', 'velocity', 'valid'
        """
        fail = {'position': None, 'time': None, 'velocity': None, 'valid': False}
        vy = velocity[1]

        if self.enable_drag:
            if vy * self.gravity_sign >= 0:
                return fail
            return self._simulate(position, velocity, stop_at_apex=True)

        t_apex = -vy / (self.gravity_sign * self.gravity)
        if t_apex <= 0:
            return fail

        return {
            'position': self.predict_position(position, velocity, t_apex),
            'time': t_apex,
            'velocity': self.predict_velocity(velocity, t_apex),
            'valid': True
        }


# ================================================================
# Convenience function
# ================================================================

def predict_ball_position(position, velocity, dt,
                          gravity=GRAVITY_CM_S2, y_down=True, enable_drag=False):
    model = PhysicsModel(gravity=gravity, y_down=y_down, enable_drag=enable_drag)
    return model.predict_position(position, velocity, dt)