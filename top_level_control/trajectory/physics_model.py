"""
Physics Model - Ball Trajectory Physics

Implements physics equations for table tennis ball trajectory:
  - Gravity (primary effect)
  - Air drag (optional, minor effect for short distances)
  - Bounce detection (optional)

Part of trajectory prediction pipeline:
  position_buffer.py  →  velocity_estimator.py  →  trajectory_predictor.py
                                                          ↓
                                                   physics_model.py
"""

import math
import numpy as np


# Physical constants
GRAVITY_CM_S2 = 981.0      # Gravity in cm/s² (9.81 m/s² = 981 cm/s²)
GRAVITY_M_S2 = 9.81        # Gravity in m/s²

# Table tennis ball physical constants (CGS units for cm/s system)
BALL_MASS_G = 2.7              # grams
BALL_RADIUS_CM = 2.0           # cm
AIR_DENSITY_G_CM3 = 0.001225   # g/cm³ (= 1.225 kg/m³)
DEFAULT_DRAG_CD = 0.45         # drag coefficient at playing speeds


class PhysicsModel:
    """
    Physics model for ball trajectory prediction.
    
    Coordinate system (matches stereo calibration):
        X: Horizontal (left/right)
        Y: Vertical (positive = down in image, but we treat as up physically)
        Z: Depth (toward ball, positive = further from camera)
    
    IMPORTANT: In camera coordinates, +Y is typically DOWN.
               Gravity acts in +Y direction (pulls ball down = increasing Y)
               Adjust gravity_sign based on your coordinate system.
    
    Usage:
        model = PhysicsModel(gravity=981, y_down=True)
        future_pos = model.predict_position(pos, vel, dt=0.1)
    """
    
    def __init__(self, gravity=981.0, y_down=True, enable_drag=False, drag_coefficient=DEFAULT_DRAG_CD):
        """
        Initialize physics model.

        Args:
            gravity: Gravitational acceleration (cm/s² if using cm)
            y_down: True if +Y is downward (typical camera coords)
            enable_drag: Include air resistance
            drag_coefficient: Air drag coefficient Cd (default 0.45)
        """
        self.gravity = gravity
        self.y_down = y_down
        self.enable_drag = enable_drag
        self.drag_coefficient = drag_coefficient

        # Gravity direction: +1 if Y increases downward, -1 if Y increases upward
        self.gravity_sign = 1.0 if y_down else -1.0

        # Precompute drag constant: k = (rho * Cd * A) / (2 * m)
        cross_section = math.pi * BALL_RADIUS_CM ** 2
        self.drag_k = (AIR_DENSITY_G_CM3 * drag_coefficient * cross_section) / (2.0 * BALL_MASS_G)
    
    def _simulate(self, position, velocity, max_time=2.0,
                  stop_at_z=None, stop_at_apex=False,
                  record_trajectory=False, record_dt=0.01,
                  sim_dt=0.001):
        """
        Numerical integration engine with air drag (semi-implicit Euler).

        Args:
            position: (X, Y, Z) starting position
            velocity: (Vx, Vy, Vz) starting velocity
            max_time: Safety time limit (seconds)
            stop_at_z: Stop when Z crosses this value (None = disabled)
            stop_at_apex: Stop when Vy changes sign (rising → falling)
            record_trajectory: If True, record waypoints every record_dt
            record_dt: Spacing between recorded trajectory points (seconds)
            sim_dt: Internal simulation timestep (seconds)

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
        prev_vy = vy
        prev_x, prev_y, prev_z = x, y, z
        prev_vx, prev_vy_store, prev_vz = vx, vy, vz

        steps = int(max_time / sim_dt)

        for _ in range(steps):
            # Record trajectory waypoint
            if record_trajectory and t >= next_record_t:
                trajectory.append((x, y, z, t))
                next_record_t += record_dt

            # Save previous state for interpolation
            prev_x, prev_y, prev_z = x, y, z
            prev_vx, prev_vy_store, prev_vz = vx, vy, vz
            prev_vy = vy
            prev_t = t

            # Compute drag deceleration
            speed = math.sqrt(vx * vx + vy * vy + vz * vz)
            drag_factor = -k * speed

            ax = drag_factor * vx
            ay = drag_factor * vy + g_acc
            az = drag_factor * vz

            # Semi-implicit Euler: update velocity first, then position
            vx += ax * sim_dt
            vy += ay * sim_dt
            vz += az * sim_dt

            x += vx * sim_dt
            y += vy * sim_dt
            z += vz * sim_dt

            t += sim_dt

            # Check Z-crossing stop condition
            if stop_at_z is not None:
                # Detect crossing: (prev_z - target) and (z - target) have opposite signs
                d_prev = prev_z - stop_at_z
                d_curr = z - stop_at_z
                if d_prev * d_curr <= 0 and abs(d_prev - d_curr) > 1e-12:
                    # Linear interpolation for sub-step accuracy
                    frac = abs(d_prev) / abs(d_prev - d_curr)
                    ix = prev_x + frac * (x - prev_x)
                    iy = prev_y + frac * (y - prev_y)
                    iz = stop_at_z
                    ivx = prev_vx + frac * (vx - prev_vx)
                    ivy = prev_vy_store + frac * (vy - prev_vy_store)
                    ivz = prev_vz + frac * (vz - prev_vz)
                    it = prev_t + frac * sim_dt
                    if record_trajectory:
                        trajectory.append((ix, iy, iz, it))
                    return {
                        'position': (ix, iy, iz),
                        'time': it,
                        'velocity': (ivx, ivy, ivz),
                        'valid': True,
                        'trajectory': trajectory
                    }

            # Check apex stop condition (Vy sign change: rising → falling)
            if stop_at_apex and prev_vy * vy <= 0 and abs(prev_vy) > 1e-9:
                # Linear interpolation
                frac = abs(prev_vy) / (abs(prev_vy) + abs(vy)) if abs(prev_vy) + abs(vy) > 1e-12 else 0.5
                ix = prev_x + frac * (x - prev_x)
                iy = prev_y + frac * (y - prev_y)
                iz = prev_z + frac * (z - prev_z)
                it = prev_t + frac * sim_dt
                # Velocity at apex: Vy ≈ 0 by definition
                ivx = prev_vx + frac * (vx - prev_vx)
                ivz = prev_vz + frac * (vz - prev_vz)
                if record_trajectory:
                    trajectory.append((ix, iy, iz, it))
                return {
                    'position': (ix, iy, iz),
                    'time': it,
                    'velocity': (ivx, 0.0, ivz),
                    'valid': True,
                    'trajectory': trajectory
                }

        # Ran out of time without hitting stop condition
        if record_trajectory:
            trajectory.append((x, y, z, t))

        # For duration-based runs (no stop condition), return final state as valid
        valid = (stop_at_z is None and not stop_at_apex)
        return {
            'position': (x, y, z),
            'time': t,
            'velocity': (vx, vy, vz),
            'valid': valid,
            'trajectory': trajectory
        }

    def predict_position(self, position, velocity, dt):
        """
        Predict position after time dt using kinematic equations.
        
        Equations:
            X(t) = X₀ + Vx × t
            Y(t) = Y₀ + Vy × t + ½ × g × t²
            Z(t) = Z₀ + Vz × t
        
        Args:
            position: (X, Y, Z) current position
            velocity: (Vx, Vy, Vz) current velocity
            dt: Time step in seconds
        
        Returns:
            (X_new, Y_new, Z_new) predicted position
        """
        if self.enable_drag:
            result = self._simulate(position, velocity, max_time=dt)
            return result['position']

        x0, y0, z0 = position
        vx, vy, vz = velocity

        x_new = x0 + vx * dt
        y_new = y0 + vy * dt + 0.5 * self.gravity_sign * self.gravity * dt * dt
        z_new = z0 + vz * dt

        return (x_new, y_new, z_new)
    
    def predict_velocity(self, velocity, dt):
        """
        Predict velocity after time dt.
        
        Equations:
            Vx(t) = Vx₀           (no horizontal deceleration)
            Vy(t) = Vy₀ + g × t   (gravity accelerates downward)
            Vz(t) = Vz₀           (no depth deceleration)
        
        Args:
            velocity: (Vx, Vy, Vz) current velocity
            dt: Time step in seconds
        
        Returns:
            (Vx_new, Vy_new, Vz_new) predicted velocity
        """
        if self.enable_drag:
            result = self._simulate((0, 0, 0), velocity, max_time=dt)
            return result['velocity']

        vx, vy, vz = velocity

        vx_new = vx
        vy_new = vy + self.gravity_sign * self.gravity * dt
        vz_new = vz

        return (vx_new, vy_new, vz_new)
    
    def predict_trajectory(self, position, velocity, duration, dt=0.001):
        """
        Predict full trajectory over duration using exact kinematics.

        Args:
            position: (X, Y, Z) starting position
            velocity: (Vx, Vy, Vz) starting velocity
            duration: Total prediction time in seconds
            dt: Time step between sample points

        Returns:
            List of (X, Y, Z, t) tuples representing trajectory
        """
        if self.enable_drag:
            result = self._simulate(position, velocity, max_time=duration,
                                    record_trajectory=True, record_dt=dt)
            return result['trajectory']

        trajectory = []
        t = 0.0

        while t <= duration:
            pred_pos = self.predict_position(position, velocity, t)
            trajectory.append((pred_pos[0], pred_pos[1], pred_pos[2], t))
            t += dt

        return trajectory
    
    def time_to_z(self, position, velocity, target_z):
        """
        Calculate time for ball to reach a specific Z distance.
        
        Assumes constant Vz (no Z-axis acceleration).
        
        Args:
            position: (X, Y, Z) current position
            velocity: (Vx, Vy, Vz) current velocity
            target_z: Target Z value (e.g., robot's reach plane)
        
        Returns:
            Time in seconds, or None if ball won't reach target_z
        """
        z0 = position[2]
        vz = velocity[2]
        
        # Z(t) = Z₀ + Vz × t = target_z
        # t = (target_z - Z₀) / Vz
        
        if abs(vz) < 1e-6:
            # Ball not moving in Z
            return None
        
        t = (target_z - z0) / vz
        
        if t < 0:
            # Target is behind current position
            return None
        
        return t
    
    def position_at_z(self, position, velocity, target_z):
        """
        Calculate ball position when it reaches target_z.

        Args:
            position: (X, Y, Z) current position
            velocity: (Vx, Vy, Vz) current velocity
            target_z: Target Z value

        Returns:
            dict with:
                'position': (X, Y, Z) at target_z
                'time': Time to reach target_z
                'velocity': (Vx, Vy, Vz) at that time
                'valid': True if ball will reach target_z
        """
        if self.enable_drag:
            return self._simulate(position, velocity, stop_at_z=target_z)

        result = {
            'position': None,
            'time': None,
            'velocity': None,
            'valid': False
        }

        t = self.time_to_z(position, velocity, target_z)

        if t is None:
            return result

        pred_pos = self.predict_position(position, velocity, t)
        pred_vel = self.predict_velocity(velocity, t)

        result['position'] = pred_pos
        result['time'] = t
        result['velocity'] = pred_vel
        result['valid'] = True

        return result

    def position_at_apex(self, position, velocity):
        """
        Calculate ball position at trajectory apex (where Vy = 0).

        The apex occurs when Vy changes sign from rising to falling.
        Without drag: t_apex = -Vy / (gravity_sign * gravity).
        With drag: found numerically via _simulate().

        Args:
            position: (X, Y, Z) current position
            velocity: (Vx, Vy, Vz) current velocity

        Returns:
            dict with:
                'position': (X, Y, Z) at apex
                'time': Time to reach apex
                'velocity': (Vx, Vy, Vz) at apex (Vy ≈ 0)
                'valid': True if ball is rising and apex exists
        """
        if self.enable_drag:
            vy = velocity[1]
            # Ball must be rising (Vy opposes gravity) for apex to exist
            if vy * self.gravity_sign >= 0:
                return {'position': None, 'time': None, 'velocity': None, 'valid': False}
            return self._simulate(position, velocity, stop_at_apex=True)

        result = {
            'position': None,
            'time': None,
            'velocity': None,
            'valid': False
        }

        vy = velocity[1]
        t_apex = -vy / (self.gravity_sign * self.gravity)

        if t_apex <= 0:
            return result

        pred_pos = self.predict_position(position, velocity, t_apex)
        pred_vel = self.predict_velocity(velocity, t_apex)

        result['position'] = pred_pos
        result['time'] = t_apex
        result['velocity'] = pred_vel
        result['valid'] = True

        return result


def predict_ball_position(position, velocity, dt, gravity=981.0, y_down=True, enable_drag=False):
    """
    Convenience function for single position prediction.

    Args:
        position: (X, Y, Z) current position
        velocity: (Vx, Vy, Vz) current velocity
        dt: Time step in seconds
        gravity: Gravitational acceleration (cm/s²)
        y_down: True if +Y is downward
        enable_drag: Include air resistance

    Returns:
        (X_new, Y_new, Z_new) predicted position
    """
    model = PhysicsModel(gravity=gravity, y_down=y_down, enable_drag=enable_drag)
    return model.predict_position(position, velocity, dt)