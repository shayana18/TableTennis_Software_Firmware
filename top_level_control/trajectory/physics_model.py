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

import numpy as np


# Physical constants
GRAVITY_CM_S2 = 981.0      # Gravity in cm/s² (9.81 m/s² = 981 cm/s²)
GRAVITY_M_S2 = 9.81        # Gravity in m/s²


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
    
    def __init__(self, gravity=981.0, y_down=True, enable_drag=False, drag_coefficient=0.5):
        """
        Initialize physics model.
        
        Args:
            gravity: Gravitational acceleration (cm/s² if using cm)
            y_down: True if +Y is downward (typical camera coords)
            enable_drag: Include air resistance (usually not needed)
            drag_coefficient: Air drag coefficient (if enabled)
        """
        self.gravity = gravity
        self.y_down = y_down
        self.enable_drag = enable_drag
        self.drag_coefficient = drag_coefficient
        
        # Gravity direction: +1 if Y increases downward, -1 if Y increases upward
        self.gravity_sign = 1.0 if y_down else -1.0
    
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
        x0, y0, z0 = position
        vx, vy, vz = velocity
        
        # Basic kinematics
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
        result = {
            'position': None,
            'time': None,
            'velocity': None,
            'valid': False
        }
        
        t = self.time_to_z(position, velocity, target_z)
        
        if t is None:
            return result
        
        # Predict position and velocity at time t
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

        The apex occurs at t_apex = -Vy / (gravity_sign * gravity).
        This is positive only when the ball is rising (Vy opposes gravity),
        so it naturally returns invalid for non-rising trajectories.

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


def predict_ball_position(position, velocity, dt, gravity=981.0, y_down=True):
    """
    Convenience function for single position prediction.
    
    Args:
        position: (X, Y, Z) current position
        velocity: (Vx, Vy, Vz) current velocity
        dt: Time step in seconds
        gravity: Gravitational acceleration (cm/s²)
        y_down: True if +Y is downward
    
    Returns:
        (X_new, Y_new, Z_new) predicted position
    """
    model = PhysicsModel(gravity=gravity, y_down=y_down)
    return model.predict_position(position, velocity, dt)