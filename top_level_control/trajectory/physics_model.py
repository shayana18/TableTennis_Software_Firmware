"""
Physics Model

Kinematic equations for ball trajectory prediction.
Includes gravity for realistic parabolic motion.

Note: This module is camera-agnostic.

COORDINATE SYSTEM:
    X: Horizontal (right positive)
    Y: Vertical (down positive in camera coords)
    Z: Depth (away from camera positive)

UNITS:
    Position: Same as calibration (typically cm)
    Velocity: cm/s (or mm/s depending on calibration)
    Gravity: 981 cm/s² (default, adjust for mm)
"""

import numpy as np

# Gravity constant in cm/s² (9.81 m/s² = 981 cm/s²)
GRAVITY_CM_S2 = 981.0


class PhysicsModel:
    """
    Physics-based trajectory prediction.
    
    Uses kinematic equations:
        X(t) = X₀ + Vx·t
        Y(t) = Y₀ + Vy·t + ½·g·t²
        Z(t) = Z₀ + Vz·t
    """

    def __init__(self, gravity=GRAVITY_CM_S2, y_down=True):
        """
        Initialize physics model.

        Args:
            gravity: Gravity acceleration (default 981 cm/s²)
            y_down: If True, Y increases downward (camera coords)
        """
        self.gravity = gravity
        self.y_down = y_down
        
        # In camera coordinates, Y typically increases downward
        # So gravity is positive when y_down=True
        self.gravity_sign = 1 if y_down else -1

    def predict_position(self, x0, y0, z0, vx, vy, vz, t):
        """
        Predict position at time t.

        Args:
            x0, y0, z0: Initial position
            vx, vy, vz: Initial velocity
            t: Time (seconds)

        Returns:
            (x, y, z) predicted position
        """
        # X: constant velocity (no air resistance)
        x = x0 + vx * t
        
        # Y: constant velocity + gravity
        # Y(t) = Y₀ + Vy·t + ½·g·t²
        g = self.gravity * self.gravity_sign
        y = y0 + vy * t + 0.5 * g * t * t
        
        # Z: constant velocity (no air resistance)
        z = z0 + vz * t
        
        return x, y, z

    def predict_velocity(self, vx, vy, vz, t):
        """
        Predict velocity at time t.

        Args:
            vx, vy, vz: Initial velocity
            t: Time (seconds)

        Returns:
            (vx, vy, vz) predicted velocity
        """
        # X, Z: constant (no air resistance)
        vx_t = vx
        vz_t = vz
        
        # Y: affected by gravity
        # Vy(t) = Vy₀ + g·t
        g = self.gravity * self.gravity_sign
        vy_t = vy + g * t
        
        return vx_t, vy_t, vz_t

    def time_to_z(self, z0, vz, target_z):
        """
        Calculate time to reach target Z.

        Args:
            z0: Current Z position
            vz: Z velocity
            target_z: Target Z position

        Returns:
            Time in seconds (None if unreachable)
        """
        if vz == 0:
            return None
        
        t = (target_z - z0) / vz
        
        # Only return positive time (future)
        if t > 0:
            return t
        return None

    def position_at_z(self, x0, y0, z0, vx, vy, vz, target_z):
        """
        Calculate position when ball reaches target Z.

        Args:
            x0, y0, z0: Current position
            vx, vy, vz: Current velocity
            target_z: Target Z plane

        Returns:
            dict with 'x', 'y', 'z', 't', 'vx', 'vy', 'vz', 'valid'
        """
        result = {
            'x': None,
            'y': None,
            'z': target_z,
            't': None,
            'vx': None,
            'vy': None,
            'vz': None,
            'valid': False
        }
        
        # Calculate time to reach target Z
        t = self.time_to_z(z0, vz, target_z)
        
        if t is None:
            return result
        
        # Predict position and velocity at that time
        x, y, z = self.predict_position(x0, y0, z0, vx, vy, vz, t)
        vx_t, vy_t, vz_t = self.predict_velocity(vx, vy, vz, t)
        
        result['x'] = x
        result['y'] = y
        result['z'] = z
        result['t'] = t
        result['vx'] = vx_t
        result['vy'] = vy_t
        result['vz'] = vz_t
        result['valid'] = True
        
        return result

    def predict_trajectory(self, x0, y0, z0, vx, vy, vz, duration, dt=0.01):
        """
        Generate full trajectory for visualization.

        Args:
            x0, y0, z0: Initial position
            vx, vy, vz: Initial velocity
            duration: Total time (seconds)
            dt: Time step (seconds)

        Returns:
            List of (x, y, z, t) tuples
        """
        trajectory = []
        t = 0
        
        while t <= duration:
            x, y, z = self.predict_position(x0, y0, z0, vx, vy, vz, t)
            trajectory.append((x, y, z, t))
            t += dt
        
        return trajectory