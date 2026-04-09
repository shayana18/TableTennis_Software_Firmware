"""
DEPRECATED — used only by trajectory_predictor.py (deprecated camera-frame predictor).
For new code, use trajectory.robot_predictor.RobotPredictor which has integrated velocity
estimation with gravity correction.

Velocity Estimator - Calculate Ball Velocity from Position History

Computes velocity vector (Vx, Vy, Vz) from timestamped positions.
Uses linear regression for noise-robust estimation.
"""

import numpy as np


class VelocityEstimator:
    """
    Estimate ball velocity from position history.
    
    Methods:
        - Simple: (last - first) / dt
        - Linear regression: Fit line to all points (more robust)
    
    Usage:
        estimator = VelocityEstimator()
        velocity = estimator.estimate(positions, timestamps)
        # velocity = {'vx': ..., 'vy': ..., 'vz': ..., 'speed': ...}
    """
    
    def __init__(self, method='regression'):
        """
        Initialize velocity estimator.
        
        Args:
            method: 'simple' or 'regression'
                    - simple: (P_last - P_first) / dt
                    - regression: Linear fit (recommended, noise-robust)
        """
        self.method = method
    
    def estimate_simple(self, positions, timestamps):
        """
        Simple velocity: (last position - first position) / time.
        
        Args:
            positions: np.array shape (N, 3)
            timestamps: np.array shape (N,)
        
        Returns:
            (vx, vy, vz) velocity in units/second
        """
        if len(positions) < 2:
            return (0.0, 0.0, 0.0)
        
        dt = timestamps[-1] - timestamps[0]
        
        if dt <= 0:
            return (0.0, 0.0, 0.0)
        
        dp = positions[-1] - positions[0]
        velocity = dp / dt
        
        return tuple(velocity)
    
    def estimate_regression(self, positions, timestamps):
        """
        Linear regression velocity: Fit line to positions vs time.
        
        More robust to noise than simple method.
        
        Args:
            positions: np.array shape (N, 3)
            timestamps: np.array shape (N,)
        
        Returns:
            (vx, vy, vz) velocity in units/second
        """
        if len(positions) < 2:
            return (0.0, 0.0, 0.0)
        
        # Normalize timestamps to start at 0
        t = timestamps - timestamps[0]
        
        # Linear regression for each axis: position = velocity * t + offset
        # Using numpy polyfit (degree 1 = linear)
        vx = np.polyfit(t, positions[:, 0], 1)[0]
        vy = np.polyfit(t, positions[:, 1], 1)[0]
        vz = np.polyfit(t, positions[:, 2], 1)[0]
        
        return (vx, vy, vz)
    
    def estimate(self, positions, timestamps):
        """
        Estimate velocity using configured method.
        
        Args:
            positions: np.array shape (N, 3) - [[x,y,z], ...]
            timestamps: np.array shape (N,) - [t1, t2, ...]
        
        Returns:
            dict with:
                'vx': X velocity (units/sec)
                'vy': Y velocity (units/sec)
                'vz': Z velocity (units/sec)
                'speed': Total speed (units/sec)
                'valid': True if estimation succeeded
        """
        result = {
            'vx': 0.0,
            'vy': 0.0,
            'vz': 0.0,
            'speed': 0.0,
            'valid': False
        }
        
        # Need at least 2 points
        if len(positions) < 2:
            return result
        
        # Estimate velocity
        if self.method == 'simple':
            vx, vy, vz = self.estimate_simple(positions, timestamps)
        else:
            vx, vy, vz = self.estimate_regression(positions, timestamps)
        
        # Calculate speed (magnitude)
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        
        result['vx'] = float(vx)
        result['vy'] = float(vy)
        result['vz'] = float(vz)
        result['speed'] = float(speed)
        result['valid'] = True
        
        return result
    
    def estimate_from_buffer(self, position_buffer, n_points=None):
        """
        Estimate velocity directly from PositionBuffer.
        
        Args:
            position_buffer: PositionBuffer instance
            n_points: Number of recent points to use (None = all)
        
        Returns:
            dict with 'vx', 'vy', 'vz', 'speed', 'valid'
        """
        positions, timestamps = position_buffer.get_as_arrays(n_points)
        return self.estimate(positions, timestamps)


def estimate_velocity(positions, timestamps, method='regression'):
    """
    Convenience function to estimate velocity.
    
    Args:
        positions: np.array shape (N, 3)
        timestamps: np.array shape (N,)
        method: 'simple' or 'regression'
    
    Returns:
        dict with 'vx', 'vy', 'vz', 'speed', 'valid'
    """
    estimator = VelocityEstimator(method=method)
    return estimator.estimate(positions, timestamps)