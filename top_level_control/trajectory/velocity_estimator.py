"""
Velocity Estimator

Calculates ball velocity from position history.
Supports simple (last-first) and regression methods.

Note: This module is camera-agnostic.
"""

import numpy as np


class VelocityEstimator:
    """
    Estimates velocity from position history.
    
    Methods:
        'simple': (last - first) / dt  (fast, noisy)
        'regression': Linear fit (slower, noise-robust)
    """

    def __init__(self, method='regression'):
        """
        Initialize velocity estimator.

        Args:
            method: 'simple' or 'regression'
        """
        self.method = method

    def estimate_simple(self, positions, times):
        """
        Simple velocity: (last - first) / dt
        
        Fast but sensitive to noise.
        """
        if len(positions) < 2:
            return None
        
        dt = times[-1] - times[0]
        if dt <= 0:
            return None
        
        return (positions[-1] - positions[0]) / dt

    def estimate_regression(self, positions, times):
        """
        Regression velocity: Linear fit slope.
        
        More robust to noise.
        """
        if len(positions) < 2:
            return None
        
        # Normalize time for numerical stability
        t0 = times[0]
        t_norm = times - t0
        
        # Linear fit: position = velocity * t + offset
        try:
            coeffs = np.polyfit(t_norm, positions, 1)
            velocity = coeffs[0]  # Slope
            return velocity
        except:
            return None

    def estimate(self, x_arr, y_arr, z_arr, t_arr):
        """
        Estimate velocity for all axes.

        Args:
            x_arr, y_arr, z_arr: Position arrays
            t_arr: Time array

        Returns:
            dict with 'vx', 'vy', 'vz', 'speed', 'valid'
        """
        result = {
            'vx': 0.0,
            'vy': 0.0,
            'vz': 0.0,
            'speed': 0.0,
            'valid': False
        }

        if len(t_arr) < 2:
            return result

        # Choose estimation method
        if self.method == 'simple':
            vx = self.estimate_simple(x_arr, t_arr)
            vy = self.estimate_simple(y_arr, t_arr)
            vz = self.estimate_simple(z_arr, t_arr)
        else:
            vx = self.estimate_regression(x_arr, t_arr)
            vy = self.estimate_regression(y_arr, t_arr)
            vz = self.estimate_regression(z_arr, t_arr)

        if vx is None or vy is None or vz is None:
            return result

        result['vx'] = vx
        result['vy'] = vy
        result['vz'] = vz
        result['speed'] = np.sqrt(vx**2 + vy**2 + vz**2)
        result['valid'] = True

        return result

    def estimate_from_buffer(self, buffer):
        """
        Convenience method to estimate from PositionBuffer.

        Args:
            buffer: PositionBuffer instance

        Returns:
            dict with velocity info
        """
        arrays = buffer.get_as_arrays()
        return self.estimate(
            arrays['x'], arrays['y'], arrays['z'], arrays['t']
        )