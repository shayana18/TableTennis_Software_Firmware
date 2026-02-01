"""
Trajectory Predictor

Main class that combines position buffer, velocity estimator,
and physics model to predict ball trajectory and interception point.

Note: This module is camera-agnostic.

USAGE:
    predictor = TrajectoryPredictor()
    
    # In tracking loop:
    predictor.add_position(x, y, z)
    
    # Get interception point at robot's reach plane:
    prediction = predictor.predict(target_z=50)
    
    if prediction['valid']:
        robot_x = prediction['intercept_x']
        robot_y = prediction['intercept_y']
        time_ms = prediction['time_to_intercept'] * 1000
"""

import time
import numpy as np

from .position_buffer import PositionBuffer
from .velocity_estimator import VelocityEstimator
from .physics_model import PhysicsModel, GRAVITY_CM_S2


class TrajectoryPredictor:
    """
    Predicts ball trajectory and interception point.
    
    Pipeline:
        1. Collect positions (PositionBuffer)
        2. Estimate velocity (VelocityEstimator)
        3. Predict trajectory with gravity (PhysicsModel)
        4. Calculate interception point at target Z
    """

    def __init__(self, buffer_size=10, min_points=3,
                 velocity_method='regression', gravity=GRAVITY_CM_S2, y_down=True):
        """
        Initialize trajectory predictor.

        Args:
            buffer_size: Max positions to store
            min_points: Min positions needed for prediction
            velocity_method: 'simple' or 'regression'
            gravity: Gravity in same units as calibration (981 cm/s²)
            y_down: True if Y increases downward (camera coords)
        """
        self.min_points = min_points
        
        # Components
        self.buffer = PositionBuffer(max_size=buffer_size)
        self.velocity_estimator = VelocityEstimator(method=velocity_method)
        self.physics = PhysicsModel(gravity=gravity, y_down=y_down)
        
        # Cached velocity
        self._velocity = None

    def add_position(self, x, y, z, t=None):
        """
        Add a new position measurement.

        Args:
            x, y, z: 3D position from triangulation
            t: Timestamp (auto-generated if None)
        """
        self.buffer.add(x, y, z, t)
        
        # Update velocity estimate
        if self.buffer.is_ready(self.min_points):
            self._velocity = self.velocity_estimator.estimate_from_buffer(self.buffer)

    def is_ready(self):
        """Check if predictor has enough data."""
        return self.buffer.is_ready(self.min_points)

    def get_velocity(self):
        """
        Get current velocity estimate.

        Returns:
            dict with 'vx', 'vy', 'vz', 'speed', 'valid'
        """
        if self._velocity is None:
            return {
                'vx': 0.0,
                'vy': 0.0,
                'vz': 0.0,
                'speed': 0.0,
                'valid': False
            }
        return self._velocity

    def predict(self, target_z):
        """
        Predict interception point at target Z plane.

        Args:
            target_z: Z coordinate of interception plane (robot's reach)

        Returns:
            dict with prediction results:
                'valid': bool - is prediction valid
                'intercept_x': X at interception
                'intercept_y': Y at interception
                'time_to_intercept': seconds until interception
                'velocity_at_intercept': (vx, vy, vz)
                'current_position': (x, y, z)
                'current_velocity': (vx, vy, vz)
        """
        result = {
            'valid': False,
            'intercept_x': None,
            'intercept_y': None,
            'time_to_intercept': None,
            'velocity_at_intercept': None,
            'current_position': None,
            'current_velocity': None
        }
        
        # Need enough points
        if not self.is_ready():
            return result
        
        # Need valid velocity
        vel = self.get_velocity()
        if not vel['valid']:
            return result
        
        # Get current position (most recent)
        latest = self.buffer.latest
        if latest is None:
            return result
        
        x0, y0, z0 = latest['x'], latest['y'], latest['z']
        vx, vy, vz = vel['vx'], vel['vy'], vel['vz']
        
        result['current_position'] = (x0, y0, z0)
        result['current_velocity'] = (vx, vy, vz)
        
        # Ball must be moving toward target (Vz has correct sign)
        # If target_z > z0, vz should be positive
        # If target_z < z0, vz should be negative
        if (target_z - z0) * vz <= 0:
            # Ball moving away from target
            return result
        
        # Predict position at target Z
        pred = self.physics.position_at_z(x0, y0, z0, vx, vy, vz, target_z)
        
        if not pred['valid']:
            return result
        
        result['valid'] = True
        result['intercept_x'] = pred['x']
        result['intercept_y'] = pred['y']
        result['time_to_intercept'] = pred['t']
        result['velocity_at_intercept'] = (pred['vx'], pred['vy'], pred['vz'])
        
        return result

    def predict_trajectory(self, duration=0.5, dt=0.01):
        """
        Generate full trajectory for visualization.

        Args:
            duration: Prediction time (seconds)
            dt: Time step (seconds)

        Returns:
            List of (x, y, z, t) tuples
        """
        if not self.is_ready():
            return []
        
        vel = self.get_velocity()
        if not vel['valid']:
            return []
        
        latest = self.buffer.latest
        if latest is None:
            return []
        
        return self.physics.predict_trajectory(
            latest['x'], latest['y'], latest['z'],
            vel['vx'], vel['vy'], vel['vz'],
            duration, dt
        )

    def reset(self):
        """Clear all data and reset predictor."""
        self.buffer.clear()
        self._velocity = None

    def get_stats(self):
        """Get buffer and velocity stats."""
        return {
            'buffer_size': len(self.buffer),
            'min_points': self.min_points,
            'is_ready': self.is_ready(),
            'velocity': self.get_velocity()
        }