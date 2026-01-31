"""
Trajectory Predictor - Main Prediction Class

Combines position buffer, velocity estimation, and physics model
to predict ball trajectory and interception point.

This is the main class you use in your application.

Part of trajectory prediction pipeline:
  position_buffer.py  →  velocity_estimator.py  →  trajectory_predictor.py
                                                          ↓
                                                   physics_model.py

Usage:
    predictor = TrajectoryPredictor()
    
    # Each frame, add triangulated position:
    predictor.add_position(X, Y, Z)
    
    # Get prediction:
    result = predictor.predict(target_z=50)  # Where will ball be at Z=50?
    if result['valid']:
        print(f"Intercept at X={result['intercept_x']}, Y={result['intercept_y']}")
        print(f"Time to intercept: {result['time_to_intercept']*1000:.0f} ms")
"""

import time
import numpy as np

from .position_buffer import PositionBuffer
from .velocity_estimator import VelocityEstimator
from .physics_model import PhysicsModel


class TrajectoryPredictor:
    """
    Main trajectory prediction class.
    
    Collects 3D positions, estimates velocity, predicts trajectory.
    
    Typical workflow:
        1. Create predictor
        2. Each frame: call add_position(x, y, z)
        3. When ready: call predict(target_z) to get interception point
    """
    
    def __init__(self, 
                 buffer_size=10,
                 min_points=3,
                 velocity_method='regression',
                 gravity=981.0,
                 y_down=True):
        """
        Initialize trajectory predictor.
        
        Args:
            buffer_size: Max positions to store (default 10)
            min_points: Min points needed before prediction (default 3)
            velocity_method: 'simple' or 'regression' (default 'regression')
            gravity: Gravity in cm/s² (default 981)
            y_down: True if +Y is downward in camera coords (default True)
        """
        self.buffer_size = buffer_size
        self.min_points = min_points
        
        # Initialize components
        self.position_buffer = PositionBuffer(max_size=buffer_size)
        self.velocity_estimator = VelocityEstimator(method=velocity_method)
        self.physics_model = PhysicsModel(gravity=gravity, y_down=y_down)
        
        # Cached velocity (updated on each add_position)
        self._current_velocity = None
        self._velocity_valid = False
    
    def add_position(self, x, y, z, timestamp=None):
        """
        Add a new 3D position from triangulation.
        
        Call this every frame when ball is detected.
        
        Args:
            x, y, z: 3D position (in calibration units, e.g., cm)
            timestamp: Time in seconds (auto-generated if None)
        """
        self.position_buffer.add(x, y, z, timestamp)
        
        # Update velocity estimate if enough points
        if self.position_buffer.is_ready(self.min_points):
            self._update_velocity()
    
    def add_position_tuple(self, position_3d, timestamp=None):
        """
        Add position from tuple (convenience method).
        
        Args:
            position_3d: (X, Y, Z) tuple
            timestamp: Time in seconds (auto-generated if None)
        """
        self.add_position(position_3d[0], position_3d[1], position_3d[2], timestamp)
    
    def _update_velocity(self):
        """Update cached velocity estimate."""
        vel = self.velocity_estimator.estimate_from_buffer(self.position_buffer)
        self._current_velocity = vel
        self._velocity_valid = vel['valid']
    
    def get_velocity(self):
        """
        Get current velocity estimate.
        
        Returns:
            dict with 'vx', 'vy', 'vz', 'speed', 'valid'
        """
        if self._current_velocity is None:
            return {'vx': 0, 'vy': 0, 'vz': 0, 'speed': 0, 'valid': False}
        return self._current_velocity
    
    def get_current_position(self):
        """
        Get most recent position.
        
        Returns:
            (X, Y, Z) or None if no positions
        """
        latest = self.position_buffer.get_latest()
        if latest is None:
            return None
        return (latest['x'], latest['y'], latest['z'])
    
    def is_ready(self):
        """
        Check if predictor has enough data for prediction.
        
        Returns:
            True if ready to predict
        """
        return self.position_buffer.is_ready(self.min_points) and self._velocity_valid
    
    def predict(self, target_z):
        """
        Predict where ball will be when it reaches target_z.
        
        This is the main prediction method.
        
        Args:
            target_z: Z distance where robot can intercept (e.g., 50 cm)
        
        Returns:
            dict with:
                'valid': True if prediction successful
                'intercept_x': X position at target_z
                'intercept_y': Y position at target_z
                'intercept_z': Should equal target_z
                'time_to_intercept': Seconds until ball reaches target_z
                'velocity_at_intercept': (Vx, Vy, Vz) at intercept
                'current_position': Current (X, Y, Z)
                'current_velocity': Current (Vx, Vy, Vz)
        """
        result = {
            'valid': False,
            'intercept_x': None,
            'intercept_y': None,
            'intercept_z': target_z,
            'time_to_intercept': None,
            'velocity_at_intercept': None,
            'current_position': None,
            'current_velocity': None
        }
        
        # Check if ready
        if not self.is_ready():
            return result
        
        # Get current state
        current_pos = self.get_current_position()
        current_vel = self.get_velocity()
        
        if current_pos is None or not current_vel['valid']:
            return result
        
        result['current_position'] = current_pos
        result['current_velocity'] = (current_vel['vx'], current_vel['vy'], current_vel['vz'])
        
        # Use physics model to predict intercept
        prediction = self.physics_model.position_at_z(
            position=current_pos,
            velocity=result['current_velocity'],
            target_z=target_z
        )
        
        if not prediction['valid']:
            return result
        
        # Fill result
        result['valid'] = True
        result['intercept_x'] = prediction['position'][0]
        result['intercept_y'] = prediction['position'][1]
        result['intercept_z'] = prediction['position'][2]
        result['time_to_intercept'] = prediction['time']
        result['velocity_at_intercept'] = prediction['velocity']
        
        return result
    
    def predict_trajectory(self, duration=0.5, dt=0.005):
        """
        Predict full trajectory for visualization.
        
        Args:
            duration: How far into future to predict (seconds)
            dt: Time step (smaller = smoother, default 5ms)
        
        Returns:
            List of (X, Y, Z, t) tuples, or empty list if not ready
        """
        if not self.is_ready():
            return []
        
        current_pos = self.get_current_position()
        current_vel = self.get_velocity()
        
        velocity = (current_vel['vx'], current_vel['vy'], current_vel['vz'])
        
        return self.physics_model.predict_trajectory(
            position=current_pos,
            velocity=velocity,
            duration=duration,
            dt=dt
        )
    
    def reset(self):
        """Clear all position history and reset predictor."""
        self.position_buffer.clear()
        self._current_velocity = None
        self._velocity_valid = False
    
    def get_stats(self):
        """
        Get current predictor statistics.
        
        Returns:
            dict with buffer stats, velocity info, etc.
        """
        vel = self.get_velocity()
        
        return {
            'buffer_size': len(self.position_buffer),
            'buffer_max': self.buffer_size,
            'is_ready': self.is_ready(),
            'time_span': self.position_buffer.get_time_span(),
            'avg_dt': self.position_buffer.get_average_dt(),
            'velocity_valid': vel['valid'],
            'speed': vel['speed'] if vel['valid'] else 0.0
        }