"""
Trajectory Module

Ball trajectory prediction using physics-based modeling.

COMPONENTS:
    PositionBuffer      - Stores timestamped 3D positions
    VelocityEstimator   - Calculates velocity from position history
    PhysicsModel        - Kinematic equations with gravity
    TrajectoryPredictor - Main class combining all components

Note: This module is camera-agnostic.
      Works with any stereo triangulation output (Arducam, Basler, etc.)

USAGE:
    from trajectory import TrajectoryPredictor
    
    predictor = TrajectoryPredictor()
    
    # In tracking loop:
    predictor.add_position(x, y, z)
    
    # Get interception point:
    prediction = predictor.predict(target_z=50)
    
    if prediction['valid']:
        robot_x = prediction['intercept_x']
        robot_y = prediction['intercept_y']
        time_ms = prediction['time_to_intercept'] * 1000
"""

from .position_buffer import PositionBuffer
from .velocity_estimator import VelocityEstimator
from .physics_model import PhysicsModel, GRAVITY_CM_S2
from .trajectory_predictor import TrajectoryPredictor

__all__ = [
    'PositionBuffer',
    'VelocityEstimator',
    'PhysicsModel',
    'TrajectoryPredictor',
    'GRAVITY_CM_S2'
]

__version__ = '1.0.0'


# Convenience functions
def estimate_velocity(positions, times, method='regression'):
    """
    Quick velocity estimation.
    
    Args:
        positions: List of (x, y, z) tuples
        times: List of timestamps
        method: 'simple' or 'regression'
    
    Returns:
        dict with 'vx', 'vy', 'vz', 'speed', 'valid'
    """
    import numpy as np
    estimator = VelocityEstimator(method=method)
    x = np.array([p[0] for p in positions])
    y = np.array([p[1] for p in positions])
    z = np.array([p[2] for p in positions])
    t = np.array(times)
    return estimator.estimate(x, y, z, t)


def predict_ball_position(x0, y0, z0, vx, vy, vz, t, gravity=981.0):
    """
    Quick position prediction.
    
    Args:
        x0, y0, z0: Initial position
        vx, vy, vz: Initial velocity
        t: Time (seconds)
        gravity: Gravity in cm/s² (default 981)
    
    Returns:
        (x, y, z) predicted position
    """
    physics = PhysicsModel(gravity=gravity)
    return physics.predict_position(x0, y0, z0, vx, vy, vz, t)