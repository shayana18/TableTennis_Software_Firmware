"""
Trajectory Prediction Module

Predicts ball trajectory using physics-based model with gravity.

COMPONENTS:
    PositionBuffer      - Stores timestamped 3D positions (circular buffer)
    VelocityEstimator   - Calculates velocity from position history
    PhysicsModel        - Kinematic equations with gravity
    TrajectoryPredictor - Main class that combines all components

USAGE:
    from trajectory import TrajectoryPredictor
    
    # Create predictor
    predictor = TrajectoryPredictor(
        buffer_size=10,      # Store last 10 positions
        min_points=3,        # Need 3+ points to predict
        gravity=981.0,       # cm/s² (use 9810 for mm/s²)
        y_down=True          # +Y is downward in camera coords
    )
    
    # In your tracking loop:
    while tracking:
        result = triangulator.update()
        
        if result['found_3d']:
            x, y, z = result['position_3d']
            predictor.add_position(x, y, z)
        
        # Predict where ball will be at Z = robot_reach
        prediction = predictor.predict(target_z=50)
        
        if prediction['valid']:
            robot_x = prediction['intercept_x']
            robot_y = prediction['intercept_y']
            time_ms = prediction['time_to_intercept'] * 1000
            
            # Send to robot!
            robot.move_to(robot_x, robot_y, time_ms)

PHYSICS:
    The predictor uses basic kinematics with gravity:
        X(t) = X₀ + Vx × t           (constant horizontal velocity)
        Y(t) = Y₀ + Vy × t + ½g×t²   (gravity accelerates downward)
        Z(t) = Z₀ + Vz × t           (constant depth velocity)
    
    Gravity = 981 cm/s² = 9.81 m/s²
"""

from .position_buffer import PositionBuffer
from .velocity_estimator import VelocityEstimator, estimate_velocity
from .physics_model import PhysicsModel, predict_ball_position
from .trajectory_predictor import TrajectoryPredictor

__all__ = [
    'PositionBuffer',
    'VelocityEstimator',
    'estimate_velocity',
    'PhysicsModel',
    'predict_ball_position',
    'TrajectoryPredictor'
]

__version__ = '1.0.0'