"""
Trajectory Prediction Module

COMPONENTS:
    RobotPredictor  - Robot-frame predictor (single source of truth for real-time)
    BallStateEstimator3D - 3D robot-frame state estimator
    workspace       - Workspace constants and boundary checks
    PhysicsModel    - Camera-frame kinematic equations (for analysis scripts)

USAGE (real-time robot control):
    from trajectory.robot_predictor import RobotPredictor
    from trajectory.workspace import ROBOT_HOME, in_workspace
    from comm_function.points_based_transform import cam_to_robot

    predictor = RobotPredictor()

    # In your tracking loop:
    while tracking:
        result = triangulator.update()
        if result['found_3d']:
            cx, cy, cz = result['position_3d']
            rx, ry, rz = cam_to_robot(R, t, scale, cx, cy, cz)
            predictor.add_position(rx, ry, rz, timestamp)

        intercept = predictor.predict_intercept()
        if intercept is not None:
            send_to_robot(intercept['x'], intercept['y'], intercept['z'],
                          intercept['time'])

USAGE (post-hoc analysis in camera frame):
    from trajectory.physics_model import PhysicsModel
    model = PhysicsModel(gravity=981, y_down=True, enable_drag=True)
"""

from .robot_predictor import RobotPredictor
from .ball_state_estimation import BallStateEstimator3D
from .workspace import (
    ELLIPSE_A, ELLIPSE_B, Z_MIN, Z_MAX, MAX_CLAMP_DIST,
    ROBOT_HOME, MAX_CART_VEL, MAX_CART_ACC,
    CM_TO_MM, GRAVITY_Z, DRAG_K,
    Z_TABLE_SURFACE, RESTITUTION_COEFF, FRICTION_COEFF, MAX_BOUNCES,
    in_workspace, clamp_to_workspace,
)
from .physics_model import PhysicsModel, predict_ball_position

__all__ = [
    'RobotPredictor',
    'BallStateEstimator3D',
    'PhysicsModel',
    'predict_ball_position',
    'ELLIPSE_A', 'ELLIPSE_B', 'Z_MIN', 'Z_MAX', 'MAX_CLAMP_DIST',
    'ROBOT_HOME', 'MAX_CART_VEL', 'MAX_CART_ACC',
    'CM_TO_MM', 'GRAVITY_Z', 'DRAG_K',
    'Z_TABLE_SURFACE', 'RESTITUTION_COEFF', 'FRICTION_COEFF', 'MAX_BOUNCES',
    'in_workspace', 'clamp_to_workspace',
]

__version__ = '2.0.0'
