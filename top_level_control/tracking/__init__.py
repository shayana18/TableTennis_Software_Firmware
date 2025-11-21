"""
Table Tennis Ball Tracking Module

Contains:
- ball_tracker: HSV color-based ball detection
- physics_model: Velocity tracking and position prediction (2D radius-based)
- triangulation: 3D position from stereo cameras
- stereo_tracker: Dual camera ball detection with 3D output
"""

from .ball_tracker import BallTracker
from .physics_model import PhysicsModel
from .triangulation import triangulate_point, compute_disparity
from .stereo_tracker import StereoTracker

__all__ = [
    'BallTracker',
    'PhysicsModel',
    'triangulate_point',
    'compute_disparity',
    'StereoTracker'
]
