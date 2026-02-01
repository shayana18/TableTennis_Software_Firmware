"""
Tracking Module for Basler Cameras

Ball detection and 3D triangulation using Basler acA1920-150uc cameras.

COMPONENTS:
    EnhancedBallTracker  - HSV+LAB ball detection (camera-agnostic)
    StereoDetector       - Detect ball in both cameras (no 3D)
    StereoTriangulator   - Full 3D tracking with triangulation

REQUIREMENTS:
    pip install pypylon
"""

from .ball_tracker import EnhancedBallTracker, BallTracker
from .stereo_detector import StereoDetector
from .stereo_triangulator import StereoTriangulator

__all__ = [
    'EnhancedBallTracker',
    'BallTracker',
    'StereoDetector',
    'StereoTriangulator'
]