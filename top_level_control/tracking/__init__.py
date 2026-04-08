"""
Tracking Module

Ball detection using MOG2 background subtraction.
Detection logic lives in ball_detector.py (single source of truth).
"""

from .ball_detector import BallDetector
from .ball_tracker import EnhancedBallTracker, BallTracker
from .stereo_detector import StereoDetector
from .stereo_triangulator import StereoTriangulator

__all__ = ['BallDetector', 'EnhancedBallTracker', 'BallTracker',
           'StereoDetector', 'StereoTriangulator']