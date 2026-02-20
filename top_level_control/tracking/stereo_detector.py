"""
Stereo Ball Detector - Arducam Edition
======================================
Detects ball in two cameras simultaneously.
Optimized for Arducam OV9782 Global Shutter USB cameras.
NO triangulation - just detection in both views.

CAMERA: Arducam OV9782 Global Shutter USB Camera
        1MP, 100fps @ 1280x800 MJPG
"""

import cv2
import numpy as np
import json
from .ball_tracker import EnhancedBallTracker
from config.camera_config import (
    configure_camera, CAMERA_LEFT_ID, CAMERA_RIGHT_ID, FRAME_WIDTH, FRAME_HEIGHT
)


class StereoDetector:
    """Detect ball in stereo camera pair. No 3D calculations."""

    def __init__(self, cam_left_id=None, cam_right_id=None, thresholds_file=None, **kwargs):
        """
        Initialize stereo detector.

        Args:
            cam_left_id: Device ID for left camera  (default: from camera_config)
            cam_right_id: Device ID for right camera (default: from camera_config)
            thresholds_file: Optional path to ball_thresholds.json
        """
        self.cam_left_id = cam_left_id if cam_left_id is not None else CAMERA_LEFT_ID
        self.cam_right_id = cam_right_id if cam_right_id is not None else CAMERA_RIGHT_ID

        # Camera objects
        self.cap_left = None
        self.cap_right = None

        # Ball trackers for each camera (independent thresholds)
        self.tracker_left = EnhancedBallTracker()
        self.tracker_right = EnhancedBallTracker()
        
        # Load custom thresholds if provided
        if thresholds_file:
            self.load_thresholds(thresholds_file)

    def load_thresholds(self, filepath):
        """Load HSV and LAB thresholds from JSON file."""
        try:
            with open(filepath, 'r') as f:
                thresholds = json.load(f)
            
            # Check for stereo format (separate left/right thresholds)
            if 'left' in thresholds and 'right' in thresholds:
                # Stereo format
                left_th = thresholds['left']
                right_th = thresholds['right']
                
                if 'hsv_lower' in left_th:
                    self.tracker_left.set_hsv_thresholds(left_th['hsv_lower'], left_th['hsv_upper'])
                if 'lab_lower' in left_th:
                    self.tracker_left.set_lab_thresholds(left_th['lab_lower'], left_th['lab_upper'])
                    
                if 'hsv_lower' in right_th:
                    self.tracker_right.set_hsv_thresholds(right_th['hsv_lower'], right_th['hsv_upper'])
                if 'lab_lower' in right_th:
                    self.tracker_right.set_lab_thresholds(right_th['lab_lower'], right_th['lab_upper'])
                
                print(f"[StereoDetector] Loaded STEREO thresholds from {filepath}")
            else:
                # Legacy single format - apply to both
                if 'hsv_lower' in thresholds:
                    self.tracker_left.set_hsv_thresholds(thresholds['hsv_lower'], thresholds['hsv_upper'])
                    self.tracker_right.set_hsv_thresholds(thresholds['hsv_lower'], thresholds['hsv_upper'])
                
                if 'lab_lower' in thresholds:
                    self.tracker_left.set_lab_thresholds(thresholds['lab_lower'], thresholds['lab_upper'])
                    self.tracker_right.set_lab_thresholds(thresholds['lab_lower'], thresholds['lab_upper'])
                
                print(f"[StereoDetector] Loaded thresholds from {filepath}")
        except Exception as e:
            print(f"[StereoDetector] Warning: Could not load thresholds: {e}")

    def start_cameras(self, width=None, height=None):
        """
        Open camera streams with Arducam MJPG configuration.
        Defaults come from camera_config.py.
        """
        if width is None:
            width = FRAME_WIDTH
        if height is None:
            height = FRAME_HEIGHT
        self.cap_left = cv2.VideoCapture(self.cam_left_id)
        self.cap_right = cv2.VideoCapture(self.cam_right_id)

        if not self.cap_left.isOpened():
            raise RuntimeError(f"Failed to open left camera (ID: {self.cam_left_id})")
        if not self.cap_right.isOpened():
            raise RuntimeError(f"Failed to open right camera (ID: {self.cam_right_id})")

        # Configure for Arducam OV9782 with MJPG + trigger mode from yaml
        print("\n[StereoDetector] Configuring cameras (Arducam OV9782 MJPG mode):")
        settings_left = configure_camera(self.cap_left, width, height)
        settings_right = configure_camera(self.cap_right, width, height)

        print(f"  LEFT:  {settings_left['width']}x{settings_left['height']} "
              f"@ {settings_left['fps']:.0f}fps ({settings_left['fourcc']})")
        print(f"  RIGHT: {settings_right['width']}x{settings_right['height']} "
              f"@ {settings_right['fps']:.0f}fps ({settings_right['fourcc']})")

        if settings_left.get('trigger_mode'):
            tl = 'OK' if settings_left.get('trigger_ok') else 'FAILED'
            tr = 'OK' if settings_right.get('trigger_ok') else 'FAILED'
            print(f"  TRIGGER: LEFT={tl}  RIGHT={tr}")

        if not settings_left['settings_match'] or not settings_right['settings_match']:
            print("  WARNING: Some camera settings don't match requested values")

        print("[StereoDetector] Cameras started successfully!")
        return True

    def stop_cameras(self):
        """Release camera streams."""
        if self.cap_left:
            self.cap_left.release()
        if self.cap_right:
            self.cap_right.release()
        print("[StereoDetector] Cameras stopped")

    def read_frames(self):
        """
        Read frames from both cameras.
        
        Returns:
            (ret_left, frame_left, ret_right, frame_right)
        """
        ret_left, frame_left = self.cap_left.read()
        ret_right, frame_right = self.cap_right.read()
        return ret_left, frame_left, ret_right, frame_right

    def detect(self):
        """
        Capture frames and detect ball in both cameras.

        Returns:
            dict with:
                - 'left_frame': Left camera image
                - 'right_frame': Right camera image
                - 'left_detection': Detection result from left camera
                - 'right_detection': Detection result from right camera
                - 'both_found': True if ball found in BOTH cameras
        """
        result = {
            'left_frame': None,
            'right_frame': None,
            'left_detection': None,
            'right_detection': None,
            'both_found': False
        }

        # Capture frames
        ret_left, frame_left, ret_right, frame_right = self.read_frames()

        if not ret_left or not ret_right:
            return result

        result['left_frame'] = frame_left
        result['right_frame'] = frame_right

        # Detect ball in each camera
        result['left_detection'] = self.tracker_left.detect(frame_left)
        result['right_detection'] = self.tracker_right.detect(frame_right)

        # Check if found in both
        left_found = result['left_detection']['found']
        right_found = result['right_detection']['found']
        result['both_found'] = left_found and right_found

        return result

    def detect_from_frames(self, frame_left, frame_right, return_debug=False):
        """
        Detect ball from provided frames.

        Args:
            frame_left: Image from left camera
            frame_right: Image from right camera
            return_debug: If True, return debug masks

        Returns:
            Same dict as detect()
        """
        result = {
            'left_frame': frame_left,
            'right_frame': frame_right,
            'left_detection': self.tracker_left.detect(frame_left, return_debug=return_debug),
            'right_detection': self.tracker_right.detect(frame_right, return_debug=return_debug),
            'both_found': False
        }

        result['both_found'] = (result['left_detection']['found'] and 
                                result['right_detection']['found'])

        return result

    def draw_detections(self, result):
        """
        Draw ball detections on frames.

        Args:
            result: Result dict from detect()

        Returns:
            (annotated_left, annotated_right) frames
        """
        left_out = result['left_frame'].copy()
        right_out = result['right_frame'].copy()

        # Draw left detection
        if result['left_detection'] and result['left_detection']['found']:
            det = result['left_detection']
            center = det['center']
            radius = int(det['radius'])
            cv2.circle(left_out, center, radius, (0, 255, 0), 2)
            cv2.circle(left_out, center, 3, (0, 0, 255), -1)
            cv2.putText(left_out, f"L: ({center[0]}, {center[1]})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(left_out, "LEFT: No ball", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw right detection
        if result['right_detection'] and result['right_detection']['found']:
            det = result['right_detection']
            center = det['center']
            radius = int(det['radius'])
            cv2.circle(right_out, center, radius, (0, 255, 0), 2)
            cv2.circle(right_out, center, 3, (0, 0, 255), -1)
            cv2.putText(right_out, f"R: ({center[0]}, {center[1]})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(right_out, "RIGHT: No ball", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return left_out, right_out

    def set_hsv_thresholds(self, lower, upper):
        """Update HSV thresholds for both trackers."""
        self.tracker_left.set_hsv_thresholds(lower, upper)
        self.tracker_right.set_hsv_thresholds(lower, upper)

    def set_lab_thresholds(self, lower, upper):
        """Update LAB thresholds for both trackers."""
        self.tracker_left.set_lab_thresholds(lower, upper)
        self.tracker_right.set_lab_thresholds(lower, upper)

    def set_hsv_thresholds_left(self, lower, upper):
        """Update HSV thresholds for left tracker only."""
        self.tracker_left.set_hsv_thresholds(lower, upper)

    def set_hsv_thresholds_right(self, lower, upper):
        """Update HSV thresholds for right tracker only."""
        self.tracker_right.set_hsv_thresholds(lower, upper)

    def set_lab_thresholds_left(self, lower, upper):
        """Update LAB thresholds for left tracker only."""
        self.tracker_left.set_lab_thresholds(lower, upper)

    def set_lab_thresholds_right(self, lower, upper):
        """Update LAB thresholds for right tracker only."""
        self.tracker_right.set_lab_thresholds(lower, upper)