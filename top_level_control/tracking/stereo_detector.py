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
import yaml
from pathlib import Path
from .ball_tracker import EnhancedBallTracker


def configure_camera_for_arducam(cap, width=1280, height=720):
    """
    Configure camera for Arducam OV9782 global shutter cameras.
    Forces MJPG codec and specified resolution.
    
    Args:
        cap: cv2.VideoCapture object
        width: Desired width (default 1280)
        height: Desired height (default 800)
    
    Returns:
        dict with actual accepted values
    """
    # Set FOURCC to MJPG first - critical for getting full framerate on Arducam
    fourcc_mjpg = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc_mjpg)
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Try to set higher framerate (Arducam OV9782 supports 100fps at 1280x800 MJPG)
    cap.set(cv2.CAP_PROP_FPS, 100)
    
    # Disable auto-exposure for consistent detection
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    
    # Read back actual values
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    
    # Decode FOURCC integer to string
    actual_fourcc_str = "".join([chr((actual_fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
    
    return {
        'requested_width': width,
        'requested_height': height,
        'requested_fourcc': 'MJPG',
        'actual_width': actual_width,
        'actual_height': actual_height,
        'actual_fps': actual_fps,
        'actual_fourcc': actual_fourcc_str,
        'settings_match': (actual_width == width and actual_height == height)
    }


class StereoDetector:
    """Detect ball in stereo camera pair. No 3D calculations."""

    def __init__(self, cam_left_id=None, cam_right_id=None, thresholds_file=None, config_path=None):
        """
        Initialize stereo detector.

        Args:
            cam_left_id: Device ID for left camera
            cam_right_id: Device ID for right camera
            thresholds_file: Optional path to ball_thresholds.json
            config_path: Optional path to stereo_config.yaml
        """
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config' / 'stereo_config.yaml'
        
        self.config = {}
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Get camera IDs from config
            cam_left_id = cam_left_id if cam_left_id is not None else self.config.get('camera_left', {}).get('id', 1)
            cam_right_id = cam_right_id if cam_right_id is not None else self.config.get('camera_right', {}).get('id', 2)
        else:
            cam_left_id = cam_left_id if cam_left_id is not None else 1
            cam_right_id = cam_right_id if cam_right_id is not None else 2

        self.cam_left_id = cam_left_id
        self.cam_right_id = cam_right_id

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

    def start_cameras(self, width=1280, height=720):
        """
        Open camera streams with Arducam MJPG configuration.
        
        Args:
            width: Frame width (default 1280 for OV9782)
            height: Frame height (default 800 for OV9782)
        """
        self.cap_left = cv2.VideoCapture(self.cam_left_id)
        self.cap_right = cv2.VideoCapture(self.cam_right_id)

        if not self.cap_left.isOpened():
            raise RuntimeError(f"Failed to open left camera (ID: {self.cam_left_id})")
        if not self.cap_right.isOpened():
            raise RuntimeError(f"Failed to open right camera (ID: {self.cam_right_id})")

        # Configure for Arducam OV9782 with MJPG
        print("\n[StereoDetector] Configuring cameras (Arducam OV9782 MJPG mode):")
        settings_left = configure_camera_for_arducam(self.cap_left, width, height)
        settings_right = configure_camera_for_arducam(self.cap_right, width, height)
        
        print(f"  LEFT:  {settings_left['actual_width']}x{settings_left['actual_height']} "
              f"@ {settings_left['actual_fps']:.0f}fps ({settings_left['actual_fourcc']})")
        print(f"  RIGHT: {settings_right['actual_width']}x{settings_right['actual_height']} "
              f"@ {settings_right['actual_fps']:.0f}fps ({settings_right['actual_fourcc']})")
        
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