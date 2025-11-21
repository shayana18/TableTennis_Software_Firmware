"""
Stereo Ball Tracker

Detects ball in two cameras simultaneously and calculates 3D position.
Uses existing BallTracker for detection + triangulation for 3D.
"""

import cv2
import numpy as np
import yaml
from pathlib import Path

from .ball_tracker import BallTracker
from .triangulation import (
    triangulate_point,
    compute_disparity,
    validate_stereo_match,
    undistort_point
)


class StereoTracker:
    """Track ball in stereo camera setup and compute 3D position."""

    def __init__(self, config_path):
        """
        Initialize stereo tracker from config file.

        Args:
            config_path: Path to stereo_config.yaml
        """
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Get base directory (where config file is)
        config_dir = Path(config_path).parent.parent

        # Camera IDs
        self.cam_left_id = config['camera_left']['id']
        self.cam_right_id = config['camera_right']['id']

        # Load intrinsics from calibration directories
        left_dir = config_dir / config['camera_left']['calibration_dir']
        right_dir = config_dir / config['camera_right']['calibration_dir']

        print("\n" + "=" * 50)
        print("LOADING STEREO CALIBRATION")
        print("=" * 50)

        # Load left camera intrinsics
        print(f"\nLeft camera ({left_dir}):")
        self.K_left = np.loadtxt(left_dir / "camera_matrix.txt").reshape(3, 3)
        self.dist_left = np.loadtxt(left_dir / "distortion_coefficients.txt")
        print(f"  Camera matrix:\n{self.K_left}")
        print(f"  Distortion: {self.dist_left}")

        # Load right camera intrinsics
        print(f"\nRight camera ({right_dir}):")
        self.K_right = np.loadtxt(right_dir / "camera_matrix.txt").reshape(3, 3)
        self.dist_right = np.loadtxt(right_dir / "distortion_coefficients.txt")
        print(f"  Camera matrix:\n{self.K_right}")
        print(f"  Distortion: {self.dist_right}")

        # Load extrinsics from stereo calibration directory
        stereo_dir = config_dir / config['stereo']['calibration_dir']
        if stereo_dir.exists() and (stereo_dir / "R.txt").exists():
            self.R = np.loadtxt(stereo_dir / "R.txt").reshape(3, 3)
            self.T = np.loadtxt(stereo_dir / "T.txt")
            self.baseline_mm = float(np.loadtxt(stereo_dir / "baseline.txt"))
            self.is_calibrated = True
            print(f"\nStereo extrinsics ({stereo_dir}):")
            print(f"  R:\n{self.R}")
            print(f"  T: {self.T}")
            print(f"  Baseline: {self.baseline_mm:.2f} mm")
        else:
            # Placeholders until calibrated
            self.R = np.eye(3)
            self.T = np.array([200.0, 0.0, 0.0])
            self.baseline_mm = 200.0
            self.is_calibrated = False
            print(f"\nStereo extrinsics: NOT CALIBRATED (using placeholders)")
            print(f"  Run calibrate_stereo_extrinsics.py first")

        print("=" * 50 + "\n")

        # Get focal length for depth estimation
        self.focal_length = (self.K_left[0, 0] + self.K_left[1, 1]) / 2

        # Initialize ball trackers for each camera
        self.tracker_left = BallTracker()
        self.tracker_right = BallTracker()

        # Video captures (initialized when needed)
        self.cap_left = None
        self.cap_right = None

        # State
        self.position_3d = None
        self.position_history_3d = []
        self.max_history = 50

    def start_cameras(self):
        """Open camera streams. Call before using update()."""
        self.cap_left = cv2.VideoCapture(self.cam_left_id)
        self.cap_right = cv2.VideoCapture(self.cam_right_id)

        # Check if cameras opened
        if not self.cap_left.isOpened():
            raise RuntimeError(f"Failed to open left camera (ID: {self.cam_left_id})")
        if not self.cap_right.isOpened():
            raise RuntimeError(f"Failed to open right camera (ID: {self.cam_right_id})")

        return True

    def stop_cameras(self):
        """Release camera streams."""
        if self.cap_left:
            self.cap_left.release()
        if self.cap_right:
            self.cap_right.release()

    def update(self):
        """
        Capture frames, detect ball, and calculate 3D position.

        Returns:
            dict with:
                - 'found': True if ball found in both cameras
                - 'left_frame': Left camera image
                - 'right_frame': Right camera image
                - 'left_detection': Detection result from left camera
                - 'right_detection': Detection result from right camera
                - 'position_3d': (X, Y, Z) in mm or None
                - 'disparity': Pixel difference or None
                - 'matched': True if detections are valid stereo pair
        """
        result = {
            'found': False,
            'left_frame': None,
            'right_frame': None,
            'left_detection': None,
            'right_detection': None,
            'position_3d': None,
            'disparity': None,
            'matched': False
        }

        # Capture frames
        ret_left, frame_left = self.cap_left.read()
        ret_right, frame_right = self.cap_right.read()

        if not ret_left or not ret_right:
            return result

        result['left_frame'] = frame_left
        result['right_frame'] = frame_right

        # Detect ball in each camera
        det_left = self.tracker_left.detect(frame_left)
        det_right = self.tracker_right.detect(frame_right)

        result['left_detection'] = det_left
        result['right_detection'] = det_right

        # Check if ball found in both cameras
        if not det_left['found'] or not det_right['found']:
            return result

        # Get 2D positions
        pt_left = det_left['center']
        pt_right = det_right['center']

        # Validate match (similar Y coordinate)
        if not validate_stereo_match(pt_left, pt_right, max_y_diff=50):
            return result

        result['matched'] = True

        # Calculate disparity
        disparity = compute_disparity(pt_left, pt_right)
        result['disparity'] = disparity

        # Triangulate to get 3D position
        # Undistort points first for accuracy
        pt_left_undist = undistort_point(pt_left, self.K_left, self.dist_left)
        pt_right_undist = undistort_point(pt_right, self.K_right, self.dist_right)

        position_3d = triangulate_point(
            pt_left_undist, pt_right_undist,
            self.K_left, self.K_right,
            self.R, self.T
        )

        if position_3d is not None:
            result['found'] = True
            result['position_3d'] = position_3d
            self.position_3d = position_3d

            # Update history
            self.position_history_3d.append(position_3d)
            if len(self.position_history_3d) > self.max_history:
                self.position_history_3d.pop(0)

        return result

    def detect_from_frames(self, frame_left, frame_right):
        """
        Detect ball and triangulate from provided frames.
        Use this if you're capturing frames yourself.

        Args:
            frame_left: Image from left camera
            frame_right: Image from right camera

        Returns:
            Same dict as update()
        """
        result = {
            'found': False,
            'left_frame': frame_left,
            'right_frame': frame_right,
            'left_detection': None,
            'right_detection': None,
            'position_3d': None,
            'disparity': None,
            'matched': False
        }

        # Detect ball in each camera
        det_left = self.tracker_left.detect(frame_left)
        det_right = self.tracker_right.detect(frame_right)

        result['left_detection'] = det_left
        result['right_detection'] = det_right

        # Check if ball found in both cameras
        if not det_left['found'] or not det_right['found']:
            return result

        # Get 2D positions
        pt_left = det_left['center']
        pt_right = det_right['center']

        # Validate match
        if not validate_stereo_match(pt_left, pt_right, max_y_diff=50):
            return result

        result['matched'] = True
        result['disparity'] = compute_disparity(pt_left, pt_right)

        # Triangulate
        pt_left_undist = undistort_point(pt_left, self.K_left, self.dist_left)
        pt_right_undist = undistort_point(pt_right, self.K_right, self.dist_right)

        position_3d = triangulate_point(
            pt_left_undist, pt_right_undist,
            self.K_left, self.K_right,
            self.R, self.T
        )

        if position_3d is not None:
            result['found'] = True
            result['position_3d'] = position_3d
            self.position_3d = position_3d

            # Update history
            self.position_history_3d.append(position_3d)
            if len(self.position_history_3d) > self.max_history:
                self.position_history_3d.pop(0)

        return result

    def get_velocity_3d(self):
        """
        Estimate 3D velocity from position history.

        Returns:
            (Vx, Vy, Vz) in mm/frame, or None if not enough history
        """
        if len(self.position_history_3d) < 3:
            return None

        # Use last 5 positions for estimate
        recent = self.position_history_3d[-5:]
        n = len(recent)
        t = np.arange(n)

        x = [p[0] for p in recent]
        y = [p[1] for p in recent]
        z = [p[2] for p in recent]

        try:
            vx = np.polyfit(t, x, 1)[0]
            vy = np.polyfit(t, y, 1)[0]
            vz = np.polyfit(t, z, 1)[0]
            return (vx, vy, vz)
        except:
            return None

    def reset(self):
        """Clear position history."""
        self.position_3d = None
        self.position_history_3d = []

    def draw_detections(self, frame_left, frame_right, result):
        """
        Draw ball detections and 3D info on frames.

        Args:
            frame_left: Left camera frame to draw on
            frame_right: Right camera frame to draw on
            result: Result dict from update()

        Returns:
            (annotated_left, annotated_right)
        """
        left_out = frame_left.copy()
        right_out = frame_right.copy()

        # Draw left detection
        if result['left_detection'] and result['left_detection']['found']:
            det = result['left_detection']
            cv2.circle(left_out, det['center'], int(det['radius']), (0, 255, 0), 2)
            cv2.circle(left_out, det['center'], 3, (0, 0, 255), -1)

        # Draw right detection
        if result['right_detection'] and result['right_detection']['found']:
            det = result['right_detection']
            cv2.circle(right_out, det['center'], int(det['radius']), (0, 255, 0), 2)
            cv2.circle(right_out, det['center'], 3, (0, 0, 255), -1)

        # Draw 3D position info
        if result['found'] and result['position_3d']:
            X, Y, Z = result['position_3d']
            text = f"3D: ({X:.0f}, {Y:.0f}, {Z:.0f}) mm"
            cv2.putText(left_out, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if result['disparity']:
                disp_text = f"Disparity: {result['disparity']:.1f} px"
                cv2.putText(left_out, disp_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            status = "Matched" if result['matched'] else "Not matched"
            if not result['left_detection'] or not result['left_detection']['found']:
                status = "Left: No ball"
            elif not result['right_detection'] or not result['right_detection']['found']:
                status = "Right: No ball"
            cv2.putText(left_out, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return left_out, right_out
