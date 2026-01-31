"""
Stereo Triangulator

Combines stereo detection with 3D triangulation.
Loads camera calibration and computes 3D ball position.

CAMERA: Arducam OV9782 Global Shutter USB Camera
        1MP, 100fps @ 1280x800 MJPG

UNITS: All outputs are in the same units as checkerboard_box_size_scale
       (typically mm if you used mm during calibration)
"""

import cv2
import numpy as np
from pathlib import Path
from .ball_tracker import EnhancedBallTracker


def configure_camera_for_arducam(cap, width=1280, height=800):
    """
    Configure camera for Arducam OV9782 global shutter cameras.
    Forces MJPG codec and specified resolution.
    """
    fourcc_mjpg = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc_mjpg)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 100)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    actual_fourcc_str = "".join([chr((actual_fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
    
    return {
        'actual_width': actual_width,
        'actual_height': actual_height,
        'actual_fps': actual_fps,
        'actual_fourcc': actual_fourcc_str,
        'settings_match': (actual_width == width and actual_height == height)
    }


class StereoTriangulator:
    """
    Stereo vision triangulation for 3D ball tracking.
    
    Loads camera calibration (intrinsic + extrinsic) and
    triangulates ball position from stereo detections.
    """

    def __init__(self, calibration_dir, cam_left_id=0, cam_right_id=1):
        """
        Initialize triangulator.

        Args:
            calibration_dir: Path to folder with calibration .dat files
            cam_left_id: Left camera device ID
            cam_right_id: Right camera device ID
        """
        self.calibration_dir = Path(calibration_dir)
        self.cam_left_id = cam_left_id
        self.cam_right_id = cam_right_id

        self.cap_left = None
        self.cap_right = None

        self.tracker_left = EnhancedBallTracker()
        self.tracker_right = EnhancedBallTracker()

        self.cmtx0 = None
        self.dist0 = None
        self.cmtx1 = None
        self.dist1 = None

        self.R0 = None
        self.T0 = None
        self.R1 = None
        self.T1 = None

        self.P0 = None
        self.P1 = None

        self._load_calibration()

    def _load_intrinsics(self, filepath):
        """Load camera intrinsic parameters from .dat file."""
        cmtx = []
        dist = []
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        reading_intrinsic = False
        reading_distortion = False
        
        for line in lines:
            line = line.strip()
            if line == 'intrinsic:':
                reading_intrinsic = True
                reading_distortion = False
                continue
            elif line == 'distortion:':
                reading_intrinsic = False
                reading_distortion = True
                continue
            elif line.startswith('reprojection error:'):
                break
            
            if reading_intrinsic and line:
                cmtx.append([float(x) for x in line.split()])
            elif reading_distortion and line:
                dist = [float(x) for x in line.split()]
        
        return np.array(cmtx, dtype=np.float64), np.array(dist, dtype=np.float64)

    def _load_extrinsics(self, filepath):
        """Load camera extrinsic parameters from .dat file."""
        R = []
        T = []
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        reading_R = False
        reading_T = False
        
        for line in lines:
            line = line.strip()
            if line == 'R:':
                reading_R = True
                reading_T = False
                continue
            elif line == 'T:':
                reading_R = False
                reading_T = True
                continue
            
            if reading_R and line:
                R.append([float(x) for x in line.split()])
            elif reading_T and line:
                T.append([float(x) for x in line.split()])
        
        return np.array(R, dtype=np.float64), np.array(T, dtype=np.float64).reshape(3, 1)

    def _load_calibration(self):
        """Load all calibration files."""
        self.cmtx0, self.dist0 = self._load_intrinsics(
            self.calibration_dir / 'camera0_intrinsics.dat')
        self.cmtx1, self.dist1 = self._load_intrinsics(
            self.calibration_dir / 'camera1_intrinsics.dat')
        
        self.R0, self.T0 = self._load_extrinsics(
            self.calibration_dir / 'camera0_rot_trans.dat')
        self.R1, self.T1 = self._load_extrinsics(
            self.calibration_dir / 'camera1_rot_trans.dat')
        
        self.P0 = self._get_projection_matrix(self.cmtx0, self.R0, self.T0)
        self.P1 = self._get_projection_matrix(self.cmtx1, self.R1, self.T1)
        
        baseline = np.linalg.norm(self.T1 - self.T0)
        print(f"[StereoTriangulator] Calibration loaded")
        print(f"  Baseline: {baseline:.2f} (same units as checkerboard_box_size_scale)")

    def _get_projection_matrix(self, cmtx, R, T):
        """Compute projection matrix P = K @ [R|T]."""
        RT = np.zeros((3, 4), dtype=np.float64)
        RT[:3, :3] = R
        RT[:3, 3] = T.flatten()
        return cmtx @ RT

    def load_thresholds(self, filepath):
        """Load ball detection thresholds from JSON file."""
        import json
        try:
            with open(filepath, 'r') as f:
                thresholds = json.load(f)
            
            if 'left' in thresholds and 'right' in thresholds:
                self.tracker_left.set_hsv_thresholds(
                    thresholds['left']['hsv_lower'], thresholds['left']['hsv_upper'])
                self.tracker_left.set_lab_thresholds(
                    thresholds['left']['lab_lower'], thresholds['left']['lab_upper'])
                
                self.tracker_right.set_hsv_thresholds(
                    thresholds['right']['hsv_lower'], thresholds['right']['hsv_upper'])
                self.tracker_right.set_lab_thresholds(
                    thresholds['right']['lab_lower'], thresholds['right']['lab_upper'])
                
                print(f"[StereoTriangulator] Loaded STEREO thresholds from {filepath}")
            else:
                if 'hsv_lower' in thresholds:
                    self.tracker_left.set_hsv_thresholds(
                        thresholds['hsv_lower'], thresholds['hsv_upper'])
                    self.tracker_right.set_hsv_thresholds(
                        thresholds['hsv_lower'], thresholds['hsv_upper'])
                
                if 'lab_lower' in thresholds:
                    self.tracker_left.set_lab_thresholds(
                        thresholds['lab_lower'], thresholds['lab_upper'])
                    self.tracker_right.set_lab_thresholds(
                        thresholds['lab_lower'], thresholds['lab_upper'])
                
                print(f"[StereoTriangulator] Loaded thresholds from {filepath}")
        except Exception as e:
            print(f"[StereoTriangulator] Warning: Could not load thresholds: {e}")

    def start_cameras(self, width=1280, height=800):
        """Open camera streams with Arducam MJPG configuration."""
        self.cap_left = cv2.VideoCapture(self.cam_left_id)
        self.cap_right = cv2.VideoCapture(self.cam_right_id)

        if not self.cap_left.isOpened():
            raise RuntimeError(f"Failed to open left camera (ID: {self.cam_left_id})")
        if not self.cap_right.isOpened():
            raise RuntimeError(f"Failed to open right camera (ID: {self.cam_right_id})")

        # Configure for Arducam OV9782 with MJPG
        settings_left = configure_camera_for_arducam(self.cap_left, width, height)
        settings_right = configure_camera_for_arducam(self.cap_right, width, height)
        
        print(f"[StereoTriangulator] Cameras started (MJPG mode):")
        print(f"  LEFT:  {settings_left['actual_width']}x{settings_left['actual_height']} "
              f"@ {settings_left['actual_fps']:.0f}fps ({settings_left['actual_fourcc']})")
        print(f"  RIGHT: {settings_right['actual_width']}x{settings_right['actual_height']} "
              f"@ {settings_right['actual_fps']:.0f}fps ({settings_right['actual_fourcc']})")

    def stop_cameras(self):
        """Release camera streams."""
        if self.cap_left:
            self.cap_left.release()
        if self.cap_right:
            self.cap_right.release()

    def triangulate(self, point_left, point_right):
        """
        Triangulate 3D point from stereo correspondences using DLT.

        Args:
            point_left: (x, y) pixel coordinates in left image
            point_right: (x, y) pixel coordinates in right image

        Returns:
            (X, Y, Z) 3D coordinates (same units as calibration)
        """
        A = np.array([
            point_left[1] * self.P0[2, :] - self.P0[1, :],
            self.P0[0, :] - point_left[0] * self.P0[2, :],
            point_right[1] * self.P1[2, :] - self.P1[1, :],
            self.P1[0, :] - point_right[0] * self.P1[2, :]
        ])

        _, _, Vh = np.linalg.svd(A)
        X = Vh[-1]
        X = X[:3] / X[3]

        return X

    def update(self):
        """
        Capture frames, detect ball, and triangulate.

        Returns:
            dict with detection and triangulation results
        """
        result = {
            'left_frame': None,
            'right_frame': None,
            'left_detection': None,
            'right_detection': None,
            'found_3d': False,
            'position_3d': None,
            'disparity': None
        }

        ret_left, frame_left = self.cap_left.read()
        ret_right, frame_right = self.cap_right.read()

        if not ret_left or not ret_right:
            return result

        result['left_frame'] = frame_left
        result['right_frame'] = frame_right

        result['left_detection'] = self.tracker_left.detect(frame_left)
        result['right_detection'] = self.tracker_right.detect(frame_right)

        left_found = result['left_detection']['found']
        right_found = result['right_detection']['found']

        if left_found and right_found:
            point_left = result['left_detection']['center']
            point_right = result['right_detection']['center']

            disparity = point_left[0] - point_right[0]
            result['disparity'] = disparity

            if disparity > 0:
                X, Y, Z = self.triangulate(point_left, point_right)
                if Z > 0:
                    result['found_3d'] = True
                    result['position_3d'] = (X, Y, Z)

        return result

    def draw_results(self, result):
        """Draw detection and triangulation results on frames."""
        left_vis = result['left_frame'].copy()
        right_vis = result['right_frame'].copy()

        if result['left_detection'] and result['left_detection']['found']:
            det = result['left_detection']
            center = det['center']
            radius = int(det['radius'])
            cv2.circle(left_vis, center, radius, (0, 255, 0), 2)
            cv2.circle(left_vis, center, 3, (0, 0, 255), -1)

        if result['right_detection'] and result['right_detection']['found']:
            det = result['right_detection']
            center = det['center']
            radius = int(det['radius'])
            cv2.circle(right_vis, center, radius, (0, 255, 0), 2)
            cv2.circle(right_vis, center, 3, (0, 0, 255), -1)

        if result['found_3d']:
            X, Y, Z = result['position_3d']
            cv2.putText(left_vis, f"3D: X={X:.1f} Y={Y:.1f} Z={Z:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(left_vis, f"Disparity: {result['disparity']:.1f}px", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            if result['left_detection']['found'] and result['right_detection']['found']:
                cv2.putText(left_vis, "Both detected but triangulation failed", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            else:
                cv2.putText(left_vis, "Need detection in both cameras", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return left_vis, right_vis

    def set_hsv_thresholds(self, lower, upper):
        """Update HSV thresholds for both trackers."""
        self.tracker_left.set_hsv_thresholds(lower, upper)
        self.tracker_right.set_hsv_thresholds(lower, upper)

    def set_lab_thresholds(self, lower, upper):
        """Update LAB thresholds for both trackers."""
        self.tracker_left.set_lab_thresholds(lower, upper)
        self.tracker_right.set_lab_thresholds(lower, upper)

    def get_hsv_thresholds(self):
        """Get current HSV thresholds."""
        return (
            self.tracker_left.hsv_lower.tolist(),
            self.tracker_left.hsv_upper.tolist()
        )

    def get_lab_thresholds(self):
        """Get current LAB thresholds."""
        return (
            self.tracker_left.lab_lower.tolist(),
            self.tracker_left.lab_upper.tolist()
        )

    def get_baseline(self):
        """Get stereo baseline (same units as calibration)."""
        return np.linalg.norm(self.T1 - self.T0)

    def get_focal_length_px(self):
        """Get average focal length in pixels."""
        fx0 = self.cmtx0[0, 0]
        fx1 = self.cmtx1[0, 0]
        return (fx0 + fx1) / 2