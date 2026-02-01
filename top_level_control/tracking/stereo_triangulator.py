"""
Stereo Triangulator for Basler Cameras

Combines stereo detection with 3D triangulation.
Loads camera calibration and computes 3D ball position.

CAMERA: Basler acA1920-150uc USB 3.0
        2.3MP, 150fps @ 1920x1200

REQUIREMENTS:
    pip install pypylon

UNITS: All outputs are in the same units as checkerboard_box_size_scale
       (typically cm if you used cm, or mm if you used mm)
"""

import cv2
import numpy as np
import json
from pathlib import Path

try:
    from pypylon import pylon
    PYPYLON_AVAILABLE = True
except ImportError:
    PYPYLON_AVAILABLE = False

from .ball_tracker import EnhancedBallTracker


class StereoTriangulator:
    """
    Stereo vision triangulation for 3D ball tracking.
    
    Loads camera calibration (intrinsic + extrinsic) and
    triangulates ball position from stereo detections.
    """

    def __init__(self, calibration_dir, left_serial=None, right_serial=None,
                 cam_left_id=0, cam_right_id=1):
        """
        Initialize triangulator.

        Args:
            calibration_dir: Path to folder with calibration .dat files
            left_serial: Left camera serial number (preferred for Basler)
            right_serial: Right camera serial number (preferred for Basler)
            cam_left_id: Left camera device index (fallback)
            cam_right_id: Right camera device index (fallback)
        """
        if not PYPYLON_AVAILABLE:
            raise RuntimeError("pypylon not installed. Run: pip install pypylon")

        self.calibration_dir = Path(calibration_dir)
        self.left_serial = left_serial
        self.right_serial = right_serial
        self.cam_left_id = cam_left_id
        self.cam_right_id = cam_right_id

        # Basler cameras
        self.cam_left = None
        self.cam_right = None
        self.converter = None

        # Ball trackers (independent thresholds per camera)
        self.tracker_left = EnhancedBallTracker()
        self.tracker_right = EnhancedBallTracker()

        # Camera matrices and distortion
        self.cmtx0 = None  # Left camera intrinsic
        self.dist0 = None  # Left camera distortion
        self.cmtx1 = None  # Right camera intrinsic
        self.dist1 = None  # Right camera distortion

        # Extrinsic parameters
        self.R0 = None  # Left camera rotation
        self.T0 = None  # Left camera translation
        self.R1 = None  # Right camera rotation
        self.T1 = None  # Right camera translation

        # Projection matrices
        self.P0 = None  # Left projection matrix
        self.P1 = None  # Right projection matrix

        # Load calibration
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
        # Load intrinsics
        self.cmtx0, self.dist0 = self._load_intrinsics(
            self.calibration_dir / 'camera0_intrinsics.dat')
        self.cmtx1, self.dist1 = self._load_intrinsics(
            self.calibration_dir / 'camera1_intrinsics.dat')
        
        # Load extrinsics
        self.R0, self.T0 = self._load_extrinsics(
            self.calibration_dir / 'camera0_rot_trans.dat')
        self.R1, self.T1 = self._load_extrinsics(
            self.calibration_dir / 'camera1_rot_trans.dat')
        
        # Compute projection matrices
        self.P0 = self._get_projection_matrix(self.cmtx0, self.R0, self.T0)
        self.P1 = self._get_projection_matrix(self.cmtx1, self.R1, self.T1)
        
        # Calculate baseline
        baseline = np.linalg.norm(self.T1 - self.T0)
        print(f"[StereoTriangulator] Calibration loaded")
        print(f"  Baseline: {baseline:.2f} (same units as checkerboard_box_size_scale)")

    def _get_projection_matrix(self, cmtx, R, T):
        """Compute projection matrix P = K @ [R|T]."""
        RT = np.zeros((3, 4), dtype=np.float64)
        RT[:3, :3] = R
        RT[:3, 3] = T.flatten()
        return cmtx @ RT

    def _get_camera(self, serial=None, device_index=0):
        """Get Basler camera by serial or index."""
        tlFactory = pylon.TlFactory.GetInstance()
        devices = tlFactory.EnumerateDevices()

        if len(devices) == 0:
            raise RuntimeError("No Basler cameras found")

        if serial and serial.strip():
            for device in devices:
                if device.GetSerialNumber() == serial:
                    return pylon.InstantCamera(tlFactory.CreateDevice(device))
            raise RuntimeError(f"Camera with serial '{serial}' not found")

        if device_index >= len(devices):
            raise RuntimeError(f"Device index {device_index} out of range")

        return pylon.InstantCamera(tlFactory.CreateDevice(devices[device_index]))

    def _configure_camera(self, camera, width, height):
        """Configure Basler camera."""
        camera.Open()

        camera.Width.SetValue(width)
        camera.Height.SetValue(height)

        max_w = camera.WidthMax.GetValue()
        max_h = camera.HeightMax.GetValue()
        camera.OffsetX.SetValue((max_w - width) // 2)
        camera.OffsetY.SetValue((max_h - height) // 2)

        camera.ExposureAuto.SetValue("Off")
        camera.ExposureTime.SetValue(3000)
        camera.GainAuto.SetValue("Off")
        camera.Gain.SetValue(0)

    def _grab_frame(self, camera):
        """Grab single frame from Basler camera."""
        if not camera.IsGrabbing():
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grab_result.GrabSucceeded():
            image = self.converter.Convert(grab_result)
            frame = image.GetArray()
            grab_result.Release()
            return frame

        grab_result.Release()
        return None

    def load_thresholds(self, filepath):
        """Load ball detection thresholds from JSON file."""
        try:
            with open(filepath, 'r') as f:
                thresholds = json.load(f)
            
            # Check for stereo format (per-camera thresholds)
            if 'left' in thresholds and 'right' in thresholds:
                # Stereo format
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
                # Single format (same for both cameras)
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

    def start_cameras(self, width=1920, height=1200):
        """Open and configure Basler cameras."""
        print("[StereoTriangulator] Opening cameras...")

        self.cam_left = self._get_camera(
            serial=self.left_serial if self.left_serial else None,
            device_index=self.cam_left_id
        )
        self.cam_right = self._get_camera(
            serial=self.right_serial if self.right_serial else None,
            device_index=self.cam_right_id
        )

        self._configure_camera(self.cam_left, width, height)
        self._configure_camera(self.cam_right, width, height)

        # Image converter
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        print(f"  Left:  {self.cam_left.GetDeviceInfo().GetSerialNumber()}")
        print(f"  Right: {self.cam_right.GetDeviceInfo().GetSerialNumber()}")
        print("[StereoTriangulator] Cameras ready")

    def stop_cameras(self):
        """Close camera streams."""
        if self.cam_left:
            self.cam_left.Close()
        if self.cam_right:
            self.cam_right.Close()
        print("[StereoTriangulator] Cameras stopped")

    def triangulate(self, point_left, point_right):
        """
        Triangulate 3D point from stereo correspondences using DLT.

        Args:
            point_left: (x, y) pixel coordinates in left image
            point_right: (x, y) pixel coordinates in right image

        Returns:
            (X, Y, Z) 3D coordinates (same units as calibration)
        """
        # DLT triangulation
        A = np.array([
            point_left[1] * self.P0[2, :] - self.P0[1, :],
            self.P0[0, :] - point_left[0] * self.P0[2, :],
            point_right[1] * self.P1[2, :] - self.P1[1, :],
            self.P1[0, :] - point_right[0] * self.P1[2, :]
        ])

        _, _, Vh = np.linalg.svd(A)
        X = Vh[-1]
        X = X[:3] / X[3]  # Convert from homogeneous

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

        # Capture frames from Basler cameras
        frame_left = self._grab_frame(self.cam_left)
        frame_right = self._grab_frame(self.cam_right)

        if frame_left is None or frame_right is None:
            return result

        result['left_frame'] = frame_left
        result['right_frame'] = frame_right

        # Detect ball in each camera
        result['left_detection'] = self.tracker_left.detect(frame_left)
        result['right_detection'] = self.tracker_right.detect(frame_right)

        # Triangulate if both cameras detect ball
        left_found = result['left_detection']['found']
        right_found = result['right_detection']['found']

        if left_found and right_found:
            point_left = result['left_detection']['center']
            point_right = result['right_detection']['center']

            # Calculate disparity
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

        # Draw left detection
        if result['left_detection'] and result['left_detection']['found']:
            det = result['left_detection']
            center = det['center']
            radius = int(det['radius'])
            cv2.circle(left_vis, center, radius, (0, 255, 0), 2)
            cv2.circle(left_vis, center, 3, (0, 0, 255), -1)

        # Draw right detection
        if result['right_detection'] and result['right_detection']['found']:
            det = result['right_detection']
            center = det['center']
            radius = int(det['radius'])
            cv2.circle(right_vis, center, radius, (0, 255, 0), 2)
            cv2.circle(right_vis, center, 3, (0, 0, 255), -1)

        # Draw 3D position
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