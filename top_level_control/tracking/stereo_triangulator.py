"""
Stereo Triangulator

Combines stereo detection with 3D triangulation.
Loads camera calibration and computes 3D ball position.

CAMERA: Arducam OV9782 Global Shutter USB Camera
        1MP, 100fps MJPG

DETECTION: MOG2 background subtraction via shared BallDetector
           (tracking/ball_detector.py — single source of truth)

STEREO RECTIFICATION:
  cv2.stereoRectify is applied at calibration load time to make both
  cameras virtually parallel. All pixel coordinates used for triangulation
  are undistorted + rectified, ensuring:
    - Epipolar lines are horizontal (y-coords match between L/R)
    - Disparity is purely in x and always positive for valid depths
    - Standard formula Z = f*B/d applies

UNITS: All 3D outputs in same units as checkerboard_box_size_scale (cm),
       in the original camera0 coordinate frame.

ROBUSTNESS FILTERS (applied in rectified image space):
  1. Disparity range — reject if too small (noise) or too large (mismatch)
  2. Epipolar consistency — rectified y-coords should match within ~1-2px
  3. Z range — reject implausible depths
  4. Reprojection error — project 3D back to rectified images, reject if off

CRITICAL: Uses grab()/retrieve() pattern for synchronized stereo capture.
"""

import cv2
import numpy as np
from pathlib import Path
from .ball_detector import BallDetector
from config.camera_config import (
    configure_camera, CAMERA_LEFT_ID, CAMERA_RIGHT_ID, FRAME_WIDTH, FRAME_HEIGHT
)


class StereoTriangulator:
    """
    Stereo vision triangulation for 3D ball tracking.

    Loads camera calibration (intrinsic + extrinsic) and
    triangulates ball position from stereo detections.
    """

    # --- Triangulation sanity filters ---
    MIN_DISPARITY    = 5      # px — below this depth is unreliable
    MAX_DISPARITY    = 500    # px — above this something is wrong
    MIN_Z            = 10     # calibration units — too close is suspect
    MAX_Z            = 500    # calibration units — too far is suspect
    MAX_EPIPOLAR_ERR = 30     # px — rectification makes epipolar ~0, but detection
                              #      centroid noise (motion blur, partial masks) adds 5-25px
    MAX_REPROJ_ERR   = 8      # px — max reprojection error to accept

    WARMUP_FRAMES = 120       # ~1.5s at 80fps — frames before detection is reliable

    def __init__(self, calibration_dir, cam_left_id=None, cam_right_id=None):
        self.calibration_dir = Path(calibration_dir)
        self.cam_left_id = cam_left_id if cam_left_id is not None else CAMERA_LEFT_ID
        self.cam_right_id = cam_right_id if cam_right_id is not None else CAMERA_RIGHT_ID

        self.cap_left = None
        self.cap_right = None

        # Shared BallDetector instances (canonical detection logic)
        self.detector_left = BallDetector()
        self.detector_right = BallDetector()

        # Warmup tracking (MOG2 needs frames to learn background)
        self._frame_count = 0

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

        # Stereo rectification (computed in _load_calibration)
        self.R_rect0 = None   # rectification rotation for camera 0
        self.R_rect1 = None   # rectification rotation for camera 1
        self.P_rect0 = None   # rectified projection matrix for camera 0
        self.P_rect1 = None   # rectified projection matrix for camera 1
        self.Q = None         # disparity-to-depth mapping matrix

        self._load_calibration()

    # ================================================================
    # UNDISTORTION
    # ================================================================

    def _undistort_point(self, point, cmtx, dist):
        """Undistort a single pixel coordinate using camera calibration."""
        pt = np.array([[[point[0], point[1]]]], dtype=np.float64)
        undist = cv2.undistortPoints(pt, cmtx, dist, P=cmtx)
        return (undist[0, 0, 0], undist[0, 0, 1])

    def _rectify_point(self, point, cmtx, dist, R_rect, P_rect):
        """Undistort and rectify a single pixel coordinate.

        Applies lens undistortion then stereo rectification so that
        epipolar lines are horizontal and disparity is purely in x.
        """
        pt = np.array([[[point[0], point[1]]]], dtype=np.float64)
        rect = cv2.undistortPoints(pt, cmtx, dist, R=R_rect, P=P_rect)
        return (rect[0, 0, 0], rect[0, 0, 1])

    # ================================================================
    # CALIBRATION LOADING
    # ================================================================

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
        """Load all calibration files, build projection matrices, and compute rectification."""
        self.cmtx0, self.dist0 = self._load_intrinsics(
            self.calibration_dir / 'camera0_intrinsics.dat')
        self.cmtx1, self.dist1 = self._load_intrinsics(
            self.calibration_dir / 'camera1_intrinsics.dat')

        self.R0, self.T0 = self._load_extrinsics(
            self.calibration_dir / 'camera0_rot_trans.dat')
        self.R1, self.T1 = self._load_extrinsics(
            self.calibration_dir / 'camera1_rot_trans.dat')

        # Original (unrectified) projection matrices — kept for project_to_image
        self.P0 = self._get_projection_matrix(self.cmtx0, self.R0, self.T0)
        self.P1 = self._get_projection_matrix(self.cmtx1, self.R1, self.T1)

        # Stereo rectification — makes cameras virtually parallel
        # R1/T1 are the relative rotation/translation (camera0 is at identity)
        image_size = (FRAME_WIDTH, FRAME_HEIGHT)
        self.R_rect0, self.R_rect1, self.P_rect0, self.P_rect1, self.Q, _, _ = \
            cv2.stereoRectify(
                self.cmtx0, self.dist0, self.cmtx1, self.dist1,
                image_size, self.R1, self.T1,
                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

        baseline = np.linalg.norm(self.T1 - self.T0)
        rect_f = self.P_rect0[0, 0]
        rect_baseline = abs(self.P_rect1[0, 3]) / rect_f

        # Relative rotation magnitude (how much rectification corrects)
        angle_rad = np.arccos(np.clip((np.trace(self.R1) - 1) / 2, -1, 1))
        angle_deg = np.degrees(angle_rad)

        print(f"[StereoTriangulator] Calibration loaded (with stereo rectification)")
        print(f"  Raw baseline: {baseline:.2f} cm  |  Relative rotation: {angle_deg:.1f} deg")
        print(f"  Rectified focal length: {rect_f:.1f} px  |  Rectified baseline: {rect_baseline:.2f} cm")
        disp_100 = rect_f * rect_baseline / 100.0
        disp_200 = rect_f * rect_baseline / 200.0
        print(f"  Expected disparity: {disp_100:.0f}px @100cm  {disp_200:.0f}px @200cm")

    def _get_projection_matrix(self, cmtx, R, T):
        """Compute projection matrix P = K @ [R|T]."""
        RT = np.zeros((3, 4), dtype=np.float64)
        RT[:3, :3] = R
        RT[:3, 3] = T.flatten()
        return cmtx @ RT

    # ================================================================
    # CAMERAS
    # ================================================================

    def start_cameras(self, width=None, height=None):
        """Open camera streams with Arducam MJPG configuration."""
        if width is None:
            width = FRAME_WIDTH
        if height is None:
            height = FRAME_HEIGHT

        self.cap_left = cv2.VideoCapture(self.cam_left_id, cv2.CAP_DSHOW)
        self.cap_right = cv2.VideoCapture(self.cam_right_id, cv2.CAP_DSHOW)

        if not self.cap_left.isOpened():
            raise RuntimeError(f"Failed to open left camera (ID: {self.cam_left_id})")
        if not self.cap_right.isOpened():
            raise RuntimeError(f"Failed to open right camera (ID: {self.cam_right_id})")

        self.cap_left.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap_right.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        settings_left = configure_camera(self.cap_left, width, height)
        settings_right = configure_camera(self.cap_right, width, height)

        print(f"[StereoTriangulator] Cameras started:")
        print(f"  LEFT:  {settings_left['width']}x{settings_left['height']} "
              f"@ {settings_left['fps']:.0f}fps"
              f" trigger={'OK' if settings_left.get('trigger_ok') else 'OFF'}")
        print(f"  RIGHT: {settings_right['width']}x{settings_right['height']} "
              f"@ {settings_right['fps']:.0f}fps"
              f" trigger={'OK' if settings_right.get('trigger_ok') else 'OFF'}")

    def stop_cameras(self):
        """Release camera streams."""
        if self.cap_left:
            self.cap_left.release()
        if self.cap_right:
            self.cap_right.release()

    # ================================================================
    # TRIANGULATION + VALIDATION
    # ================================================================

    def triangulate(self, point_left_rect, point_right_rect):
        """
        Triangulate 3D point from rectified stereo correspondences using DLT.

        Args:
            point_left_rect: rectified pixel coords from left camera
            point_right_rect: rectified pixel coords from right camera

        Returns:
            numpy array (X, Y, Z) in calibration units, in original camera0 frame
        """
        A = np.array([
            point_left_rect[1] * self.P_rect0[2, :] - self.P_rect0[1, :],
            self.P_rect0[0, :] - point_left_rect[0] * self.P_rect0[2, :],
            point_right_rect[1] * self.P_rect1[2, :] - self.P_rect1[1, :],
            self.P_rect1[0, :] - point_right_rect[0] * self.P_rect1[2, :]
        ])

        _, _, Vh = np.linalg.svd(A)
        X = Vh[-1]
        X_rect = X[:3] / X[3]

        # Convert from rectified world frame back to original camera0 frame
        X_cam0 = self.R_rect0.T @ X_rect

        return X_cam0

    def _reprojection_error(self, point_3d_cam0, point_left_rect, point_right_rect):
        """
        Project 3D point back to both rectified images and measure pixel
        distance from rectified detections.

        Args:
            point_3d_cam0: 3D point in original camera0 frame
            point_left_rect: rectified pixel coords from left camera
            point_right_rect: rectified pixel coords from right camera

        Returns (err_left, err_right) in pixels.
        """
        # Convert to rectified world frame for projection
        point_3d_rect = self.R_rect0 @ np.array(point_3d_cam0, dtype=np.float64)
        pt = np.array([point_3d_rect[0], point_3d_rect[1], point_3d_rect[2], 1.0])

        uv_l = self.P_rect0 @ pt
        uv_l = uv_l[:2] / uv_l[2]

        uv_r = self.P_rect1 @ pt
        uv_r = uv_r[:2] / uv_r[2]

        err_l = np.hypot(uv_l[0] - point_left_rect[0], uv_l[1] - point_left_rect[1])
        err_r = np.hypot(uv_r[0] - point_right_rect[0], uv_r[1] - point_right_rect[1])

        return err_l, err_r

    def _validate_triangulation(self, point_left_rect, point_right_rect, point_3d):
        """
        Run all sanity checks on a triangulation result.

        All pixel coordinates are in the RECTIFIED image space (after
        stereo rectification), so epipolar lines are horizontal and
        disparity is purely in x.

        Checks (in order):
          1. Disparity in valid range
          2. Epipolar consistency (y-coords match — should be <1-2px after rectification)
          3. Z in plausible range
          4. Reprojection error below threshold

        Returns (is_valid, reject_reason_or_None)
        """
        disparity = point_left_rect[0] - point_right_rect[0]

        # Check 1: Disparity range
        if disparity < self.MIN_DISPARITY:
            return False, f"low_disp({disparity:.0f}px)"
        if disparity > self.MAX_DISPARITY:
            return False, f"high_disp({disparity:.0f}px)"

        # Check 2: Epipolar consistency (near-zero after rectification)
        y_diff = abs(point_left_rect[1] - point_right_rect[1])
        if y_diff > self.MAX_EPIPOLAR_ERR:
            return False, f"epipolar(dy={y_diff:.0f}px)"

        X, Y, Z = point_3d

        # Check 3: Z range
        if Z <= 0:
            return False, f"z_negative({Z:.1f})"
        if Z < self.MIN_Z:
            return False, f"z_close({Z:.1f})"
        if Z > self.MAX_Z:
            return False, f"z_far({Z:.1f})"

        # Check 4: Reprojection error
        err_l, err_r = self._reprojection_error(point_3d, point_left_rect, point_right_rect)
        max_err = max(err_l, err_r)
        if max_err > self.MAX_REPROJ_ERR:
            return False, f"reproj({max_err:.1f}px)"

        return True, None

    # ================================================================
    # MAIN UPDATE LOOP
    # ================================================================

    def update(self):
        """
        Capture synchronized frames, detect ball, and triangulate.

        Uses grab()/retrieve() to ensure both frames come from the
        same trigger pulse.

        Returns:
            dict with all detection and triangulation results
        """
        result = {
            'left_frame': None,
            'right_frame': None,
            'left_detection': None,
            'right_detection': None,
            'left_all_candidates': [],
            'right_all_candidates': [],
            'left_rejected': [],
            'right_rejected': [],
            'left_mask': None,
            'right_mask': None,
            'found_3d': False,
            'position_3d': None,
            'disparity': None,
            'reject_reason': None,
            'reproj_err': None
        }

        # --- CRITICAL: grab both before retrieving ---
        grabbed_l = self.cap_left.grab()
        grabbed_r = self.cap_right.grab()

        if not grabbed_l or not grabbed_r:
            return result

        ret_left, frame_left = self.cap_left.retrieve()
        ret_right, frame_right = self.cap_right.retrieve()

        if not ret_left or not ret_right:
            return result

        result['left_frame'] = frame_left
        result['right_frame'] = frame_right
        self._frame_count += 1

        # --- Detect ball in both frames ---
        best_l, cands_l, rej_l, mask_l = self.detector_left.detect(frame_left)
        best_r, cands_r, rej_r, mask_r = self.detector_right.detect(frame_right)

        result['left_detection'] = best_l
        result['right_detection'] = best_r
        result['left_all_candidates'] = cands_l
        result['right_all_candidates'] = cands_r
        result['left_rejected'] = rej_l
        result['right_rejected'] = rej_r
        result['left_mask'] = mask_l
        result['right_mask'] = mask_r

        # --- Triangulate if both detected ---
        if best_l is None or best_r is None:
            return result

        # Undistort + rectify detected pixel coords (makes cameras virtually parallel)
        point_left = self._rectify_point(
            best_l['center'], self.cmtx0, self.dist0, self.R_rect0, self.P_rect0)
        point_right = self._rectify_point(
            best_r['center'], self.cmtx1, self.dist1, self.R_rect1, self.P_rect1)

        disparity = point_left[0] - point_right[0]
        result['disparity'] = disparity

        point_3d = self.triangulate(point_left, point_right)

        is_valid, reject_reason = self._validate_triangulation(
            point_left, point_right, point_3d)

        if is_valid:
            result['found_3d'] = True
            result['position_3d'] = tuple(point_3d)
            err_l, err_r = self._reprojection_error(point_3d, point_left, point_right)
            result['reproj_err'] = max(err_l, err_r)
        else:
            result['reject_reason'] = reject_reason

        return result

    # ================================================================
    # VISUALIZATION
    # ================================================================

    def draw_results(self, result):
        """Draw detection and triangulation results on frames."""
        left_vis = result['left_frame'].copy()
        right_vis = result['right_frame'].copy()

        # Draw detections
        for vis, det in [(left_vis, result['left_detection']),
                         (right_vis, result['right_detection'])]:
            if det is not None:
                cx, cy = int(det['center'][0]), int(det['center'][1])
                r = max(8, int(np.sqrt(det['area'] / np.pi)))
                cv2.circle(vis, (cx, cy), r, (0, 255, 0), 2)
                cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)
                label = f"A:{det['area']:.0f} S:{det['score']:.2f}"
                if det['is_orange']:
                    label += " [O]"
                cv2.putText(vis, label, (cx+10, cy-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 3D result or rejection reason
        if result['found_3d']:
            X, Y, Z = result['position_3d']
            cv2.putText(left_vis, f"3D: X={X:.1f} Y={Y:.1f} Z={Z:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(left_vis, f"Disp: {result['disparity']:.1f}px  "
                        f"Reproj: {result['reproj_err']:.1f}px", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        elif result['reject_reason']:
            cv2.putText(left_vis, f"REJECTED: {result['reject_reason']}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif result['left_detection'] is not None or result['right_detection'] is not None:
            l_ok = result['left_detection'] is not None
            r_ok = result['right_detection'] is not None
            cv2.putText(left_vis, f"L:{'OK' if l_ok else '--'} R:{'OK' if r_ok else '--'}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        return left_vis, right_vis

    # ================================================================
    # BACKGROUND / UTILITY
    # ================================================================

    def build_background(self, frame_left, frame_right):
        """Feed frames to both detectors during background learning."""
        self.detector_left.build_background(frame_left)
        self.detector_right.build_background(frame_right)
        self._frame_count += 1

    def reset_background(self):
        """Reset MOG2 background model on both detectors."""
        self.detector_left.reset()
        self.detector_right.reset()
        self._frame_count = 0

    def warmup_status(self):
        """
        Get background model warmup progress.

        Returns:
            dict with 'ready', 'progress' (0.0-1.0), 'frames'
        """
        progress = min(self._frame_count / self.WARMUP_FRAMES, 1.0)
        return {
            'ready': self._frame_count >= self.WARMUP_FRAMES,
            'left_ready': self._frame_count >= self.WARMUP_FRAMES,
            'right_ready': self._frame_count >= self.WARMUP_FRAMES,
            'left_progress': progress,
            'right_progress': progress,
            'progress': progress,
            'frames': self._frame_count
        }

    def project_to_image(self, point_3d, camera='left'):
        """
        Project a 3D point to pixel coordinates in camera image.
        Applies lens distortion for accurate overlay on raw frames.

        Args:
            point_3d: (X, Y, Z) in calibration units
            camera: 'left' or 'right'

        Returns:
            (u, v) pixel coordinates, or None if behind camera
        """
        if camera == 'left':
            R, T, cmtx, dist = self.R0, self.T0, self.cmtx0, self.dist0
        else:
            R, T, cmtx, dist = self.R1, self.T1, self.cmtx1, self.dist1

        # Check if point is in front of camera
        pt_w = np.array([point_3d[0], point_3d[1], point_3d[2]], dtype=np.float64)
        pt_cam = R @ pt_w + T.flatten()
        if pt_cam[2] <= 0:
            return None

        rvec, _ = cv2.Rodrigues(R)
        pts = np.array([[point_3d[0], point_3d[1], point_3d[2]]], dtype=np.float64)
        uv, _ = cv2.projectPoints(pts, rvec, T, cmtx, dist)
        return (float(uv[0, 0, 0]), float(uv[0, 0, 1]))

    def get_baseline(self):
        """Get stereo baseline (same units as calibration)."""
        return np.linalg.norm(self.T1 - self.T0)

    def get_focal_length_px(self):
        """Get average focal length in pixels."""
        return (self.cmtx0[0, 0] + self.cmtx1[0, 0]) / 2