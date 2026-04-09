"""
Find camera->robot transform using click + local circle detection.

Same output as test_find_points_based_transform.py (Kabsch R, t saved to JSON),
but with local circle detection at the click point for better accuracy.

Instead of using the raw click coordinate, this script crops a small ROI around
the click and runs HoughCircles + adaptive threshold to find the exact ball/marker
center. This is more robust than raw clicking, especially for small or distant objects.

WORKFLOW:
  1. Place a ball or circular marker at a known robot position.
  2. Press SPACE to freeze frame.
  3. Click the ball/marker on the LEFT image — circle auto-detected.
  4. Click the same ball/marker on the RIGHT image — circle auto-detected.
  5. Script triangulates the refined 3D point.
  6. Enter the matching robot-frame (x, y, z) in the terminal.
  7. Press SPACE to unfreeze, move to next position, repeat.
  8. Collect enough pairs (8-15 recommended), press 's' to solve.

DETECTION PIPELINE (per click):
  1. Crop 150x150 ROI around click point.
  2. HoughCircles on grayscale ROI — pick circle closest to click.
  3. Fallback: adaptive threshold + contour + minEnclosingCircle.
  4. If neither works, raw click coordinate is used.

How to run:
    python scripts/test_find_click_circle_transform.py
    python scripts/test_find_click_circle_transform.py --required-points 12
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from comm_function.points_based_transform import (
    DEFAULT_POINTS_BASED_TRANSFORM_FILE,
    points_based_transform,
    save_points_based_transform,
    transform_points,
)
from config.camera_config import load_camera_settings
from tracking.stereo_triangulator import StereoTriangulator


def _print(*args, **kwargs) -> None:
    print("from planner in terminal", *args, **kwargs)


class ClickCircleTransformFinder:
    """Click + local circle detection for camera->robot transform calibration."""

    # Click mode ROI size (half-width)
    CLICK_ROI_HALF = 50

    # HoughCircles parameters for small ROI
    HOUGH_DP = 1.2
    HOUGH_MIN_DIST = 50
    HOUGH_PARAM1 = 50
    HOUGH_PARAM2 = 25
    HOUGH_MIN_RADIUS = 5
    HOUGH_MAX_RADIUS = 35

    def __init__(
        self,
        required_points: int = 15,
        camera_scale_to_robot_units: float = 10.0,
        output_file: str = DEFAULT_POINTS_BASED_TRANSFORM_FILE,
    ) -> None:
        self.script_dir = SCRIPT_DIR
        self.base_dir = PARENT_DIR
        self.calibration_dir = os.path.join(
            self.base_dir, "camera_calibration", "camera_parameters"
        )

        cam_settings = load_camera_settings()
        self.frame_width = cam_settings["frame_width"]
        self.frame_height = cam_settings["frame_height"]
        self.cam_left_id = cam_settings["camera0"]
        self.cam_right_id = cam_settings["camera1"]

        self.display_width = 640
        self.display_height = int(self.display_width * self.frame_height / self.frame_width)

        self.required_points = int(required_points)
        self.camera_scale_to_robot_units = float(camera_scale_to_robot_units)
        self.output_file = os.path.abspath(output_file)

        self.triangulator: Optional[StereoTriangulator] = None
        self.latest_left = None
        self.latest_right = None
        self.frozen = False
        self.frozen_left = None
        self.frozen_right = None

        # Click state: 0=waiting left, 1=waiting right, 2=both done
        self.click_stage = 0
        self.click_left_raw: Optional[tuple[int, int]] = None
        self.click_right_raw: Optional[tuple[int, int]] = None
        self.detect_left: Optional[dict] = None  # refined detection
        self.detect_right: Optional[dict] = None
        self._pending_pair = False

        # Collected correspondences
        self.camera_points_raw: list[tuple[float, float, float]] = []
        self.camera_points_scaled: list[tuple[float, float, float]] = []
        self.robot_points: list[tuple[float, float, float]] = []
        self.reproj_errors: list[float] = []

        self.window_name = "Click-Circle Transform Finder"
        self.transform_result = None
        self._last_status_print = 0.0

    def check_calibration(self) -> bool:
        required = [
            "camera0_intrinsics.dat", "camera1_intrinsics.dat",
            "camera0_rot_trans.dat", "camera1_rot_trans.dat",
        ]
        missing = [f for f in required if not os.path.exists(os.path.join(self.calibration_dir, f))]
        if missing:
            _print("ERROR: Missing calibration files:")
            for f in missing:
                _print(f"  - {f}")
            return False
        return True

    # --- Circle detection (from test_static_triangulation.py) ---

    def detect_circle_near_click(self, frame, click_pt):
        """
        Detect circle near a click point using local detection.

        1. Crop 150x150 ROI around click
        2. HoughCircles for precise circle
        3. Fallback: adaptive threshold + contour + minEnclosingCircle

        Returns:
            dict with 'center', 'area', 'radius', 'method' or None
        """
        cx, cy = int(click_pt[0]), int(click_pt[1])
        h, w = frame.shape[:2]
        half = self.CLICK_ROI_HALF

        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half)
        y2 = min(h, cy + half)
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Try HoughCircles first
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT,
            dp=self.HOUGH_DP,
            minDist=self.HOUGH_MIN_DIST,
            param1=self.HOUGH_PARAM1,
            param2=self.HOUGH_PARAM2,
            minRadius=self.HOUGH_MIN_RADIUS,
            maxRadius=self.HOUGH_MAX_RADIUS,
        )

        if circles is not None:
            roi_cx = cx - x1
            roi_cy = cy - y1
            best_dist = float('inf')
            best_circle = None
            for c in circles[0]:
                d = np.hypot(c[0] - roi_cx, c[1] - roi_cy)
                if d < best_dist:
                    best_dist = d
                    best_circle = c
            if best_circle is not None:
                lx = float(best_circle[0] + x1)
                ly = float(best_circle[1] + y1)
                radius = float(best_circle[2])
                return {
                    'center': (lx, ly),
                    'area': np.pi * radius * radius,
                    'radius': radius,
                    'method': 'hough',
                }

        # Fallback: adaptive threshold + contour
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 5)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        roi_cx = cx - x1
        roi_cy = cy - y1
        best = None
        best_dist = float('inf')
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20:
                continue
            (ccx, ccy), radius = cv2.minEnclosingCircle(cnt)
            d = np.hypot(ccx - roi_cx, ccy - roi_cy)
            if d < best_dist:
                best_dist = d
                best = (float(ccx + x1), float(ccy + y1), float(radius), float(area))

        if best is not None:
            return {
                'center': (best[0], best[1]),
                'area': best[3],
                'radius': best[2],
                'method': 'contour',
            }

        return None

    # --- Coordinate transforms ---

    def pixel_to_original(self, x_display: int, y_display: int, is_right: bool) -> tuple[int, int]:
        if is_right:
            x_display -= self.display_width
        scale_x = self.frame_width / self.display_width
        scale_y = self.frame_height / self.display_height
        return int(x_display * scale_x), int(y_display * scale_y)

    # --- Triangulation ---

    def _triangulate_pair(self, pt_left, pt_right):
        """Triangulate a pair of points. Returns (pos_3d, reproj_err) or (None, None)."""
        tri = self.triangulator
        pt_l_rect = tri._rectify_point(pt_left, tri.cmtx0, tri.dist0, tri.R_rect0, tri.P_rect0)
        pt_r_rect = tri._rectify_point(pt_right, tri.cmtx1, tri.dist1, tri.R_rect1, tri.P_rect1)

        disp = pt_l_rect[0] - pt_r_rect[0]
        if disp <= 0:
            _print("ERROR: Negative disparity. Click left then right on the same target.")
            return None, None

        pos = tri.triangulate(pt_l_rect, pt_r_rect)
        if pos[2] <= 0:
            _print("ERROR: Point behind camera.")
            return None, None

        err_l, err_r = tri._reprojection_error(pos, pt_l_rect, pt_r_rect)
        reproj = max(err_l, err_r)
        return (float(pos[0]), float(pos[1]), float(pos[2])), float(reproj)

    # --- User input ---

    def _prompt_robot_point(self) -> Optional[tuple[float, float, float]]:
        _print("Enter matching ROBOT point as: x y z  (or x,y,z). Type 'skip' to discard.")
        while True:
            raw = input("robot xyz> ").strip()
            if not raw:
                continue
            if raw.lower() in {"skip", "s"}:
                return None
            txt = raw.replace(",", " ")
            parts = [p for p in txt.split() if p]
            if len(parts) != 3:
                _print("Please enter exactly 3 numbers.")
                continue
            try:
                return (float(parts[0]), float(parts[1]), float(parts[2]))
            except ValueError:
                _print("Invalid numbers. Try again.")

    # --- Mouse callback ---

    def on_mouse(self, event, x, y, _flags, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        is_right = x >= self.display_width
        orig_x, orig_y = self.pixel_to_original(x, y, is_right)

        # Get the frame to run circle detection on
        if self.frozen:
            frame = self.frozen_right if is_right else self.frozen_left
        else:
            frame = self.latest_right if is_right else self.latest_left

        if frame is None:
            return

        if self.click_stage == 0:
            if is_right:
                return
            self.click_left_raw = (orig_x, orig_y)
            # Run circle detection
            det = self.detect_circle_near_click(frame, (orig_x, orig_y))
            if det is not None:
                self.detect_left = det
                _print(f"LEFT: circle detected at ({det['center'][0]:.1f}, "
                       f"{det['center'][1]:.1f}) via {det['method']}, r={det['radius']:.1f}px")
            else:
                self.detect_left = {
                    'center': (float(orig_x), float(orig_y)),
                    'area': 0, 'radius': 0, 'method': 'raw_click',
                }
                _print(f"LEFT: no circle found, using raw click ({orig_x}, {orig_y})")
            self.click_stage = 1
            return

        if self.click_stage == 1:
            if not is_right:
                return
            self.click_right_raw = (orig_x, orig_y)
            det = self.detect_circle_near_click(frame, (orig_x, orig_y))
            if det is not None:
                self.detect_right = det
                _print(f"RIGHT: circle detected at ({det['center'][0]:.1f}, "
                       f"{det['center'][1]:.1f}) via {det['method']}, r={det['radius']:.1f}px")
            else:
                self.detect_right = {
                    'center': (float(orig_x), float(orig_y)),
                    'area': 0, 'radius': 0, 'method': 'raw_click',
                }
                _print(f"RIGHT: no circle found, using raw click ({orig_x}, {orig_y})")
            self.click_stage = 2
            self._pending_pair = True
            return

    # --- Process completed pair ---

    def _reset_click(self) -> None:
        self.click_stage = 0
        self.click_left_raw = None
        self.click_right_raw = None
        self.detect_left = None
        self.detect_right = None
        self._pending_pair = False

    def _process_pending_pair(self) -> None:
        if not self._pending_pair:
            return
        self._pending_pair = False

        pt_left = self.detect_left['center']
        pt_right = self.detect_right['center']

        cam_raw, reproj = self._triangulate_pair(pt_left, pt_right)
        if cam_raw is None:
            self._reset_click()
            return

        _print(f"  3D point (cam): ({cam_raw[0]:+.2f}, {cam_raw[1]:+.2f}, {cam_raw[2]:+.2f})")
        _print(f"  Reproj error: {reproj:.2f} px")

        if reproj > 3.0:
            _print(f"  REJECTED: reproj {reproj:.2f}px > 3px threshold. Re-click.")
            self._reset_click()
            return
        if reproj > 1.0:
            _print(f"  WARNING: reproj {reproj:.2f}px is marginal (1-3px).")

        robot_xyz = self._prompt_robot_point()
        if robot_xyz is None:
            _print("Pair skipped.")
            self._reset_click()
            return

        scale = self.camera_scale_to_robot_units
        cam_scaled = (cam_raw[0] * scale, cam_raw[1] * scale, cam_raw[2] * scale)

        self.camera_points_raw.append(cam_raw)
        self.camera_points_scaled.append(cam_scaled)
        self.robot_points.append(robot_xyz)
        self.reproj_errors.append(reproj)

        idx = len(self.camera_points_raw)
        _print(
            f"[PAIR {idx}/{self.required_points}] "
            f"cam_scaled=({cam_scaled[0]:+.1f}, {cam_scaled[1]:+.1f}, {cam_scaled[2]:+.1f}) "
            f"robot=({robot_xyz[0]:+.1f}, {robot_xyz[1]:+.1f}, {robot_xyz[2]:+.1f}) "
            f"reproj={reproj:.2f}px"
        )

        self._reset_click()

        if len(self.camera_points_scaled) >= self.required_points:
            result = self._compute_transform()
            self._print_transform_summary(result)
            self._save_transform(result)

    # --- Transform computation ---

    def _compute_transform(self) -> dict:
        cam = np.asarray(self.camera_points_scaled, dtype=float)
        rob = np.asarray(self.robot_points, dtype=float)

        R, t, diag = points_based_transform(cam, rob, return_diagnostics=True)
        fitted = transform_points(cam, R, t)
        residuals = rob - fitted
        per_point_err = np.linalg.norm(residuals, axis=1)

        median_err = float(np.median(per_point_err))
        outlier_threshold = max(median_err * 2.0, 5.0)
        outlier_mask = per_point_err > outlier_threshold
        outlier_indices = list(np.where(outlier_mask)[0])

        result = {
            "R": R, "t": t, "diagnostics": diag,
            "residuals": residuals, "per_point_err": per_point_err,
            "outlier_indices": outlier_indices, "outlier_threshold": outlier_threshold,
        }

        if outlier_indices:
            _print(f"\n  OUTLIER WARNING: {len(outlier_indices)} point(s) have "
                   f"residual > {outlier_threshold:.1f}:")
            for idx in outlier_indices:
                _print(f"    Point {idx}: error={per_point_err[idx]:.2f}")
            if len(cam) - len(outlier_indices) >= 3:
                answer = input("  Re-solve excluding outliers? (y/n)> ").strip().lower()
                if answer == "y":
                    keep = ~outlier_mask
                    R2, t2, diag2 = points_based_transform(
                        cam[keep], rob[keep], return_diagnostics=True
                    )
                    _print(f"  Re-solved with {int(keep.sum())} points: "
                           f"rmse={diag2['rmse']:.4f} (was {diag['rmse']:.4f})")
                    keep_list = list(np.where(keep)[0])
                    self.camera_points_raw = [self.camera_points_raw[i] for i in keep_list]
                    self.camera_points_scaled = [self.camera_points_scaled[i] for i in keep_list]
                    self.robot_points = [self.robot_points[i] for i in keep_list]
                    self.reproj_errors = [self.reproj_errors[i] for i in keep_list]

                    fitted2 = transform_points(cam[keep], R2, t2)
                    residuals2 = rob[keep] - fitted2
                    per_point_err2 = np.linalg.norm(residuals2, axis=1)
                    result = {
                        "R": R2, "t": t2, "diagnostics": diag2,
                        "residuals": residuals2, "per_point_err": per_point_err2,
                        "outlier_indices": [], "outlier_threshold": outlier_threshold,
                    }

        self.transform_result = result
        return result

    def _print_transform_summary(self, result: dict) -> None:
        R = result["R"]
        t = result["t"]
        diag = result["diagnostics"]
        per_point_err = result["per_point_err"]

        _print("")
        _print("=" * 74)
        _print(" POINTS-BASED CAMERA->ROBOT TRANSFORM (CLICK+CIRCLE) ")
        _print("=" * 74)
        _print(f"Pairs used: {diag['num_points']}")
        _print(f"Camera scale: {self.camera_scale_to_robot_units:.6f}")
        _print("")
        _print("R =")
        _print(np.array2string(R, precision=8, suppress_small=False))
        _print("")
        _print("t =")
        _print(np.array2string(t, precision=8, suppress_small=False))
        _print("")
        _print(f"Fit quality: rmse={diag['rmse']:.4f}mm, "
               f"mean={diag['mean_error']:.4f}mm, "
               f"max={diag['max_error']:.4f}mm, "
               f"det(R)={diag['rotation_det']:.6f}")

        _print("")
        _print(f"{'#':>3}  {'Camera Scaled (X,Y,Z)':>30}  "
               f"{'Robot (X,Y,Z)':>30}  {'Residual':>9}  {'Flag':>7}")
        _print("-" * 90)
        for i in range(len(self.camera_points_scaled)):
            cs = self.camera_points_scaled[i]
            rp = self.robot_points[i]
            err = per_point_err[i] if i < len(per_point_err) else float("nan")
            flag = "OUTLIER" if i in result.get("outlier_indices", []) else ""
            _print(f"{i:>3}  "
                   f"({cs[0]:+8.1f}, {cs[1]:+8.1f}, {cs[2]:+8.1f})  "
                   f"({rp[0]:+8.1f}, {rp[1]:+8.1f}, {rp[2]:+8.1f})  "
                   f"{err:>8.2f}  {flag:>7}")
        _print("-" * 90)
        _print("=" * 74)

    def _save_transform(self, result: dict) -> None:
        path = save_points_based_transform(
            rotation=result["R"],
            translation=result["t"],
            output_path=self.output_file,
            camera_scale_to_robot_units=self.camera_scale_to_robot_units,
        )
        _print(f"Saved transform: {path}")

    # --- Visualization ---

    def _draw_overlay(self, left_small, right_small) -> None:
        h = self.display_height
        scale_x = self.display_width / self.frame_width
        scale_y = self.display_height / self.frame_height

        for img, name in ((left_small, "LEFT"), (right_small, "RIGHT")):
            cv2.rectangle(img, (0, 0), (430, 50), (0, 0, 0), -1)
            cv2.putText(img, f"{name} {'[FROZEN]' if self.frozen else '[LIVE]'}",
                        (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(img, f"Pairs: {len(self.camera_points_scaled)}/{self.required_points}",
                        (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1)

        # Stage guidance
        if self.click_stage == 0:
            cv2.rectangle(left_small, (0, 0),
                          (self.display_width - 1, self.display_height - 1), (0, 255, 255), 2)
            cv2.putText(left_small, "Click target on LEFT",
                        (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        elif self.click_stage == 1:
            cv2.rectangle(right_small, (0, 0),
                          (self.display_width - 1, self.display_height - 1), (0, 255, 255), 2)
            cv2.putText(right_small, "Click SAME target on RIGHT",
                        (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # Draw click crosshairs and detection circles
        if self.click_left_raw is not None:
            px = int(self.click_left_raw[0] * scale_x)
            py = int(self.click_left_raw[1] * scale_y)
            cv2.drawMarker(left_small, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 20, 1)
            if self.detect_left is not None:
                cx = int(self.detect_left['center'][0] * scale_x)
                cy = int(self.detect_left['center'][1] * scale_y)
                r = max(5, int(self.detect_left.get('radius', 8) * scale_x))
                cv2.circle(left_small, (cx, cy), r, (0, 255, 0), 2)
                cv2.circle(left_small, (cx, cy), 3, (0, 0, 255), -1)

        if self.click_right_raw is not None:
            px = int(self.click_right_raw[0] * scale_x)
            py = int(self.click_right_raw[1] * scale_y)
            cv2.drawMarker(right_small, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 20, 1)
            if self.detect_right is not None:
                cx = int(self.detect_right['center'][0] * scale_x)
                cy = int(self.detect_right['center'][1] * scale_y)
                r = max(5, int(self.detect_right.get('radius', 8) * scale_x))
                cv2.circle(right_small, (cx, cy), r, (0, 255, 0), 2)
                cv2.circle(right_small, (cx, cy), 3, (0, 0, 255), -1)

        cv2.putText(right_small,
                    "SPACE freeze | r reset | d del-last | s solve | q quit",
                    (10, self.display_height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (190, 190, 190), 1)

    # --- Main loop ---

    def run(self) -> int:
        _print("\n" + "=" * 74)
        _print(" FIND TRANSFORM (CLICK + CIRCLE DETECTION) ")
        _print("=" * 74)
        _print(f"Cameras: L={self.cam_left_id} R={self.cam_right_id}")
        _print(f"Required pairs: {self.required_points}")
        _print(f"Scale: {self.camera_scale_to_robot_units} (camera units -> robot mm)")
        _print(f"Output: {self.output_file}")
        _print("")
        _print("WORKFLOW:")
        _print("  1. Place ball/marker at known robot position")
        _print("  2. Press SPACE to freeze frame")
        _print("  3. Click target on LEFT, then RIGHT — circle auto-detected")
        _print("  4. Enter robot XYZ in terminal")
        _print("  5. Repeat across workspace")
        _print(f"  6. Press 's' to solve (need >= 3, recommend {self.required_points})")
        _print("")
        _print("Controls: SPACE freeze | r reset click | d delete-last | s solve | q quit")
        _print("=" * 74)

        if not self.check_calibration():
            return 1

        try:
            self.triangulator = StereoTriangulator(
                calibration_dir=self.calibration_dir,
                cam_left_id=self.cam_left_id,
                cam_right_id=self.cam_right_id,
            )
            self.triangulator.start_cameras(self.frame_width, self.frame_height)
        except Exception as exc:
            _print(f"ERROR starting triangulator/cameras: {exc}")
            return 1

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        exit_code = 0
        try:
            while True:
                if not self.frozen:
                    result = self.triangulator.update()
                    if result["left_frame"] is None:
                        continue
                    self.latest_left = result["left_frame"]
                    self.latest_right = result["right_frame"]

                if self.latest_left is None or self.latest_right is None:
                    continue

                left_small = cv2.resize(self.latest_left, (self.display_width, self.display_height))
                right_small = cv2.resize(self.latest_right, (self.display_width, self.display_height))
                self._draw_overlay(left_small, right_small)
                cv2.imshow(self.window_name, cv2.hconcat([left_small, right_small]))

                if self._pending_pair:
                    cv2.waitKey(1)
                    self._process_pending_pair()

                now = time.time()
                if (now - self._last_status_print > 8.0
                        and len(self.camera_points_scaled) < self.required_points
                        and len(self.camera_points_scaled) > 0):
                    _print(f"Progress: {len(self.camera_points_scaled)}/{self.required_points} pairs")
                    self._last_status_print = now

                if (len(self.camera_points_scaled) >= self.required_points
                        and self.transform_result is not None):
                    break

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord(" "):
                    if not self.frozen:
                        self.frozen_left = self.latest_left.copy()
                        self.frozen_right = self.latest_right.copy()
                        self.latest_left = self.frozen_left
                        self.latest_right = self.frozen_right
                        self.frozen = True
                        _print("[FROZEN] Click target on LEFT image")
                    else:
                        self.frozen = False
                        self.frozen_left = None
                        self.frozen_right = None
                        self._reset_click()
                        _print("[LIVE]")
                elif key == ord("r"):
                    self._reset_click()
                    _print("Click pair reset.")
                elif key == ord("d"):
                    if self.camera_points_scaled:
                        self.camera_points_scaled.pop()
                        self.camera_points_raw.pop()
                        self.robot_points.pop()
                        self.reproj_errors.pop()
                        self.transform_result = None
                        _print(f"Deleted last pair. Remaining: {len(self.camera_points_scaled)}")
                elif key == ord("s"):
                    if len(self.camera_points_scaled) < 3:
                        _print(f"Need at least 3 pairs (have {len(self.camera_points_scaled)}).")
                    else:
                        result = self._compute_transform()
                        self._print_transform_summary(result)
                        self._save_transform(result)
                        _print("\nContinue adding pairs to improve, or 'q' to quit.")

        except KeyboardInterrupt:
            _print("Interrupted.")
        finally:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            try:
                self.triangulator.stop_cameras()
            except Exception:
                pass

        if self.transform_result is None and len(self.camera_points_scaled) >= 3:
            _print("\nComputing final transform...")
            result = self._compute_transform()
            self._print_transform_summary(result)
            self._save_transform(result)

        if self.transform_result is None:
            _print("Not enough pairs to solve transform (need >= 3).")
            exit_code = 1

        return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find camera->robot transform using click + local circle detection."
    )
    parser.add_argument("--required-points", type=int, default=15,
                        help="Number of matched pairs to collect (default: 15)")
    parser.add_argument("--camera-scale-to-robot-units", type=float, default=10.0,
                        help="Scale for camera units -> robot mm (default: 10.0 for cm->mm)")
    parser.add_argument("--output-file", type=str,
                        default=DEFAULT_POINTS_BASED_TRANSFORM_FILE,
                        help="Path to write solved transform JSON.")
    args = parser.parse_args()
    if args.required_points < 3:
        parser.error("--required-points must be >= 3")

    app = ClickCircleTransformFinder(
        required_points=args.required_points,
        camera_scale_to_robot_units=args.camera_scale_to_robot_units,
        output_file=args.output_file,
    )
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
