"""
Find camera->robot transform using ArUco marker detection.

Same output as test_find_points_based_transform.py (Kabsch R, t saved to JSON),
but uses cv2.aruco.detectMarkers instead of manual clicking or checkerboard.

ArUco markers are far more robust than checkerboards at long distances because
they use thick binary patterns that survive low resolution and blur.

A single ArUco marker placement yields 4 correspondence pairs (one per corner)
with sub-pixel accuracy.

WHAT YOU NEED:
  - Print a SINGLE large ArUco marker (4x4 dictionary, ID 0) at known size.
    Recommended: 150-200mm side length on A4/letter paper, mounted on cardboard.
  - Generate one at: https://chev.me/arucogen/ (Dictionary: 4x4, ID: 0)
    Or run: python -c "import cv2; img=cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),0,400); cv2.imwrite('aruco_0.png',img)"

WORKFLOW:
  1. Place ArUco marker flat on table, visible to BOTH cameras.
  2. Press SPACE to freeze and auto-detect the marker.
  3. Script marks corner[0] with RED, shows width/height direction arrows.
  4. Enter robot-frame (x, y, z) of the RED corner (corner[0]).
  5. Enter which robot axis the WIDTH direction (red arrow) corresponds to.
  6. Enter which robot axis the HEIGHT direction (blue arrow) corresponds to.
  7. Script triangulates all 4 corners and computes robot coords from geometry.
  8. Press SPACE to unfreeze, move marker to another position, repeat.
  9. Press 's' to solve the transform, or 'q' to quit.

CORNER ORDERING (ArUco standard, looking at the marker right-side up):
    corner[0] = top-left      corner[1] = top-right
    corner[3] = bottom-left   corner[2] = bottom-right

    WIDTH  direction: corner[0] -> corner[1] (red arrow)
    HEIGHT direction: corner[0] -> corner[3] (blue arrow)

DEFAULTS: 4x4 dictionary, marker ID 0, 150mm side.

How to run:
    python scripts/test_find_aruco_transform.py
    python scripts/test_find_aruco_transform.py --marker-size 200 --marker-id 0
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
PROJECT_DIR = os.path.dirname(PARENT_DIR)
CAMERA_PROPERTIES_DIR = os.path.join(PROJECT_DIR, "camera_params", "camera_properties")
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from comm_functions.points_based_transform import (
    DEFAULT_POINTS_BASED_TRANSFORM_FILE,
    points_based_transform,
    save_points_based_transform,
    transform_points,
)
from config.camera_config import load_camera_settings
from ball_tracking.stereo_triangulator import StereoTriangulator


def _print(*args, **kwargs) -> None:
    print("from planner in terminal", *args, **kwargs)


AXIS_MAP = {
    "+x": np.array([1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
    "+z": np.array([0.0, 0.0, 1.0]),
    "-z": np.array([0.0, 0.0, -1.0]),
}


class ArucoTransformFinder:
    def __init__(
        self,
        marker_size_mm: float = 180.0,
        marker_id: int = 0,
        camera_scale_to_robot_units: float = 10.0,
        required_placements: int = 3,
        output_file: str = DEFAULT_POINTS_BASED_TRANSFORM_FILE,
    ) -> None:
        self.script_dir = SCRIPT_DIR
        self.base_dir = PROJECT_DIR
        self.calibration_dir = CAMERA_PROPERTIES_DIR

        cam_settings = load_camera_settings()
        self.frame_width = cam_settings["frame_width"]
        self.frame_height = cam_settings["frame_height"]
        self.cam_left_id = cam_settings["camera0"]
        self.cam_right_id = cam_settings["camera1"]

        self.display_width = 640
        self.display_height = int(self.display_width * self.frame_height / self.frame_width)

        self.marker_size_mm = float(marker_size_mm)
        self.marker_id = int(marker_id)
        self.camera_scale_to_robot_units = float(camera_scale_to_robot_units)
        self.required_placements = int(required_placements)
        self.output_file = os.path.abspath(output_file)

        # ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self.triangulator: Optional[StereoTriangulator] = None
        self.latest_left = None
        self.latest_right = None
        self.frozen = False

        # Detection state
        self.detected_left = None   # 4x2 corners from left image
        self.detected_right = None  # 4x2 corners from right image
        self._pending_detection = False

        # Collected correspondences
        self.camera_points_raw: list[tuple[float, float, float]] = []
        self.camera_points_scaled: list[tuple[float, float, float]] = []
        self.robot_points: list[tuple[float, float, float]] = []
        self.reproj_errors: list[float] = []
        self.placement_count = 0

        self.window_name = "ArUco Transform Finder"
        self.transform_result = None

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

    # --- ArUco detection ---

    def _detect_marker(self, frame) -> Optional[np.ndarray]:
        """Detect target ArUco marker. Returns 4x2 corners or None."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners_list, ids, _rejected = self.aruco_detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return None

        # Find our target marker ID
        for i, mid in enumerate(ids):
            if int(mid[0]) == self.marker_id:
                # corners_list[i] is shape (1, 4, 2)
                corners = corners_list[i].reshape(4, 2)

                # Sub-pixel refinement on each corner
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
                for j in range(4):
                    pt = np.array([[corners[j]]], dtype=np.float32)
                    refined = cv2.cornerSubPix(gray, pt, (5, 5), (-1, -1), criteria)
                    corners[j] = refined[0, 0]

                return corners

        return None

    # --- Triangulation ---

    def _triangulate_corner_pair(self, pt_left, pt_right):
        """Triangulate a single corner pair. Returns (pos_3d, reproj_err) or (None, None)."""
        tri = self.triangulator
        pt_l_rect = tri._rectify_point(pt_left, tri.cmtx0, tri.dist0, tri.R_rect0, tri.P_rect0)
        pt_r_rect = tri._rectify_point(pt_right, tri.cmtx1, tri.dist1, tri.R_rect1, tri.P_rect1)

        disp = pt_l_rect[0] - pt_r_rect[0]
        if disp <= 0:
            return None, None

        pos = tri.triangulate(pt_l_rect, pt_r_rect)
        if pos[2] <= 0:
            return None, None

        err_l, err_r = tri._reprojection_error(pos, pt_l_rect, pt_r_rect)
        reproj = max(err_l, err_r)
        return (float(pos[0]), float(pos[1]), float(pos[2])), float(reproj)

    def _triangulate_all_corners(self, left_corners, right_corners):
        """Triangulate all 4 matched corner pairs."""
        results = []
        for i in range(4):
            pt_l = (float(left_corners[i][0]), float(left_corners[i][1]))
            pt_r = (float(right_corners[i][0]), float(right_corners[i][1]))
            pos, reproj = self._triangulate_corner_pair(pt_l, pt_r)
            results.append((pos, reproj))
        return results

    # --- Robot position computation ---

    def _compute_robot_corners(self, origin, width_dir, height_dir):
        """Compute robot-frame positions for all 4 marker corners.

        ArUco corner ordering (looking at marker right-side up):
          [0] = top-left      [1] = top-right
          [3] = bottom-left   [2] = bottom-right

        Width direction:  corner[0] -> corner[1]
        Height direction: corner[0] -> corner[3]
        """
        s = self.marker_size_mm
        positions = [
            origin,                                # [0] top-left (origin)
            origin + s * width_dir,                # [1] top-right
            origin + s * width_dir + s * height_dir,  # [2] bottom-right
            origin + s * height_dir,               # [3] bottom-left
        ]
        return [(float(p[0]), float(p[1]), float(p[2])) for p in positions]

    # --- User input ---

    @staticmethod
    def _prompt_xyz(prompt_text: str) -> Optional[tuple[float, float, float]]:
        _print(prompt_text)
        while True:
            raw = input("> ").strip()
            if not raw:
                continue
            if raw.lower() in {"skip", "s", "cancel"}:
                return None
            txt = raw.replace(",", " ")
            parts = [p for p in txt.split() if p]
            if len(parts) != 3:
                _print("Enter exactly 3 numbers (x y z). Try again.")
                continue
            try:
                return (float(parts[0]), float(parts[1]), float(parts[2]))
            except ValueError:
                _print("Invalid numbers. Try again.")

    @staticmethod
    def _prompt_axis(prompt_text: str) -> Optional[np.ndarray]:
        _print(prompt_text)
        _print("  Options: +x, -x, +y, -y, +z, -z  (or 'skip' to cancel)")
        while True:
            raw = input("> ").strip().lower()
            if not raw:
                continue
            if raw in {"skip", "s", "cancel"}:
                return None
            if raw in AXIS_MAP:
                return AXIS_MAP[raw]
            _print(f"  Unknown axis '{raw}'. Use +x, -x, +y, -y, +z, or -z.")

    # --- Process a detection ---

    def _process_detection(self) -> None:
        """Process detected marker corners: triangulate, prompt user, add pairs."""
        self._pending_detection = False

        if self.detected_left is None or self.detected_right is None:
            _print("Marker not detected in both images. Reposition and try again.")
            return

        _print(f"\nDetected ArUco marker ID {self.marker_id} in both images (4 corners).")

        # Triangulate all 4 corners
        tri_results = self._triangulate_all_corners(self.detected_left, self.detected_right)

        valid = [(i, pos, err) for i, (pos, err) in enumerate(tri_results) if pos is not None]
        failed = 4 - len(valid)
        if failed > 0:
            _print(f"  WARNING: {failed}/4 corners failed to triangulate.")
        if len(valid) < 3:
            _print("  Too few valid corners. Reposition the marker and try again.")
            return

        reproj_errs = [err for _, _, err in valid]
        _print(f"  Reproj errors: mean={np.mean(reproj_errs):.3f}px, "
               f"max={np.max(reproj_errs):.3f}px")

        # --- Prompt user for marker pose in robot frame ---
        origin_xyz = self._prompt_xyz(
            "Enter robot-frame (x y z) in mm of the RED corner (corner[0], top-left):"
        )
        if origin_xyz is None:
            _print("Placement cancelled.")
            return
        origin = np.array(origin_xyz)

        width_dir = self._prompt_axis(
            "WIDTH direction (RED arrow, corner[0] -> corner[1]) in robot frame:"
        )
        if width_dir is None:
            _print("Placement cancelled.")
            return

        height_dir = self._prompt_axis(
            "HEIGHT direction (BLUE arrow, corner[0] -> corner[3]) in robot frame:"
        )
        if height_dir is None:
            _print("Placement cancelled.")
            return

        # Validate axes are orthogonal
        dot = float(np.dot(width_dir, height_dir))
        if abs(dot) > 0.01:
            _print(f"  WARNING: Width and height axes are not orthogonal (dot={dot:.3f}).")
            _print("  This usually means the marker isn't axis-aligned. Proceed anyway.")

        # Compute robot positions for all 4 corners
        robot_corners = self._compute_robot_corners(origin, width_dir, height_dir)

        # Add valid correspondences
        scale = self.camera_scale_to_robot_units
        added = 0
        max_reproj = 3.0
        for idx, cam_raw, reproj in valid:
            if reproj > max_reproj:
                continue
            cam_scaled = (cam_raw[0] * scale, cam_raw[1] * scale, cam_raw[2] * scale)
            robot_xyz = robot_corners[idx]

            self.camera_points_raw.append(cam_raw)
            self.camera_points_scaled.append(cam_scaled)
            self.robot_points.append(robot_xyz)
            self.reproj_errors.append(reproj)
            added += 1

        skipped = len(valid) - added
        self.placement_count += 1
        _print(f"\n  Placement #{self.placement_count}: added {added} pairs"
               f"{f' (skipped {skipped} with reproj > {max_reproj}px)' if skipped else ''}.")
        _print(f"  Total pairs so far: {len(self.camera_points_scaled)}")
        _print("  Press SPACE to unfreeze and reposition marker, or 's' to solve.\n")

    # --- Transform computation ---

    def _compute_transform(self) -> dict:
        cam = np.asarray(self.camera_points_scaled, dtype=float)
        rob = np.asarray(self.robot_points, dtype=float)

        R, t, diag = points_based_transform(cam, rob, return_diagnostics=True)
        fitted = transform_points(cam, R, t)
        residuals = rob - fitted
        per_point_err = np.linalg.norm(residuals, axis=1)

        median_err = float(np.median(per_point_err))
        outlier_threshold = max(median_err * 2.5, 8.0)
        outlier_mask = per_point_err > outlier_threshold
        outlier_indices = list(np.where(outlier_mask)[0])

        result = {
            "R": R, "t": t, "diagnostics": diag,
            "residuals": residuals, "per_point_err": per_point_err,
            "outlier_indices": outlier_indices, "outlier_threshold": outlier_threshold,
        }

        if outlier_indices:
            _print(f"\n  OUTLIER WARNING: {len(outlier_indices)} point(s) have "
                   f"residual > {outlier_threshold:.1f}mm:")
            for idx in outlier_indices:
                _print(f"    Point {idx}: error={per_point_err[idx]:.2f}mm")
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
        _print(" POINTS-BASED CAMERA->ROBOT TRANSFORM (ARUCO) ")
        _print("=" * 74)
        _print(f"Pairs used: {diag['num_points']}  ({self.placement_count} placements)")
        _print(f"Camera scale: {self.camera_scale_to_robot_units:.6f}")
        _print(f"Marker: ID {self.marker_id}, {self.marker_size_mm:.1f}mm side, 4x4 dictionary")
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

        _print(f"\nPer-point residual stats: "
               f"median={float(np.median(per_point_err)):.2f}mm, "
               f"90th={float(np.percentile(per_point_err, 90)):.2f}mm")
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
            cv2.putText(img, f"Pairs: {len(self.camera_points_scaled)} "
                        f"({self.placement_count} placements)",
                        (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1)

        # Draw detected marker corners on left image
        if self.frozen and self.detected_left is not None:
            corners = self.detected_left.copy()
            for i in range(4):
                px = int(corners[i][0] * scale_x)
                py = int(corners[i][1] * scale_y)
                color = (0, 0, 255) if i == 0 else (0, 255, 0)
                cv2.circle(left_small, (px, py), 8, color, 2)
                cv2.putText(left_small, str(i), (px + 10, py - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Corner[0] = big RED circle
            c0 = corners[0]
            px0, py0 = int(c0[0] * scale_x), int(c0[1] * scale_y)
            cv2.circle(left_small, (px0, py0), 14, (0, 0, 255), 3)

            # Width arrow: corner[0] -> corner[1] (RED)
            c1 = corners[1]
            px1, py1 = int(c1[0] * scale_x), int(c1[1] * scale_y)
            cv2.arrowedLine(left_small, (px0, py0), (px1, py1), (0, 0, 255), 2,
                            tipLength=0.2)
            cv2.putText(left_small, "WIDTH", (px1 + 8, py1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # Height arrow: corner[0] -> corner[3] (BLUE)
            c3 = corners[3]
            px3, py3 = int(c3[0] * scale_x), int(c3[1] * scale_y)
            cv2.arrowedLine(left_small, (px0, py0), (px3, py3), (255, 0, 0), 2,
                            tipLength=0.2)
            cv2.putText(left_small, "HEIGHT", (px3 + 8, py3 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            # Draw outline
            for i in range(4):
                j = (i + 1) % 4
                p1 = (int(corners[i][0] * scale_x), int(corners[i][1] * scale_y))
                p2 = (int(corners[j][0] * scale_x), int(corners[j][1] * scale_y))
                cv2.line(left_small, p1, p2, (0, 255, 0), 1)

        # Draw detected marker on right image
        if self.frozen and self.detected_right is not None:
            corners = self.detected_right.copy()
            for i in range(4):
                px = int(corners[i][0] * scale_x)
                py = int(corners[i][1] * scale_y)
                color = (0, 0, 255) if i == 0 else (0, 255, 0)
                cv2.circle(right_small, (px, py), 8, color, 2)
            for i in range(4):
                j = (i + 1) % 4
                p1 = (int(corners[i][0] * scale_x), int(corners[i][1] * scale_y))
                p2 = (int(corners[j][0] * scale_x), int(corners[j][1] * scale_y))
                cv2.line(right_small, p1, p2, (0, 255, 0), 1)

        if not self.frozen:
            # Show live marker detection feedback
            if self.latest_left is not None:
                live_det = self._detect_marker(self.latest_left)
                if live_det is not None:
                    for i in range(4):
                        px = int(live_det[i][0] * scale_x)
                        py = int(live_det[i][1] * scale_y)
                        cv2.circle(left_small, (px, py), 5, (0, 200, 0), 1)
                    cv2.putText(left_small, f"ArUco ID {self.marker_id} detected",
                                (10, h - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
                else:
                    cv2.putText(left_small, "No marker detected",
                                (10, h - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)

            cv2.putText(left_small, "SPACE to freeze + detect",
                        (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        cv2.putText(right_small,
                    "SPACE freeze | s solve | d del-last | q quit",
                    (10, self.display_height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (190, 190, 190), 1)

    # --- Main loop ---

    def run(self) -> int:
        _print("\n" + "=" * 74)
        _print(" FIND TRANSFORM (ARUCO MARKER DETECTION) ")
        _print("=" * 74)
        _print(f"Marker: ID {self.marker_id}, {self.marker_size_mm:.1f}mm side, 4x4 dictionary")
        _print(f"Cameras: L={self.cam_left_id} R={self.cam_right_id}")
        _print(f"Scale: {self.camera_scale_to_robot_units} (camera units -> robot mm)")
        _print(f"Output: {self.output_file}")
        _print("")
        _print("WORKFLOW:")
        _print("  1. Place ArUco marker flat on table, visible to BOTH cameras")
        _print("  2. Press SPACE to freeze + detect")
        _print("  3. Enter robot XYZ of RED corner, width axis, height axis in terminal")
        _print("  4. Move marker to new position, repeat")
        _print(f"  5. Press 's' to solve (recommend >= {self.required_placements} placements)")
        _print("")
        _print("Controls: SPACE freeze/detect | s solve | d delete-last | q quit")
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

                if self._pending_detection:
                    cv2.waitKey(1)
                    self._process_detection()

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                elif key == ord(" "):
                    if not self.frozen:
                        self.frozen = True
                        _print("[FROZEN] Detecting ArUco marker...")
                        corners_l = self._detect_marker(self.latest_left)
                        corners_r = self._detect_marker(self.latest_right)

                        if corners_l is not None and corners_r is not None:
                            self.detected_left = corners_l
                            self.detected_right = corners_r
                            self._pending_detection = True
                            _print(f"  Found marker ID {self.marker_id} in both images!")
                        else:
                            self.detected_left = None
                            self.detected_right = None
                            status = f"L={'OK' if corners_l is not None else 'FAIL'} " \
                                     f"R={'OK' if corners_r is not None else 'FAIL'}"
                            _print(f"  Detection failed ({status}). "
                                   f"Ensure marker is visible in BOTH cameras.")
                            _print("  Press SPACE to unfreeze and try again.")
                    else:
                        self.frozen = False
                        self.detected_left = None
                        self.detected_right = None
                        _print("[LIVE]")

                elif key == ord("s"):
                    if len(self.camera_points_scaled) < 2:
                        _print(f"Need at least 2 pairs to solve "
                               f"(have {len(self.camera_points_scaled)}).")
                    else:
                        result = self._compute_transform()
                        self._print_transform_summary(result)
                        self._save_transform(result)
                        _print("\nContinue adding placements to improve, or 'q' to quit.")

                elif key == ord("d"):
                    if self.placement_count > 0 and self.camera_points_scaled:
                        to_remove = min(4, len(self.camera_points_scaled))
                        for _ in range(to_remove):
                            self.camera_points_raw.pop()
                            self.camera_points_scaled.pop()
                            self.robot_points.pop()
                            self.reproj_errors.pop()
                        self.placement_count = max(0, self.placement_count - 1)
                        self.transform_result = None
                        _print(f"  Removed {to_remove} pairs. "
                               f"Remaining: {len(self.camera_points_scaled)} pairs "
                               f"({self.placement_count} placements)")
                    else:
                        _print("Nothing to delete.")

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

        if self.transform_result is None and len(self.camera_points_scaled) >= 2:
            _print("\nComputing final transform...")
            result = self._compute_transform()
            self._print_transform_summary(result)
            self._save_transform(result)

        if self.transform_result is None:
            _print("Not enough pairs to solve transform (need >= 2).")
            exit_code = 1

        return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find camera->robot transform using ArUco marker detection."
    )
    parser.add_argument("--marker-size", type=float, default=180.0,
                        help="ArUco marker side length in mm (default: 150)")
    parser.add_argument("--marker-id", type=int, default=0,
                        help="ArUco marker ID to detect (default: 0)")
    parser.add_argument("--camera-scale-to-robot-units", type=float, default=10.0,
                        help="Scale for camera units -> robot mm (default: 10.0 for cm->mm)")
    parser.add_argument("--required-placements", type=int, default=3,
                        help="Recommended minimum placements (default: 3)")
    parser.add_argument("--output-file", type=str,
                        default=DEFAULT_POINTS_BASED_TRANSFORM_FILE,
                        help="Path to save transform JSON.")
    args = parser.parse_args()

    app = ArucoTransformFinder(
        marker_size_mm=args.marker_size,
        marker_id=args.marker_id,
        camera_scale_to_robot_units=args.camera_scale_to_robot_units,
        required_placements=args.required_placements,
        output_file=args.output_file,
    )
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
