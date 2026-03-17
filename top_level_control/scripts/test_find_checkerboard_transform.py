"""
Find camera->robot transform using automatic checkerboard corner detection.

Same output as test_find_points_based_transform.py (Kabsch R, t saved to JSON),
but uses cv2.findChessboardCorners + cornerSubPix instead of manual clicking.

A single checkerboard placement yields (rows x cols) correspondence pairs
with sub-pixel accuracy, compared to one pair per manual click.

WORKFLOW:
  1. Place checkerboard flat on the table, visible to BOTH cameras.
  2. Press SPACE to freeze and auto-detect corners.
  3. Script marks corner[0] with a RED circle, shows row/col direction arrows.
  4. Enter the robot-frame (x, y, z) of the RED corner (corner[0]).
  5. Enter which robot axis the ROW direction (red arrow) corresponds to.
  6. Enter which robot axis the COL direction (blue arrow) corresponds to.
  7. Script triangulates all corners and computes robot coords from geometry.
  8. Press SPACE to unfreeze, move board to another position, repeat.
  9. Press 's' to solve the transform, or 'q' to quit.

WHAT TO MEASURE:
  - The robot-frame (x, y, z) of corner[0] (the RED dot). Use a ruler or
    the robot's known geometry. For example, if the robot home is (0, 0, -900mm)
    and the board is 500mm to the right and 200mm forward, corner[0] might be
    (500, -200, -tableZ).
  - Which way the board's row direction (RED arrow, corner[0]->corner[1])
    points in robot frame: e.g. "+x" means increasing robot X.
  - Which way the board's col direction (BLUE arrow, corner[0]->next row)
    points in robot frame: e.g. "+y" means increasing robot Y.

DEFAULTS: 7x4 inner corners, 31.8mm squares (matches calibration board).

How to run:
    python scripts/test_find_checkerboard_transform.py
    python scripts/test_find_checkerboard_transform.py --square-size 31.8 --cols 7 --rows 4
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


AXIS_MAP = {
    "+x": np.array([1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
    "+z": np.array([0.0, 0.0, 1.0]),
    "-z": np.array([0.0, 0.0, -1.0]),
}


class CheckerboardTransformFinder:
    def __init__(
        self,
        pattern_cols: int = 7,
        pattern_rows: int = 4,
        square_size_mm: float = 31.8,
        camera_scale_to_robot_units: float = 10.0,
        required_placements: int = 2,
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

        self.pattern_cols = int(pattern_cols)
        self.pattern_rows = int(pattern_rows)
        self.pattern_size = (self.pattern_cols, self.pattern_rows)
        self.total_corners = self.pattern_cols * self.pattern_rows
        self.square_size_mm = float(square_size_mm)
        self.camera_scale_to_robot_units = float(camera_scale_to_robot_units)
        self.required_placements = int(required_placements)
        self.output_file = os.path.abspath(output_file)

        self.triangulator: Optional[StereoTriangulator] = None
        self.latest_left = None
        self.latest_right = None
        self.frozen = False

        # Detection state
        self.detected_left = None   # corners array from left image
        self.detected_right = None  # corners array from right image
        self._pending_detection = False

        # Collected correspondences (accumulated across placements)
        self.camera_points_raw: list[tuple[float, float, float]] = []
        self.camera_points_scaled: list[tuple[float, float, float]] = []
        self.robot_points: list[tuple[float, float, float]] = []
        self.reproj_errors: list[float] = []
        self.placement_count = 0

        self.window_name = "Checkerboard Transform Finder"
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

    # --- Corner detection ---

    def _detect_corners(self, frame):
        """Detect checkerboard corners with sub-pixel refinement."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, flags)
        if ret and corners is not None:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return ret, corners

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
        """Triangulate all matched corner pairs. Returns list of (pos_3d, reproj_err)."""
        results = []
        for i in range(len(left_corners)):
            pt_l = (float(left_corners[i][0][0]), float(left_corners[i][0][1]))
            pt_r = (float(right_corners[i][0][0]), float(right_corners[i][0][1]))
            pos, reproj = self._triangulate_corner_pair(pt_l, pt_r)
            results.append((pos, reproj))
        return results

    # --- Robot position computation ---

    def _compute_robot_grid(self, origin, row_dir, col_dir):
        """Compute robot-frame positions for all corners from board geometry.

        Corner layout (OpenCV ordering):
          corner[row * cols + col]
          Row 0: [0, 1, 2, ..., cols-1]       <- row direction (corner[0] -> corner[1])
          Row 1: [cols, cols+1, ...]           <- col direction (corner[0] -> corner[cols])
        """
        positions = []
        for r in range(self.pattern_rows):
            for c in range(self.pattern_cols):
                p = origin + c * self.square_size_mm * row_dir + r * self.square_size_mm * col_dir
                positions.append((float(p[0]), float(p[1]), float(p[2])))
        return positions

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
        """Process detected corners: triangulate, prompt user, add pairs."""
        self._pending_detection = False

        if self.detected_left is None or self.detected_right is None:
            _print("No corners detected. Try repositioning the board.")
            return

        n = len(self.detected_left)
        _print(f"\nDetected {n} corners in both images.")

        # Triangulate all corners
        tri_results = self._triangulate_all_corners(self.detected_left, self.detected_right)

        # Check how many triangulated successfully
        valid = [(i, pos, err) for i, (pos, err) in enumerate(tri_results) if pos is not None]
        failed = n - len(valid)
        if failed > 0:
            _print(f"  WARNING: {failed}/{n} corners failed to triangulate.")
        if len(valid) < 3:
            _print("  Too few valid corners. Reposition the board and try again.")
            return

        # Report reproj errors
        reproj_errs = [err for _, _, err in valid]
        _print(f"  Reproj errors: mean={np.mean(reproj_errs):.3f}px, "
               f"max={np.max(reproj_errs):.3f}px, "
               f"median={np.median(reproj_errs):.3f}px")

        high_err = [i for i, _, err in valid if err > 2.0]
        if high_err:
            _print(f"  WARNING: {len(high_err)} corners have reproj > 2px.")

        # --- Prompt user for board pose in robot frame ---
        origin_xyz = self._prompt_xyz(
            "Enter robot-frame (x y z) in mm of the RED corner (corner[0]):"
        )
        if origin_xyz is None:
            _print("Placement cancelled.")
            return
        origin = np.array(origin_xyz)

        row_dir = self._prompt_axis(
            "Row direction (RED arrow, corner[0] -> corner[1]) in robot frame:"
        )
        if row_dir is None:
            _print("Placement cancelled.")
            return

        col_dir = self._prompt_axis(
            "Col direction (BLUE arrow, corner[0] -> next row) in robot frame:"
        )
        if col_dir is None:
            _print("Placement cancelled.")
            return

        # Validate axes are orthogonal
        dot = float(np.dot(row_dir, col_dir))
        if abs(dot) > 0.01:
            _print(f"  WARNING: Row and col axes are not orthogonal (dot={dot:.3f}).")
            _print("  This usually means the board isn't axis-aligned. Proceed anyway.")

        # Compute robot positions for all corners
        robot_grid = self._compute_robot_grid(origin, row_dir, col_dir)

        # Add valid correspondences
        scale = self.camera_scale_to_robot_units
        added = 0
        max_reproj = 3.0  # reject corners with reproj > this
        for idx, cam_raw, reproj in valid:
            if reproj > max_reproj:
                continue
            cam_scaled = (cam_raw[0] * scale, cam_raw[1] * scale, cam_raw[2] * scale)
            robot_xyz = robot_grid[idx]

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
        _print("  Press SPACE to unfreeze and reposition board, or 's' to solve.\n")

    # --- Transform computation (reused from existing script) ---

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
                    fitted2 = transform_points(cam[keep], R2, t2)
                    residuals2 = rob[keep] - fitted2
                    per_point_err2 = np.linalg.norm(residuals2, axis=1)
                    _print(f"  Re-solved with {int(keep.sum())} points: "
                           f"rmse={diag2['rmse']:.4f} (was {diag['rmse']:.4f})")

                    keep_list = list(np.where(keep)[0])
                    self.camera_points_raw = [self.camera_points_raw[i] for i in keep_list]
                    self.camera_points_scaled = [self.camera_points_scaled[i] for i in keep_list]
                    self.robot_points = [self.robot_points[i] for i in keep_list]
                    self.reproj_errors = [self.reproj_errors[i] for i in keep_list]
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
        _print(" POINTS-BASED CAMERA->ROBOT TRANSFORM (CHECKERBOARD) ")
        _print("=" * 74)
        _print(f"Pairs used: {diag['num_points']}  ({self.placement_count} placements)")
        _print(f"Camera scale: {self.camera_scale_to_robot_units:.6f}")
        _print(f"Board: {self.pattern_cols}x{self.pattern_rows} inner corners, "
               f"{self.square_size_mm:.1f}mm squares")
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

        # Per-placement breakdown
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

        # Draw detected corners on left image
        if self.frozen and self.detected_left is not None:
            corners_disp = self.detected_left.copy()
            corners_disp[:, 0, 0] *= scale_x
            corners_disp[:, 0, 1] *= scale_y
            cv2.drawChessboardCorners(left_small, self.pattern_size, corners_disp, True)

            # Mark corner[0] with big RED circle + label
            c0 = corners_disp[0][0]
            px0, py0 = int(c0[0]), int(c0[1])
            cv2.circle(left_small, (px0, py0), 12, (0, 0, 255), 3)
            cv2.putText(left_small, "0", (px0 + 16, py0 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Row direction arrow (corner[0] -> corner[1]) in RED
            if len(corners_disp) > 1:
                c1 = corners_disp[1][0]
                px1, py1 = int(c1[0]), int(c1[1])
                cv2.arrowedLine(left_small, (px0, py0), (px1, py1), (0, 0, 255), 2,
                                tipLength=0.3)
                cv2.putText(left_small, "ROW", (px1 + 8, py1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

            # Col direction arrow (corner[0] -> corner[cols]) in BLUE
            if len(corners_disp) > self.pattern_cols:
                cc = corners_disp[self.pattern_cols][0]
                pxc, pyc = int(cc[0]), int(cc[1])
                cv2.arrowedLine(left_small, (px0, py0), (pxc, pyc), (255, 0, 0), 2,
                                tipLength=0.3)
                cv2.putText(left_small, "COL", (pxc + 8, pyc - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)

        # Draw detected corners on right image
        if self.frozen and self.detected_right is not None:
            corners_disp_r = self.detected_right.copy()
            corners_disp_r[:, 0, 0] *= scale_x
            corners_disp_r[:, 0, 1] *= scale_y
            cv2.drawChessboardCorners(right_small, self.pattern_size, corners_disp_r, True)

        # Guidance text
        if not self.frozen:
            cv2.putText(left_small, "SPACE to freeze + detect",
                        (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        cv2.putText(right_small,
                    "SPACE freeze | s solve | d del-last | q quit",
                    (10, self.display_height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (190, 190, 190), 1)

    # --- Main loop ---

    def run(self) -> int:
        _print("\n" + "=" * 74)
        _print(" FIND TRANSFORM (CHECKERBOARD AUTO-DETECTION) ")
        _print("=" * 74)
        _print(f"Board: {self.pattern_cols}x{self.pattern_rows} inner corners, "
               f"{self.square_size_mm:.1f}mm squares")
        _print(f"Cameras: L={self.cam_left_id} R={self.cam_right_id}")
        _print(f"Scale: {self.camera_scale_to_robot_units} (camera units -> robot mm)")
        _print(f"Output: {self.output_file}")
        _print("")
        _print("WORKFLOW:")
        _print("  1. Place checkerboard flat on table, visible to BOTH cameras")
        _print("  2. Press SPACE to freeze + detect corners")
        _print("  3. Enter robot XYZ of RED corner, row axis, col axis in terminal")
        _print("  4. Move board to new position, repeat")
        _print(f"  5. Press 's' to solve (recommend >= {self.required_placements} placements)")
        _print("")
        _print("Controls: SPACE freeze/detect | s solve | d delete-last-placement | q quit")
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

                # Process pending detection (after overlay draws so user sees corners)
                if self._pending_detection:
                    cv2.waitKey(1)
                    self._process_detection()

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                elif key == ord(" "):
                    if not self.frozen:
                        # Freeze and detect
                        self.frozen = True
                        _print("[FROZEN] Detecting corners...")
                        ret_l, corners_l = self._detect_corners(self.latest_left)
                        ret_r, corners_r = self._detect_corners(self.latest_right)

                        if ret_l and ret_r:
                            self.detected_left = corners_l
                            self.detected_right = corners_r
                            self._pending_detection = True
                            _print(f"  Found {len(corners_l)} corners in both images!")
                        else:
                            self.detected_left = None
                            self.detected_right = None
                            status = f"L={'OK' if ret_l else 'FAIL'} R={'OK' if ret_r else 'FAIL'}"
                            _print(f"  Detection failed ({status}). "
                                   f"Ensure full board is visible in BOTH cameras.")
                            _print("  Press SPACE to unfreeze and try again.")
                    else:
                        # Unfreeze
                        self.frozen = False
                        self.detected_left = None
                        self.detected_right = None
                        _print("[LIVE]")

                elif key == ord("s"):
                    if len(self.camera_points_scaled) < 3:
                        _print(f"Need at least 3 pairs to solve (have {len(self.camera_points_scaled)}).")
                    else:
                        result = self._compute_transform()
                        self._print_transform_summary(result)
                        self._save_transform(result)
                        _print("\nContinue adding placements to improve, or 'q' to quit.")

                elif key == ord("d"):
                    if self.placement_count > 0 and self.camera_points_scaled:
                        # Remove the last placement's worth of points
                        # We track how many points each placement added
                        _print("Deleting last placement's points...")
                        # Simple approach: we don't track per-placement count,
                        # so just remove last total_corners worth
                        to_remove = min(self.total_corners, len(self.camera_points_scaled))
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

        # Final solve if we have enough points
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
        description="Find camera->robot transform using checkerboard auto-detection."
    )
    parser.add_argument("--cols", type=int, default=7,
                        help="Number of inner corners along board columns (default: 7)")
    parser.add_argument("--rows", type=int, default=4,
                        help="Number of inner corners along board rows (default: 4)")
    parser.add_argument("--square-size", type=float, default=31.8,
                        help="Square size in mm (default: 31.8)")
    parser.add_argument("--camera-scale-to-robot-units", type=float, default=10.0,
                        help="Scale for camera units -> robot mm (default: 10.0 for cm->mm)")
    parser.add_argument("--required-placements", type=int, default=2,
                        help="Recommended minimum placements (default: 2)")
    parser.add_argument("--output-file", type=str,
                        default=DEFAULT_POINTS_BASED_TRANSFORM_FILE,
                        help="Path to save transform JSON.")
    args = parser.parse_args()

    app = CheckerboardTransformFinder(
        pattern_cols=args.cols,
        pattern_rows=args.rows,
        square_size_mm=args.square_size,
        camera_scale_to_robot_units=args.camera_scale_to_robot_units,
        required_placements=args.required_placements,
        output_file=args.output_file,
    )
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
