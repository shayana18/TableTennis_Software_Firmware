"""
Find camera->robot transform from clicked stereo correspondences.

This script uses the same click-style workflow as test_triangulation_verify:
- freeze frame
- click marker on LEFT image
- click same marker on RIGHT image
- triangulate 3D camera point
- enter matching robot-frame point from your robot controller

Then, after enough matched pairs are collected, it computes:
    p_robot = R @ p_camera + t
using `comm_function.points_based_transform.points_based_transform`.

How to run:
    python scripts/test_find_points_based_transform.py --required-points 8

Steps to find the transformation matrix:
1) Mount a visible marker on a rigid, known robot point (for example paddle tip).
2) Move robot to a static pose, keep marker still.
3) Press SPACE to freeze image.
4) Click marker on LEFT image, then click same marker on RIGHT image.
5) Enter robot XYZ for that same marker point when prompted in terminal.
6) Press SPACE to unfreeze, move to next pose, repeat.
7) Collect points across the workspace (corners + middle are best).
8) When required points are collected, script prints R and t.

Notes:
- One frame per point is allowed (this script does that by default).
- For less noise, repeat the same pose and keep the best reprojection error.
- Triangulator outputs camera points in cm; robot is often in mm.
  Use --camera-scale-to-robot-units 10.0 (default) for cm->mm.
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


class PointsBasedTransformFinder:
    def __init__(
        self,
        required_points: int = 8,
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

        # Pair-click state for one correspondence: left then right.
        self.click_stage = 0
        self.click_points: list[tuple[int, int]] = []

        # Collected correspondences.
        self.camera_points_raw: list[tuple[float, float, float]] = []
        self.camera_points_scaled: list[tuple[float, float, float]] = []
        self.robot_points: list[tuple[float, float, float]] = []
        self.reproj_errors: list[float] = []

        self.window_name = "Find Points-Based Transform"
        self.transform_result = None
        self._last_status_print = 0.0

    def check_calibration(self) -> bool:
        required = [
            "camera0_intrinsics.dat",
            "camera1_intrinsics.dat",
            "camera0_rot_trans.dat",
            "camera1_rot_trans.dat",
        ]
        missing = [
            f for f in required if not os.path.exists(os.path.join(self.calibration_dir, f))
        ]
        if missing:
            _print("ERROR: Missing calibration files:")
            for f in missing:
                _print(f"  - {f}")
            return False
        return True

    def pixel_to_original(self, x_display: int, y_display: int, is_right: bool) -> tuple[int, int]:
        if is_right:
            x_display = x_display - self.display_width
        scale_x = self.frame_width / self.display_width
        scale_y = self.frame_height / self.display_height
        return int(x_display * scale_x), int(y_display * scale_y)

    def _triangulate_click_pair(
        self, pt_left: tuple[int, int], pt_right: tuple[int, int]
    ) -> tuple[Optional[tuple[float, float, float]], Optional[float]]:
        tri = self.triangulator
        pt_l_rect = tri._rectify_point(pt_left, tri.cmtx0, tri.dist0, tri.R_rect0, tri.P_rect0)
        pt_r_rect = tri._rectify_point(pt_right, tri.cmtx1, tri.dist1, tri.R_rect1, tri.P_rect1)

        disp = pt_l_rect[0] - pt_r_rect[0]
        if disp <= 0:
            _print("ERROR: Negative disparity. Click left then right on the same marker.")
            return None, None

        pos = tri.triangulate(pt_l_rect, pt_r_rect)
        if pos[2] <= 0:
            _print("ERROR: Point behind camera.")
            return None, None

        err_l, err_r = tri._reprojection_error(pos, pt_l_rect, pt_r_rect)
        reproj = max(err_l, err_r)
        return (float(pos[0]), float(pos[1]), float(pos[2])), float(reproj)

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
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                _print("Invalid numbers. Try again.")
                continue
            return (x, y, z)

    def _reset_current_click_pair(self) -> None:
        self.click_stage = 0
        self.click_points = []

    def _add_correspondence(
        self,
        cam_raw: tuple[float, float, float],
        reproj_err: float,
        robot_xyz: tuple[float, float, float],
    ) -> None:
        scale = self.camera_scale_to_robot_units
        cam_scaled = (cam_raw[0] * scale, cam_raw[1] * scale, cam_raw[2] * scale)

        self.camera_points_raw.append(cam_raw)
        self.camera_points_scaled.append(cam_scaled)
        self.robot_points.append(robot_xyz)
        self.reproj_errors.append(reproj_err)

        idx = len(self.camera_points_raw)
        _print(
            f"[PAIR {idx}/{self.required_points}] "
            f"cam_raw=({cam_raw[0]:+.2f}, {cam_raw[1]:+.2f}, {cam_raw[2]:+.2f}) "
            f"cam_scaled=({cam_scaled[0]:+.2f}, {cam_scaled[1]:+.2f}, {cam_scaled[2]:+.2f}) "
            f"robot=({robot_xyz[0]:+.2f}, {robot_xyz[1]:+.2f}, {robot_xyz[2]:+.2f}) "
            f"reproj={reproj_err:.2f}px"
        )

    def _compute_transform(self) -> dict:
        cam = np.asarray(self.camera_points_scaled, dtype=float)
        rob = np.asarray(self.robot_points, dtype=float)

        R, t, diag = points_based_transform(cam, rob, return_diagnostics=True)
        fitted = transform_points(cam, R, t)
        residuals = rob - fitted
        per_point_err = np.linalg.norm(residuals, axis=1)

        result = {
            "R": R,
            "t": t,
            "diagnostics": diag,
            "residuals": residuals,
            "per_point_err": per_point_err,
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
        _print(" POINTS-BASED CAMERA->ROBOT TRANSFORM ")
        _print("=" * 74)
        _print(f"Pairs used: {diag['num_points']}")
        _print(f"Camera scale used before fit: {self.camera_scale_to_robot_units:.6f}")
        _print("Equation:")
        _print("  p_robot = R @ p_camera_scaled + t")
        _print("")
        _print("R =")
        _print(np.array2string(R, precision=8, suppress_small=False))
        _print("")
        _print("t =")
        _print(np.array2string(t, precision=8, suppress_small=False))
        _print("")
        _print(
            f"Fit quality: rmse={diag['rmse']:.4f}, mean={diag['mean_error']:.4f}, "
            f"max={diag['max_error']:.4f}, det(R)={diag['rotation_det']:.6f}"
        )
        _print(f"Per-point error: {np.array2string(per_point_err, precision=4)}")
        _print("=" * 74)

    def _save_transform(self, result: dict) -> None:
        path = save_points_based_transform(
            rotation=result["R"],
            translation=result["t"],
            output_path=self.output_file,
            camera_scale_to_robot_units=self.camera_scale_to_robot_units,
        )
        _print(f"Saved points-based transform file: {path}")

    def on_mouse(self, event, x, y, _flags, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.latest_left is None or self.latest_right is None:
            return

        is_right = x >= self.display_width
        orig_x, orig_y = self.pixel_to_original(x, y, is_right=is_right)

        if self.click_stage == 0:
            if is_right:
                return
            self.click_points = [(orig_x, orig_y)]
            self.click_stage = 1
            _print(f"LEFT click: ({orig_x}, {orig_y})")
            return

        if self.click_stage == 1:
            if not is_right:
                return
            self.click_points.append((orig_x, orig_y))
            self.click_stage = 2
            _print(f"RIGHT click: ({orig_x}, {orig_y})")

            cam_raw, reproj = self._triangulate_click_pair(self.click_points[0], self.click_points[1])
            if cam_raw is None:
                self._reset_current_click_pair()
                return

            robot_xyz = self._prompt_robot_point()
            if robot_xyz is None:
                _print("Pair skipped.")
                self._reset_current_click_pair()
                return

            self._add_correspondence(cam_raw, reproj, robot_xyz)
            self._reset_current_click_pair()

            if len(self.camera_points_scaled) >= self.required_points:
                result = self._compute_transform()
                self._print_transform_summary(result)
                self._save_transform(result)
            return

    def _draw_overlay(self, left_small, right_small) -> None:
        h = self.display_height
        scale_x = self.display_width / self.frame_width
        scale_y = self.display_height / self.frame_height

        # Header
        for img, name in ((left_small, "LEFT"), (right_small, "RIGHT")):
            cv2.rectangle(img, (0, 0), (430, 50), (0, 0, 0), -1)
            cv2.putText(
                img,
                f"{name} {'[FROZEN]' if self.frozen else '[LIVE]'}",
                (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                img,
                f"Pairs: {len(self.camera_points_scaled)}/{self.required_points}",
                (8, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 255, 255),
                1,
            )

        # Stage guidance
        if self.click_stage == 0:
            cv2.rectangle(
                left_small,
                (0, 0),
                (self.display_width - 1, self.display_height - 1),
                (0, 255, 255),
                2,
            )
            cv2.putText(
                left_small,
                "Click marker on LEFT",
                (10, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
        elif self.click_stage == 1:
            cv2.rectangle(
                right_small,
                (0, 0),
                (self.display_width - 1, self.display_height - 1),
                (0, 255, 255),
                2,
            )
            cv2.putText(
                right_small,
                "Click SAME marker on RIGHT",
                (10, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

        # Draw current clicked points
        panels = [left_small, right_small]
        for i, pt in enumerate(self.click_points[:2]):
            img = panels[i]
            px = int(pt[0] * scale_x)
            py = int(pt[1] * scale_y)
            cv2.circle(img, (px, py), 16, (255, 255, 0), 2)
            cv2.drawMarker(img, (px, py), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)

        # Controls footer
        cv2.putText(
            right_small,
            "SPACE freeze | r reset pair | d delete last | q quit",
            (10, self.display_height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (190, 190, 190),
            1,
        )

    def run(self) -> int:
        _print("\n" + "=" * 74)
        _print(" FIND POINTS-BASED TRANSFORM (CLICK WORKFLOW) ")
        _print("=" * 74)
        _print(f"Cameras: L={self.cam_left_id} R={self.cam_right_id}")
        _print(f"Required pairs: {self.required_points}")
        _print(f"Camera scale to robot units: {self.camera_scale_to_robot_units}")
        _print(f"Output transform file: {self.output_file}")
        _print("Controls: SPACE freeze/unfreeze, r reset pair, d delete last pair, q quit")
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

                # Throttled status print while collecting.
                now = time.time()
                if (
                    now - self._last_status_print > 8.0
                    and len(self.camera_points_scaled) < self.required_points
                ):
                    _print(
                        f"Progress: {len(self.camera_points_scaled)}/{self.required_points} pairs collected"
                    )
                    self._last_status_print = now

                # Auto-finish once enough pairs collected.
                if (
                    len(self.camera_points_scaled) >= self.required_points
                    and self.transform_result is not None
                ):
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
                        _print("[FROZEN]")
                    else:
                        self.frozen = False
                        self.frozen_left = None
                        self.frozen_right = None
                        _print("[LIVE]")
                elif key == ord("r"):
                    self._reset_current_click_pair()
                    _print("Current click pair reset.")
                elif key == ord("d"):
                    if self.camera_points_scaled:
                        self.camera_points_scaled.pop()
                        self.camera_points_raw.pop()
                        self.robot_points.pop()
                        self.reproj_errors.pop()
                        self.transform_result = None
                        _print(
                            f"Deleted last pair. Remaining: {len(self.camera_points_scaled)}"
                        )
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

        if self.transform_result is None:
            if len(self.camera_points_scaled) >= 3:
                _print(
                    "Required count not reached, but at least 3 pairs exist. "
                    "Computing partial transform..."
                )
                result = self._compute_transform()
                self._print_transform_summary(result)
                self._save_transform(result)
            else:
                _print("Not enough pairs to solve transform (need >= 3).")
                exit_code = 1

        return exit_code


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find camera->robot transform from clicked stereo correspondences."
    )
    parser.add_argument(
        "--required-points",
        type=int,
        default=8,
        help="Number of matched pairs to collect before solving (recommended 8-15).",
    )
    parser.add_argument(
        "--camera-scale-to-robot-units",
        type=float,
        default=10.0,
        help=(
            "Scale applied to triangulated camera points before fitting. "
            "Use 10.0 for camera cm -> robot mm. Use 1.0 if both already same units."
        ),
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=DEFAULT_POINTS_BASED_TRANSFORM_FILE,
        help="Path to write solved points-based transform JSON.",
    )
    args = parser.parse_args()
    if args.required_points < 3:
        parser.error("--required-points must be >= 3")
    return args


def main() -> int:
    args = _parse_args()
    app = PointsBasedTransformFinder(
        required_points=args.required_points,
        camera_scale_to_robot_units=args.camera_scale_to_robot_units,
        output_file=args.output_file,
    )
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
