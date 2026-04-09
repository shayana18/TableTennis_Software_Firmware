"""
Find camera->robot transform from fixed ArUco markers.

This method uses all visible fixed markers in one capture to solve a single
rigid transform. It avoids manually averaging many placements and instead uses
known robot-frame geometry of the markers.

Assumptions:
  - Markers are visible in BOTH cameras at solve time.
  - Markers share the same orientation (same width/height axis directions).
  - You know robot-frame (x,y,z) of corner[0] (top-left) for each marker.
  - Marker side length is known in mm.

ArUco corner ordering (OpenCV):
  [0] top-left, [1] top-right, [2] bottom-right, [3] bottom-left
  width  direction = corner[0] -> corner[1]
  height direction = corner[0] -> corner[3]

Run examples:
  python scripts/test_find_dual_aruco_transform.py

  # Default: 2-marker mode (ID 0 + ID 1)
  python scripts/test_find_dual_aruco_transform.py

  # Optional: include a 3rd marker
  python scripts/test_find_dual_aruco_transform.py ^
      --marker2-id 3 --marker2-origin "150 1400 -900"
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import cv2
import numpy as np
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
CAMERA_PARAMS_DIR = os.path.join(PARENT_DIR, "camera_params")
CAMERA_PROPERTIES_DIR = os.path.join(CAMERA_PARAMS_DIR, "camera_properties")
CALIBRATION_SETTINGS_PATH = os.path.join(CAMERA_PARAMS_DIR, "calibration_settings.yaml")
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from comm_functions.points_based_transform import (
    DEFAULT_POINTS_BASED_TRANSFORM_FILE,
    points_based_transform,
    save_points_based_transform,
    transform_points,
)
from ball_tracking.stereo_triangulator import StereoTriangulator


def _print(*args, **kwargs) -> None:
    print("from planner in terminal", *args, **kwargs)


def load_camera_settings(settings_path: str = CALIBRATION_SETTINGS_PATH) -> dict:
    with open(settings_path, "r", encoding="utf-8") as infile:
        settings = yaml.safe_load(infile)

    required = ("camera0", "camera1", "frame_width", "frame_height")
    missing = [key for key in required if key not in settings]
    if missing:
        raise KeyError(f"Missing keys in {settings_path}: {missing}")

    return settings


AXIS_MAP = {
    "+x": np.array([1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
    "+z": np.array([0.0, 0.0, 1.0]),
    "-z": np.array([0.0, 0.0, -1.0]),
}

# ============================================================
# MARKER CONFIGURATION — Edit these to match your setup
# ============================================================
# Map each ArUco ID to its corner[0] (top-left) robot-frame (x, y, z) in mm.
# All markers must share the same width/height axis orientation.
MARKERS = {
    0: (-769.7, 1559.6, -858.44),       # ID 0 — fill in measured (x, y, z)
    1: (-769.7, 1776.5, -858.44),       # ID 1 — fill in measured (x, y, z)
    2: (-769.7, 1991.4, -858.44),       # ID 2 — fill in measured (x, y, z)
    3: (-769.7,2207.3, -858.44),       # ID 3 — fill in measured (x, y, z)
}
MARKER_SIZE_MM = 195.0
DEFAULT_WIDTH_AXIS = "+y"
DEFAULT_HEIGHT_AXIS = "-z"
# ============================================================


def _parse_xyz_arg(text: Optional[str]) -> Optional[tuple[float, float, float]]:
    if text is None:
        return None
    parts = text.replace(",", " ").split()
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Expected 3 values for xyz, got: {text!r}"
        )
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid xyz values: {text!r}") from exc


class DualArucoTransformFinder:
    def __init__(
        self,
        markers: dict[int, tuple[float, float, float]] = None,
        marker_size_mm: float = MARKER_SIZE_MM,
        width_axis: str = DEFAULT_WIDTH_AXIS,
        height_axis: str = DEFAULT_HEIGHT_AXIS,
        camera_scale_to_robot_units: float = 10.0,
        max_reproj_px: float = 3.0,
        output_file: str = DEFAULT_POINTS_BASED_TRANSFORM_FILE,
    ) -> None:
        self.base_dir = PARENT_DIR
        self.calibration_dir = CAMERA_PROPERTIES_DIR

        cam_settings = load_camera_settings()
        self.frame_width = cam_settings["frame_width"]
        self.frame_height = cam_settings["frame_height"]
        self.cam_left_id = cam_settings["camera0"]
        self.cam_right_id = cam_settings["camera1"]

        self.display_width = 640
        self.display_height = int(self.display_width * self.frame_height / self.frame_width)

        # Marker config: {aruco_id: (x, y, z)} in robot frame mm
        self.markers = dict(markers) if markers is not None else dict(MARKERS)
        self.marker_size_mm = float(marker_size_mm)
        self.width_axis_name = width_axis.strip().lower()
        self.height_axis_name = height_axis.strip().lower()
        self.camera_scale_to_robot_units = float(camera_scale_to_robot_units)
        self.max_reproj_px = float(max_reproj_px)
        self.output_file = os.path.abspath(output_file)

        if self.width_axis_name not in AXIS_MAP or self.height_axis_name not in AXIS_MAP:
            raise ValueError("Invalid axis name. Use one of: +x, -x, +y, -y, +z, -z")
        self.width_axis_vec = AXIS_MAP[self.width_axis_name]
        self.height_axis_vec = AXIS_MAP[self.height_axis_name]

        # Keep dictionary same as existing script behavior.
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self.triangulator: Optional[StereoTriangulator] = None
        self.latest_left = None
        self.latest_right = None
        self.frozen = False
        self.window_name = "Multi ArUco Transform Finder"

    def check_calibration(self) -> bool:
        required = [
            "camera0_intrinsics.dat",
            "camera1_intrinsics.dat",
            "camera0_rot_trans.dat",
            "camera1_rot_trans.dat",
        ]
        missing = [f for f in required if not os.path.exists(os.path.join(self.calibration_dir, f))]
        if missing:
            _print("ERROR: Missing calibration files:")
            for f in missing:
                _print(f"  - {f}")
            return False
        return True

    def _detect_markers(self, frame) -> dict[int, np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners_list, ids, _rejected = self.aruco_detector.detectMarkers(gray)

        found: dict[int, np.ndarray] = {}
        if ids is None or len(ids) == 0:
            return found

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        for i, marker_id in enumerate(ids.flatten()):
            corners = corners_list[i].reshape(4, 2)
            for j in range(4):
                pt = np.array([[corners[j]]], dtype=np.float32)
                refined = cv2.cornerSubPix(gray, pt, (5, 5), (-1, -1), criteria)
                corners[j] = refined[0, 0]
            found[int(marker_id)] = corners
        return found

    def _triangulate_corner_pair(self, pt_left, pt_right):
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

    def _triangulate_marker(self, corners_left: np.ndarray, corners_right: np.ndarray):
        results = []
        for i in range(4):
            pt_l = (float(corners_left[i][0]), float(corners_left[i][1]))
            pt_r = (float(corners_right[i][0]), float(corners_right[i][1]))
            pos, reproj = self._triangulate_corner_pair(pt_l, pt_r)
            results.append((pos, reproj))
        return results

    def _compute_robot_corners(self, origin_xyz: tuple[float, float, float]):
        origin = np.array(origin_xyz, dtype=float)
        s = self.marker_size_mm
        positions = [
            origin,
            origin + s * self.width_axis_vec,
            origin + s * self.width_axis_vec + s * self.height_axis_vec,
            origin + s * self.height_axis_vec,
        ]
        return [(float(p[0]), float(p[1]), float(p[2])) for p in positions]

    @staticmethod
    def _prompt_xyz(prompt_text: str) -> tuple[float, float, float]:
        _print(prompt_text)
        while True:
            raw = input("> ").strip().replace(",", " ")
            parts = [p for p in raw.split() if p]
            if len(parts) != 3:
                _print("Enter exactly 3 numbers: x y z")
                continue
            try:
                return float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                _print("Invalid numbers. Try again.")

    def _active_marker_origins(self) -> dict[int, tuple[float, float, float]]:
        return dict(self.markers)

    def _ensure_fixture_definition(self) -> None:
        for mid in sorted(self.markers.keys()):
            origin = self.markers[mid]
            if origin == (0.0, 0.0, 0.0):
                _print(f"WARNING: Marker ID {mid} origin is (0, 0, 0) — did you forget to measure it?")

        dot = float(np.dot(self.width_axis_vec, self.height_axis_vec))
        if abs(dot) > 1e-6:
            _print(
                f"WARNING: width-axis {self.width_axis_name} and height-axis "
                f"{self.height_axis_name} are not orthogonal (dot={dot:.3f})."
            )

    # Multi-frame averaging
    AVERAGE_FRAMES = 500
    MAX_STD_MM = 2.5  # reject corners with std > this

    def _collect_averaged_correspondences(self):
        """Grab N frames, triangulate all markers in each, reject outliers, average."""
        marker_origins = self._active_marker_origins()
        needed_ids = set(marker_origins.keys())
        n_frames = self.AVERAGE_FRAMES

        # {(marker_id, corner_idx): [list of 3D positions]}
        accum: dict[tuple[int, int], list[np.ndarray]] = {}
        accum_reproj: dict[tuple[int, int], list[float]] = {}
        for mid in needed_ids:
            for cidx in range(4):
                accum[(mid, cidx)] = []
                accum_reproj[(mid, cidx)] = []

        good_frames = 0
        total_frames = 0
        max_attempts = n_frames * 3

        _print(f"\n  Averaging {n_frames} frames (hold markers still)...")

        while good_frames < n_frames and total_frames < max_attempts:
            result = self.triangulator.update()
            if result["left_frame"] is None:
                continue
            total_frames += 1

            left_markers = self._detect_markers(result["left_frame"])
            right_markers = self._detect_markers(result["right_frame"])

            if not needed_ids.issubset(left_markers.keys()):
                continue
            if not needed_ids.issubset(right_markers.keys()):
                continue

            frame_ok = True
            frame_pts: dict[tuple[int, int], tuple] = {}
            frame_reproj: dict[tuple[int, int], float] = {}
            for mid in needed_ids:
                tri_results = self._triangulate_marker(left_markers[mid], right_markers[mid])
                for cidx, (pos, reproj) in enumerate(tri_results):
                    if pos is None or reproj is None or reproj > self.max_reproj_px:
                        frame_ok = False
                        break
                    frame_pts[(mid, cidx)] = pos
                    frame_reproj[(mid, cidx)] = reproj
                if not frame_ok:
                    break

            if not frame_ok:
                continue

            for key, pos in frame_pts.items():
                accum[key].append(np.array(pos))
                accum_reproj[key].append(frame_reproj[key])
            good_frames += 1

            if good_frames % 10 == 0:
                _print(f"    {good_frames}/{n_frames} frames collected...")

            cv2.waitKey(1)

        if good_frames < 5:
            return {
                "ok": False,
                "reason": f"Only {good_frames}/{n_frames} valid frames.",
            }

        _print(f"  Collected {good_frames} frames. Computing averages...")

        camera_raw = []
        camera_scaled = []
        robot_points = []
        reproj_errors = []
        per_marker_counts: dict[int, int] = {}

        for mid, origin_xyz in marker_origins.items():
            if origin_xyz is None:
                return {"ok": False, "reason": f"Missing origin for marker ID {mid}"}

            robot_corners = self._compute_robot_corners(origin_xyz)
            added = 0

            for cidx in range(4):
                key = (mid, cidx)
                pts = accum[key]
                if len(pts) < 5:
                    continue

                pts_arr = np.array(pts)
                std_pos = pts_arr.std(axis=0)
                std_mm = std_pos * self.camera_scale_to_robot_units
                max_std = float(np.max(std_mm))

                # Reject outlier samples (> 2 std from mean) before averaging
                avg_raw = pts_arr.mean(axis=0)
                dists = np.linalg.norm(pts_arr - avg_raw, axis=1)
                threshold = 2.0 * np.std(dists)
                inlier_mask = dists <= max(threshold, 1e-6)
                inlier_pts = pts_arr[inlier_mask]

                if len(inlier_pts) < 3:
                    _print(f"    ID {mid} corner {cidx}: too few inliers ({len(inlier_pts)})")
                    continue

                avg_pos = inlier_pts.mean(axis=0)
                inlier_std = inlier_pts.std(axis=0) * self.camera_scale_to_robot_units
                avg_reproj = float(np.mean(accum_reproj[key]))

                rejected = max_std > self.MAX_STD_MM
                tag = f"  REJECTED (std {max_std:.1f} > {self.MAX_STD_MM})" if rejected else ""

                _print(f"    ID {mid} corner {cidx}: "
                       f"std=({inlier_std[0]:.1f}, {inlier_std[1]:.1f}, {inlier_std[2]:.1f})mm "
                       f"reproj={avg_reproj:.2f}px  "
                       f"({len(inlier_pts)}/{len(pts)} inliers){tag}")

                if rejected:
                    continue

                cam_raw = (float(avg_pos[0]), float(avg_pos[1]), float(avg_pos[2]))
                cam_s = (
                    cam_raw[0] * self.camera_scale_to_robot_units,
                    cam_raw[1] * self.camera_scale_to_robot_units,
                    cam_raw[2] * self.camera_scale_to_robot_units,
                )

                camera_raw.append(cam_raw)
                camera_scaled.append(cam_s)
                robot_points.append(robot_corners[cidx])
                reproj_errors.append(avg_reproj)
                added += 1

            per_marker_counts[mid] = added

        if len(camera_scaled) < 3:
            return {
                "ok": False,
                "reason": f"Only {len(camera_scaled)} valid averaged pairs (need >= 3).",
            }

        return {
            "ok": True,
            "camera_raw": camera_raw,
            "camera_scaled": camera_scaled,
            "robot_points": robot_points,
            "reproj_errors": reproj_errors,
            "per_marker_counts": per_marker_counts,
        }

    def _collect_correspondences(self, left_markers: dict[int, np.ndarray], right_markers: dict[int, np.ndarray]):
        camera_raw: list[tuple[float, float, float]] = []
        camera_scaled: list[tuple[float, float, float]] = []
        robot_points: list[tuple[float, float, float]] = []
        reproj_errors: list[float] = []
        per_marker_counts: dict[int, int] = {}

        marker_origins = self._active_marker_origins()
        for marker_id, origin_xyz in marker_origins.items():
            if origin_xyz is None:
                return {
                    "ok": False,
                    "reason": f"Missing robot origin for marker ID {marker_id}",
                }

        missing = []
        for marker_id in marker_origins:
            if marker_id not in left_markers or marker_id not in right_markers:
                missing.append(marker_id)

        if missing:
            return {
                "ok": False,
                "reason": f"Missing marker(s) in one/both views: {missing}",
            }

        for marker_id, origin_xyz in marker_origins.items():
            robot_corners = self._compute_robot_corners(origin_xyz)
            tri_results = self._triangulate_marker(
                left_markers[marker_id], right_markers[marker_id]
            )

            added = 0
            for idx, (cam_raw, reproj) in enumerate(tri_results):
                if cam_raw is None or reproj is None:
                    continue
                if reproj > self.max_reproj_px:
                    continue

                cam_scaled = (
                    cam_raw[0] * self.camera_scale_to_robot_units,
                    cam_raw[1] * self.camera_scale_to_robot_units,
                    cam_raw[2] * self.camera_scale_to_robot_units,
                )

                camera_raw.append(cam_raw)
                camera_scaled.append(cam_scaled)
                robot_points.append(robot_corners[idx])
                reproj_errors.append(reproj)
                added += 1

            per_marker_counts[marker_id] = added

        if len(camera_scaled) < 3:
            return {
                "ok": False,
                "reason": (
                    f"Only {len(camera_scaled)} valid pairs after reproj filter "
                    f"(need >= 3). Marker counts: {per_marker_counts}"
                ),
            }

        return {
            "ok": True,
            "camera_raw": camera_raw,
            "camera_scaled": camera_scaled,
            "robot_points": robot_points,
            "reproj_errors": reproj_errors,
            "per_marker_counts": per_marker_counts,
        }

    def _solve_transform(self, camera_scaled, robot_points):
        cam = np.asarray(camera_scaled, dtype=float)
        rob = np.asarray(robot_points, dtype=float)
        R, t, diag = points_based_transform(cam, rob, return_diagnostics=True)

        fitted = transform_points(cam, R, t)
        residuals = rob - fitted
        per_point_err = np.linalg.norm(residuals, axis=1)

        return {
            "R": R,
            "t": t,
            "diagnostics": diag,
            "per_point_err": per_point_err,
        }

    def _print_summary(self, result: dict, corr: dict) -> None:
        diag = result["diagnostics"]
        per_point_err = result["per_point_err"]

        _print("")
        _print("=" * 74)
        _print(" MULTI-ARUCO CAMERA->ROBOT TRANSFORM ")
        _print("=" * 74)
        marker_ids = sorted(self._active_marker_origins().keys())
        marker_label = " + ".join(f"ID {marker_id}" for marker_id in marker_ids)
        _print(f"Markers used: {marker_label}")
        _print(f"Pairs used: {diag['num_points']}")
        _print(f"Per-marker valid pairs: {corr['per_marker_counts']}")
        _print(
            f"Marker size: {self.marker_size_mm:.1f}mm, "
            f"axes: width={self.width_axis_name}, height={self.height_axis_name}"
        )
        _print(f"Camera scale: {self.camera_scale_to_robot_units:.6f}")
        _print("")
        _print("R =")
        _print(np.array2string(result["R"], precision=8, suppress_small=False))
        _print("")
        _print("t =")
        _print(np.array2string(result["t"], precision=8, suppress_small=False))
        _print("")
        _print(
            f"Fit quality: rmse={diag['rmse']:.4f}mm, mean={diag['mean_error']:.4f}mm, "
            f"max={diag['max_error']:.4f}mm, det(R)={diag['rotation_det']:.6f}"
        )
        _print(
            f"Per-point residual stats: median={float(np.median(per_point_err)):.2f}mm, "
            f"90th={float(np.percentile(per_point_err, 90)):.2f}mm"
        )
        _print("=" * 74)

    def _save_transform(self, result: dict) -> None:
        path = save_points_based_transform(
            rotation=result["R"],
            translation=result["t"],
            output_path=self.output_file,
            camera_scale_to_robot_units=self.camera_scale_to_robot_units,
        )
        _print(f"Saved transform: {path}")

    def _draw_overlay(self, left_small, right_small, left_markers: dict[int, np.ndarray], right_markers: dict[int, np.ndarray]):
        scale_x = self.display_width / self.frame_width
        scale_y = self.display_height / self.frame_height
        needed = set(self._active_marker_origins().keys())
        need_ids_text = ", ".join(str(marker_id) for marker_id in sorted(needed))

        for img, name in ((left_small, "LEFT"), (right_small, "RIGHT")):
            cv2.rectangle(img, (0, 0), (460, 52), (0, 0, 0), -1)
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
                f"Need IDs: {need_ids_text}",
                (8, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
            )

        def draw_markers(img, marker_map):
            for marker_id, corners in marker_map.items():
                color = (
                    (0, 220, 0)
                    if marker_id in needed
                    else (120, 120, 120)
                )
                for i in range(4):
                    x = int(corners[i][0] * scale_x)
                    y = int(corners[i][1] * scale_y)
                    cv2.circle(img, (x, y), 4, color, -1)
                    x2 = int(corners[(i + 1) % 4][0] * scale_x)
                    y2 = int(corners[(i + 1) % 4][1] * scale_y)
                    cv2.line(img, (x, y), (x2, y2), color, 1)

                x0 = int(corners[0][0] * scale_x)
                y0 = int(corners[0][1] * scale_y)
                cv2.circle(img, (x0, y0), 8, (0, 0, 255), 2)
                cv2.putText(
                    img,
                    f"ID {marker_id}",
                    (x0 + 8, y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                )

        draw_markers(left_small, left_markers)
        draw_markers(right_small, right_markers)

        found_left = set(left_markers.keys())
        found_right = set(right_markers.keys())
        ok = needed.issubset(found_left) and needed.issubset(found_right)
        status = "READY (all required IDs in both views)" if ok else "Waiting for required IDs in both views"

        cv2.putText(
            left_small,
            status,
            (10, self.display_height - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 0) if ok else (0, 180, 255),
            1,
        )
        cv2.putText(
            right_small,
            "SPACE freeze | s solve/save | q quit",
            (10, self.display_height - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (190, 190, 190),
            1,
        )

    def run(self) -> int:
        marker_origins = self._active_marker_origins()
        marker_ids = sorted(marker_origins.keys())
        marker_id_text = ", ".join(f"ID {marker_id}" for marker_id in marker_ids)
        marker_origin_text = ", ".join(
            f"ID{marker_id}={marker_origins[marker_id]}" for marker_id in marker_ids
        )
        marker_count = len(marker_ids)
        _print("\n" + "=" * 74)
        _print(f" FIND TRANSFORM ({marker_count} ARUCO MARKERS) ")
        _print("=" * 74)
        _print(f"Markers: {marker_id_text}")
        _print(f"Marker size: {self.marker_size_mm:.1f} mm")
        _print(f"Marker origins (corner[0], mm): {marker_origin_text}")
        _print(f"Axes: width={self.width_axis_name}, height={self.height_axis_name}")
        _print(f"Scale: {self.camera_scale_to_robot_units} (camera units -> robot mm)")
        _print(f"Reproj filter: <= {self.max_reproj_px:.2f}px")
        _print(f"Output: {self.output_file}")
        _print("Controls: SPACE freeze/unfreeze | s solve+save | q quit")
        _print("Using default fixture values; override with CLI args if needed.")
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
        solved_once = False

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

                left_markers = self._detect_markers(self.latest_left)
                right_markers = self._detect_markers(self.latest_right)

                left_small = cv2.resize(self.latest_left, (self.display_width, self.display_height))
                right_small = cv2.resize(self.latest_right, (self.display_width, self.display_height))
                self._draw_overlay(left_small, right_small, left_markers, right_markers)
                cv2.imshow(self.window_name, cv2.hconcat([left_small, right_small]))

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    self.frozen = not self.frozen
                    _print("[FROZEN]" if self.frozen else "[LIVE]")
                    continue
                if key == ord("s"):
                    self._ensure_fixture_definition()
                    _print(f"\n  Multi-frame averaging ({self.AVERAGE_FRAMES} frames)...")
                    corr = self._collect_averaged_correspondences()
                    if not corr["ok"]:
                        _print(f"Cannot solve: {corr['reason']}")
                        continue

                    result = self._solve_transform(
                        corr["camera_scaled"],
                        corr["robot_points"],
                    )
                    self._print_summary(result, corr)
                    self._save_transform(result)
                    solved_once = True

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

        if not solved_once:
            _print("No transform solved.")
            exit_code = 1
        return exit_code


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find camera->robot transform from fixed ArUco markers."
    )
    parser.add_argument("--marker-size", type=float, default=MARKER_SIZE_MM,
                        help=f"Marker side length in mm (default: {MARKER_SIZE_MM})")
    parser.add_argument("--width-axis", type=str, default=DEFAULT_WIDTH_AXIS, choices=sorted(AXIS_MAP.keys()),
                        help=f"Robot axis for marker width direction (default: {DEFAULT_WIDTH_AXIS})")
    parser.add_argument("--height-axis", type=str, default=DEFAULT_HEIGHT_AXIS, choices=sorted(AXIS_MAP.keys()),
                        help=f"Robot axis for marker height direction (default: {DEFAULT_HEIGHT_AXIS})")
    parser.add_argument("--camera-scale-to-robot-units", type=float, default=10.0,
                        help="Scale camera units -> robot units (default: 10.0 for cm->mm)")
    parser.add_argument("--max-reproj-px", type=float, default=3.0,
                        help="Reject triangulated corners above this reprojection error (default: 3.0)")
    parser.add_argument("--output-file", type=str, default=DEFAULT_POINTS_BASED_TRANSFORM_FILE,
                        help="Path to save transform JSON")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    app = DualArucoTransformFinder(
        markers=MARKERS,
        marker_size_mm=args.marker_size,
        width_axis=args.width_axis,
        height_axis=args.height_axis,
        camera_scale_to_robot_units=args.camera_scale_to_robot_units,
        max_reproj_px=args.max_reproj_px,
        output_file=args.output_file,
    )
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
