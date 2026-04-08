"""
Robot transform verification and comparison tool.

Compares two camera->robot transform methods on clicked stereo points:
1) Offset/pose-based transform from trajectory_predictor.py (current pipeline)
2) Points-based transform loaded from points_based_transform_result.json

Workflow:
  1. Freeze frame (SPACE), click same target in LEFT then RIGHT image.
  2. Script triangulates camera point (cm).
  3. Script transforms that camera point using both methods.
  4. Compare robot XYZ outputs against tape-measure ground truth.

Controls:
  SPACE - freeze/unfreeze frame
  r     - reset current click pair
  u     - reload offset/pose transform from trajectory_predictor.py constants
  o     - reload points-based transform file
  p     - print summary table
  q     - quit

Prerequisite: run trigger_setup.py first for camera sync.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comm_function.points_based_transform import (
    DEFAULT_POINTS_BASED_TRANSFORM_FILE,
    load_points_based_transform,
)
from config.camera_config import load_camera_settings
from tracking.stereo_triangulator import StereoTriangulator
from trajectory.trajectory_predictor import (
    CAM_POSE_PITCH,
    CAM_POSE_ROLL,
    CAM_POSE_X_MM,
    CAM_POSE_Y_MM,
    CAM_POSE_YAW,
    CAM_POSE_Z_MM,
    ROBOT_HOME,
    TrajectoryPredictor,
)


class RobotTransformVerifier:
    """Click-to-verify and compare camera->robot transforms."""

    def __init__(self, points_transform_file: str) -> None:
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.calibration_dir = os.path.join(
            self.script_dir, "..", "camera_calibration", "camera_parameters"
        )

        cam = load_camera_settings()
        self.frame_width = cam["frame_width"]
        self.frame_height = cam["frame_height"]
        self.cam_left_id = cam["camera0"]
        self.cam_right_id = cam["camera1"]

        self.display_width = 640
        self.display_height = int(
            self.display_width * self.frame_height / self.frame_width
        )

        self.triangulator: Optional[StereoTriangulator] = None

        # Offset/pose-based transform used by current trajectory predictor.
        self.predictor = TrajectoryPredictor(
            buffer_size=5, min_points=3, gravity=981.0, y_down=True, enable_drag=False
        )

        # Points-based transform loaded from file.
        self.points_transform_file = os.path.abspath(points_transform_file)
        self.points_transform: Optional[dict] = None
        self.points_transform_ok = self._load_points_transform(verbose=False)

        # Click state
        self.frozen = False
        self.frozen_left = None
        self.frozen_right = None
        self.click_points: list[tuple[int, int]] = []
        self.click_stage = 0  # 0=waiting left, 1=waiting right, 2=done

        # Results
        self.measurements: list[dict] = []

        # FPS
        self._fps_n = 0
        self._fps_t = time.perf_counter()
        self._fps = 0.0

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
            print("ERROR: Missing calibration files:", missing)
            return False
        return True

    def verify_trigger_sync(self) -> bool:
        cap_l = self.triangulator.cap_left
        cap_r = self.triangulator.cap_right
        trig_l = int(cap_l.get(cv2.CAP_PROP_BACKLIGHT))
        trig_r = int(cap_r.get(cv2.CAP_PROP_BACKLIGHT))
        print(
            f"\n  Trigger: LEFT={trig_l} {'[OK]' if trig_l == 1 else '[FAIL]'}  "
            f"RIGHT={trig_r} {'[OK]' if trig_r == 1 else '[FAIL]'}"
        )
        if trig_l != 1 or trig_r != 1:
            print("  WARNING: Trigger mode not active. Run trigger_setup.py first.")
        return trig_l == 1 and trig_r == 1

    def pixel_to_original(self, x_display: int, y_display: int, is_right: bool = False):
        if is_right:
            x_display = x_display - self.display_width
        scale_x = self.frame_width / self.display_width
        scale_y = self.frame_height / self.display_height
        return int(x_display * scale_x), int(y_display * scale_y)

    def _triangulate_click_pair(self, pt_left, pt_right):
        tri = self.triangulator
        pt_l_rect = tri._rectify_point(
            pt_left, tri.cmtx0, tri.dist0, tri.R_rect0, tri.P_rect0
        )
        pt_r_rect = tri._rectify_point(
            pt_right, tri.cmtx1, tri.dist1, tri.R_rect1, tri.P_rect1
        )
        disp = pt_l_rect[0] - pt_r_rect[0]
        if disp <= 0:
            print("  ERROR: Negative disparity; left/right points may be swapped")
            return None, None
        pos = tri.triangulate(pt_l_rect, pt_r_rect)
        if pos[2] <= 0:
            print("  ERROR: Point behind camera")
            return None, None
        err_l, err_r = tri._reprojection_error(pos, pt_l_rect, pt_r_rect)
        reproj = max(err_l, err_r)
        return tuple(pos), reproj

    def _offset_transform(self) -> tuple[np.ndarray, np.ndarray]:
        """Return current predictor rotation/translation (camera cm -> robot mm)."""
        R = np.asarray(self.predictor._R_cam_to_robot, dtype=float)
        t = np.asarray(self.predictor._t_cam_to_robot, dtype=float).reshape(3)
        return R, t

    def _load_points_transform(self, verbose: bool = True) -> bool:
        try:
            self.points_transform = load_points_based_transform(self.points_transform_file)
            if verbose:
                print(f"  Loaded points-based transform: {self.points_transform_file}")
            return True
        except Exception as exc:
            self.points_transform = None
            if verbose:
                print(
                    "  Points-based transform unavailable "
                    f"({self.points_transform_file}): {exc}"
                )
            return False

    def _points_based_cam_to_robot(self, cam_x: float, cam_y: float, cam_z: float):
        if self.points_transform is None:
            return None
        R = self.points_transform["rotation"]
        t = self.points_transform["translation"]
        scale = float(self.points_transform.get("camera_scale_to_robot_units", 1.0))
        cam_vec = np.array([cam_x, cam_y, cam_z], dtype=float) * scale
        p_robot = R @ cam_vec + t
        return float(p_robot[0]), float(p_robot[1]), float(p_robot[2])

    def _refresh_measurement_transform_values(self, m: dict) -> None:
        cam_x, cam_y, cam_z = m["cam_x"], m["cam_y"], m["cam_z"]

        # Offset/pose-based output.
        off_x, off_y, off_z = self.predictor.cam_to_robot(cam_x, cam_y, cam_z)
        m["offset_x"] = off_x
        m["offset_y"] = off_y
        m["offset_z"] = off_z
        m["offset_in_workspace"] = self.predictor.check_workspace(off_x, off_y, off_z)

        # Points-based output (if file is loaded).
        pts_xyz = self._points_based_cam_to_robot(cam_x, cam_y, cam_z)
        if pts_xyz is None:
            m["points_x"] = None
            m["points_y"] = None
            m["points_z"] = None
            m["points_in_workspace"] = None
            m["delta_x"] = None
            m["delta_y"] = None
            m["delta_z"] = None
            m["delta_norm"] = None
            return

        pts_x, pts_y, pts_z = pts_xyz
        m["points_x"] = pts_x
        m["points_y"] = pts_y
        m["points_z"] = pts_z
        m["points_in_workspace"] = self.predictor.check_workspace(pts_x, pts_y, pts_z)

        dx = pts_x - off_x
        dy = pts_y - off_y
        dz = pts_z - off_z
        m["delta_x"] = dx
        m["delta_y"] = dy
        m["delta_z"] = dz
        m["delta_norm"] = float(np.linalg.norm([dx, dy, dz]))

    def _recompute_existing_measurements(self) -> None:
        for m in self.measurements:
            self._refresh_measurement_transform_values(m)

    def _print_transform_matrices(self) -> None:
        R_off, t_off = self._offset_transform()
        print("\n  Offset/Pose-based transform (trajectory_predictor)")
        print("    Equation: p_robot = R_off @ (p_cam_cm * 10.0) + t_off")
        print("    R_off =")
        print(np.array2string(R_off, precision=8, suppress_small=False))
        print("    t_off =")
        print(np.array2string(t_off, precision=8, suppress_small=False))

        if self.points_transform is None:
            print(
                f"\n  Points-based transform not loaded: {self.points_transform_file}\n"
            )
            return

        R_pts = self.points_transform["rotation"]
        t_pts = self.points_transform["translation"]
        scale = float(self.points_transform.get("camera_scale_to_robot_units", 1.0))
        print(f"\n  Points-based transform (file: {self.points_transform_file})")
        print("    Equation: p_robot = R_pts @ (p_cam * scale) + t_pts")
        print(f"    scale = {scale}")
        print("    R_pts =")
        print(np.array2string(R_pts, precision=8, suppress_small=False))
        print("    t_pts =")
        print(np.array2string(t_pts, precision=8, suppress_small=False))
        print()

    def reload_camera_pose(self) -> None:
        """Reload CAM_POSE_* constants from trajectory_predictor.py."""
        import trajectory.trajectory_predictor as tp_mod

        importlib.reload(tp_mod)
        self.predictor.set_camera_pose(
            tp_mod.CAM_POSE_X_MM,
            tp_mod.CAM_POSE_Y_MM,
            tp_mod.CAM_POSE_Z_MM,
            tp_mod.CAM_POSE_YAW,
            tp_mod.CAM_POSE_PITCH,
            tp_mod.CAM_POSE_ROLL,
        )
        print("  Offset/pose transform reloaded from trajectory_predictor.py")
        self._recompute_existing_measurements()

    def _print_point_report(self, m: dict) -> None:
        n = m["n"]
        cx, cy, cz = m["cam_x"], m["cam_y"], m["cam_z"]
        reproj = m["reproj"]
        off_x, off_y, off_z = m["offset_x"], m["offset_y"], m["offset_z"]
        off_ws = "IN" if m["offset_in_workspace"] else "OUT"

        print()
        print(f"  POINT #{n}")
        print("  Camera (cm):")
        print(f"    X={cx:+8.2f} Y={cy:+8.2f} Z={cz:+8.2f}")
        print("  Offset/Pose-based robot (mm):")
        print(f"    X={off_x:+8.1f} Y={off_y:+8.1f} Z={off_z:+8.1f}  WS:{off_ws}")

        if m["points_x"] is None:
            print("  Points-based robot (mm):")
            print("    unavailable (transform file not loaded)")
        else:
            pts_x, pts_y, pts_z = m["points_x"], m["points_y"], m["points_z"]
            pts_ws = "IN" if m["points_in_workspace"] else "OUT"
            print("  Points-based robot (mm):")
            print(f"    X={pts_x:+8.1f} Y={pts_y:+8.1f} Z={pts_z:+8.1f}  WS:{pts_ws}")
            print("  Delta (points - offset) (mm):")
            print(
                f"    dX={m['delta_x']:+7.1f} dY={m['delta_y']:+7.1f} "
                f"dZ={m['delta_z']:+7.1f}  |d|={m['delta_norm']:.1f}"
            )

        print(f"  Reprojection error: {reproj:.2f}px")
        print(
            f"  Home ref (mm): ({ROBOT_HOME[0]:.0f}, {ROBOT_HOME[1]:.0f}, {ROBOT_HOME[2]:.0f})"
        )

    # ================================================================
    # MOUSE CALLBACK
    # ================================================================

    def on_mouse(self, event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        is_right = x >= self.display_width
        orig_x, orig_y = self.pixel_to_original(x, y, is_right)

        if self.click_stage == 0:
            if is_right:
                return
            self.click_points = [(orig_x, orig_y)]
            self.click_stage = 1
            print(f"  LEFT click: ({orig_x}, {orig_y}) -> click same point on RIGHT")
            return

        if self.click_stage == 1:
            if not is_right:
                return
            self.click_points.append((orig_x, orig_y))
            self.click_stage = 2
            print(f"  RIGHT click: ({orig_x}, {orig_y})")

            pos, reproj = self._triangulate_click_pair(
                self.click_points[0], self.click_points[1]
            )
            if pos is None:
                print("  ERROR: Triangulation failed. Try a cleaner click pair.")
                return

            cam_x, cam_y, cam_z = float(pos[0]), float(pos[1]), float(pos[2])
            n = len(self.measurements) + 1
            m = {
                "n": n,
                "cam_x": cam_x,
                "cam_y": cam_y,
                "cam_z": cam_z,
                "reproj": float(reproj),
            }
            self._refresh_measurement_transform_values(m)
            self.measurements.append(m)
            self._print_point_report(m)

    # ================================================================
    # DISPLAY
    # ================================================================

    def _draw_overlay(self, left_small, right_small):
        h = self.display_height
        scale_x = self.display_width / self.frame_width
        scale_y = self.display_height / self.frame_height

        freeze_label = " [FROZEN]" if self.frozen else ""
        pts_label = "PtsFile:ON" if self.points_transform is not None else "PtsFile:OFF"
        for img, label in ((left_small, "LEFT"), (right_small, "RIGHT")):
            cv2.rectangle(img, (0, 0), (420, 52), (0, 0, 0), -1)
            cv2.putText(
                img,
                f"{label} | ROBOT TRANSFORM VERIFY{freeze_label}",
                (5, 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                img,
                f"{self._fps:.0f} fps  {pts_label}",
                (5, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0) if self.points_transform is not None else (0, 180, 255),
                1,
            )
            cv2.putText(
                img,
                f"Points: {len(self.measurements)}",
                (5, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 255),
                1,
            )

        if self.click_stage == 0:
            cv2.rectangle(
                left_small, (0, 0), (self.display_width - 1, h - 1), (0, 255, 255), 3
            )
            cv2.putText(
                left_small,
                "Click TARGET on LEFT image",
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )
        elif self.click_stage == 1:
            cv2.rectangle(
                right_small, (0, 0), (self.display_width - 1, h - 1), (0, 255, 255), 3
            )
            cv2.putText(
                left_small,
                "Click SAME point on RIGHT image",
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

        # Click markers
        panels = [left_small, right_small]
        labels = ["L", "R"]
        for i, pt in enumerate(self.click_points[:2]):
            img = panels[i]
            px = int(pt[0] * scale_x)
            py = int(pt[1] * scale_y)
            cv2.circle(img, (px, py), 25, (255, 255, 0), 2)
            cv2.drawMarker(img, (px, py), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
            cv2.circle(img, (px, py), 3, (0, 0, 255), -1)
            cv2.putText(
                img,
                labels[i],
                (px + 28, py - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        if self.click_stage >= 2 and self.measurements:
            last = self.measurements[-1]

            # Camera coordinates + reproj on left panel.
            box_l = h - 55
            cv2.rectangle(left_small, (0, box_l), (self.display_width, h), (0, 0, 0), -1)
            cv2.putText(
                left_small,
                f"Cam: X={last['cam_x']:+.1f} Y={last['cam_y']:+.1f} Z={last['cam_z']:+.1f} cm",
                (10, box_l + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
            )
            cv2.putText(
                left_small,
                f"Reproj: {last['reproj']:.1f}px",
                (10, box_l + 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (150, 150, 150),
                1,
            )

            # Comparison on right panel.
            box_r = h - 120
            cv2.rectangle(right_small, (0, box_r), (self.display_width, h), (0, 0, 0), -1)
            cv2.putText(
                right_small,
                "Offset: "
                f"X={last['offset_x']:+.0f} Y={last['offset_y']:+.0f} Z={last['offset_z']:+.0f} mm",
                (10, box_r + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (0, 255, 255),
                1,
            )

            if last["points_x"] is None:
                cv2.putText(
                    right_small,
                    "Points: unavailable (press 'o' after generating transform file)",
                    (10, box_r + 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 180, 255),
                    1,
                )
            else:
                cv2.putText(
                    right_small,
                    "Points: "
                    f"X={last['points_x']:+.0f} Y={last['points_y']:+.0f} Z={last['points_z']:+.0f} mm",
                    (10, box_r + 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (0, 255, 0),
                    1,
                )
                cv2.putText(
                    right_small,
                    "Delta(points-offset): "
                    f"dX={last['delta_x']:+.0f} dY={last['delta_y']:+.0f} dZ={last['delta_z']:+.0f} mm",
                    (10, box_r + 64),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (160, 220, 255),
                    1,
                )

            off_ws = "IN" if last["offset_in_workspace"] else "OUT"
            pts_ws = (
                ("IN" if last["points_in_workspace"] else "OUT")
                if last["points_in_workspace"] is not None
                else "NA"
            )
            cv2.putText(
                right_small,
                f"WS offset:{off_ws}  points:{pts_ws}",
                (10, box_r + 86),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (180, 180, 180),
                1,
            )
            cv2.putText(
                right_small,
                "r=reset  u=reload offset  o=reload points  p=summary",
                (10, box_r + 108),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.33,
                (150, 150, 150),
                1,
            )

    # ================================================================
    # SUMMARY
    # ================================================================

    @staticmethod
    def _fmt_opt(value: Optional[float], width: int = 7, prec: int = 1) -> str:
        if value is None:
            return " " * (width - 3) + "n/a"
        return f"{value:+{width}.{prec}f}"

    def print_summary(self):
        if not self.measurements:
            print("\n  No measurements yet.\n")
            return

        bar = "=" * 132
        print(f"\n  {bar}")
        print(f"  ROBOT TRANSFORM VERIFY SUMMARY - {len(self.measurements)} points")
        print(f"  {bar}")
        print(
            "    #     CamX    CamY    CamZ |"
            "   OffX   OffY   OffZ |"
            "   PtsX   PtsY   PtsZ |"
            "   |d|  Reprj  WS(o/p)"
        )
        print(f"  {'-' * 132}")

        for m in self.measurements:
            off_ws = "OK" if m["offset_in_workspace"] else "OUT"
            if m["points_in_workspace"] is None:
                pts_ws = "NA"
            else:
                pts_ws = "OK" if m["points_in_workspace"] else "OUT"
            print(
                f"  {m['n']:4d} "
                f"{m['cam_x']:+7.1f} {m['cam_y']:+7.1f} {m['cam_z']:+7.1f} |"
                f" {m['offset_x']:+6.0f} {m['offset_y']:+6.0f} {m['offset_z']:+6.0f} |"
                f" {self._fmt_opt(m['points_x'], 6, 0)}"
                f" {self._fmt_opt(m['points_y'], 6, 0)}"
                f" {self._fmt_opt(m['points_z'], 6, 0)} |"
                f" {self._fmt_opt(m['delta_norm'], 5, 1)}"
                f"  {m['reproj']:5.1f}   {off_ws:>3}/{pts_ws:<3}"
            )

        print(f"  {bar}")
        print("  Compare both robot-frame outputs against tape-measure values.")
        print("  Smaller real-world error is the better transform.")
        print()

    # ================================================================
    # RUN
    # ================================================================

    def run(self):
        print("\n" + "=" * 70)
        print("  ROBOT TRANSFORM VERIFICATION (OFFSET VS POINTS-BASED)")
        print("=" * 70)
        print("  Click known points and compare both transformed robot-frame outputs.")
        print("  Controls: SPACE=freeze  r=reset  u=reload offset  o=reload points  p=summary  q=quit")
        print("=" * 70)

        print("\n  Current camera pose constants (offset method):")
        print(f"    Position: ({CAM_POSE_X_MM}, {CAM_POSE_Y_MM}, {CAM_POSE_Z_MM}) mm")
        print(f"    Yaw={CAM_POSE_YAW} deg  Pitch={CAM_POSE_PITCH} deg  Roll={CAM_POSE_ROLL} deg")
        if CAM_POSE_X_MM == 0 and CAM_POSE_Y_MM == 0 and CAM_POSE_Z_MM == 0:
            print("    WARNING: camera position constants are all zeros.")
        print(f"  Points-based transform file: {self.points_transform_file}")
        print(
            "  Points-based status: "
            f"{'LOADED' if self.points_transform_ok else 'NOT LOADED'}"
        )
        self._print_transform_matrices()

        if not self.check_calibration():
            return

        try:
            self.triangulator = StereoTriangulator(
                calibration_dir=self.calibration_dir,
                cam_left_id=self.cam_left_id,
                cam_right_id=self.cam_right_id,
            )
            self.triangulator.start_cameras(self.frame_width, self.frame_height)
        except Exception as exc:
            print(f"ERROR: {exc}")
            return

        self.verify_trigger_sync()
        cv2.namedWindow("Robot Transform Verify")
        cv2.setMouseCallback("Robot Transform Verify", self.on_mouse)
        print("\n  Ready. Freeze frame (SPACE), then click target.\n")

        try:
            while True:
                if not self.triangulator.cap_left.grab():
                    continue
                if not self.triangulator.cap_right.grab():
                    continue
                ret_l, frame_l = self.triangulator.cap_left.retrieve()
                ret_r, frame_r = self.triangulator.cap_right.retrieve()
                if not ret_l or not ret_r or frame_l is None or frame_r is None:
                    continue

                self._fps_n += 1
                if self._fps_n % 30 == 0:
                    now = time.perf_counter()
                    self._fps = 30.0 / (now - self._fps_t)
                    self._fps_t = now

                if self.frozen:
                    show_l = self.frozen_left.copy()
                    show_r = self.frozen_right.copy()
                else:
                    show_l = cv2.resize(frame_l, (self.display_width, self.display_height))
                    show_r = cv2.resize(frame_r, (self.display_width, self.display_height))

                self._draw_overlay(show_l, show_r)
                cv2.imshow("Robot Transform Verify", np.hstack([show_l, show_r]))

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    if not self.frozen:
                        self.frozen = True
                        self.frozen_left = cv2.resize(
                            frame_l, (self.display_width, self.display_height)
                        )
                        self.frozen_right = cv2.resize(
                            frame_r, (self.display_width, self.display_height)
                        )
                        print("  [FROZEN] click target now")
                    else:
                        self.frozen = False
                        print("  [UNFROZEN]")
                    continue
                if key == ord("r"):
                    self.click_points = []
                    self.click_stage = 0
                    print("  [RESET] click next point")
                    continue
                if key == ord("u"):
                    self.reload_camera_pose()
                    if self.measurements:
                        print(f"  Re-transformed {len(self.measurements)} points with offset method")
                        self.print_summary()
                    self._print_transform_matrices()
                    continue
                if key == ord("o"):
                    self.points_transform_ok = self._load_points_transform(verbose=True)
                    self._recompute_existing_measurements()
                    if self.measurements:
                        print(f"  Re-transformed {len(self.measurements)} points with points method")
                        self.print_summary()
                    self._print_transform_matrices()
                    continue
                if key == ord("p"):
                    self.print_summary()
                    continue

        except KeyboardInterrupt:
            pass
        finally:
            self.triangulator.stop_cameras()
            cv2.destroyAllWindows()

        if self.measurements:
            self.print_summary()
        print("\nDone!")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify camera->robot transform by click triangulation and compare "
            "offset-based vs points-based outputs."
        )
    )
    parser.add_argument(
        "--points-transform-file",
        type=str,
        default=DEFAULT_POINTS_BASED_TRANSFORM_FILE,
        help="Path to points-based transform JSON (R,t) produced by test_find_points_based_transform.py",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    RobotTransformVerifier(points_transform_file=args.points_transform_file).run()


if __name__ == "__main__":
    main()
