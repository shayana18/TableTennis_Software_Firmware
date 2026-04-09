"""
Distance Triangulation Verification (Camera Frame + Robot Frame)
===============================================================

Standalone click-to-triangulate verifier, modeled after the DISTANCE mode in
test_triangulation_verify.py:

1) Freeze image (SPACE)
2) Click target in LEFT image
3) Click same target in RIGHT image
4) Script prints:
   - Triangulated point in camera frame (cm)
   - Same point transformed to robot frame (mm) using TrajectoryPredictor
   - Optional Z-depth error vs ground-truth tape measurement

CONTROLS
--------
SPACE : freeze/unfreeze
r     : reset clicks
n     : next measurement (same as reset clicks)
g     : set/clear ground-truth Z depth (cm)
p     : print summary
q     : quit
"""

import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.camera_config import load_camera_settings
from tracking.stereo_triangulator import StereoTriangulator
from comm_function.points_based_transform import load_points_based_transform


class DistanceRobotFrameVerifier:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.calibration_dir = os.path.join(
            self.script_dir, "..", "camera_calibration", "camera_parameters"
        )

        cam_settings = load_camera_settings()
        self.frame_width = cam_settings["frame_width"]
        self.frame_height = cam_settings["frame_height"]
        self.cam_left_id = cam_settings["camera0"]
        self.cam_right_id = cam_settings["camera1"]

        self.display_width = 640
        self.display_height = int(
            self.display_width * self.frame_height / self.frame_width
        )

        self.triangulator = None

        # Load points-based cam->robot transform
        tf = load_points_based_transform()
        self._tf_R = tf["rotation"]
        self._tf_t = tf["translation"]
        self._tf_scale = tf["camera_scale_to_robot_units"]
        print(f"Loaded points-based transform (scale={self._tf_scale})")

        self.frozen = False
        self.frozen_left = None
        self.frozen_right = None

        self.click_points = []  # [(x,y)_left, (x,y)_right] in full-res image coords
        self.click_stage = 0

        self.measurements = []
        self.gt_depth_z_cm = None

        self._fps_time = time.time()
        self._fps_display = 0.0

    def check_calibration(self):
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
            print("ERROR: Missing calibration files:")
            for f in missing:
                print(f"  - {f}")
            return False
        return True

    def pixel_to_original(self, x_display, y_display, is_right=False):
        if is_right:
            x_display -= self.display_width
        scale_x = self.frame_width / self.display_width
        scale_y = self.frame_height / self.display_height
        return int(x_display * scale_x), int(y_display * scale_y)

    def _capture_pair(self):
        cap_l = self.triangulator.cap_left
        cap_r = self.triangulator.cap_right
        if cap_l is None or cap_r is None:
            return None, None

        grabbed_l = cap_l.grab()
        grabbed_r = cap_r.grab()
        if not grabbed_l or not grabbed_r:
            return None, None

        ret_l, frame_l = cap_l.retrieve()
        ret_r, frame_r = cap_r.retrieve()
        if not ret_l or not ret_r:
            return None, None

        return frame_l, frame_r

    def _triangulate_click_pair(self, pt_left, pt_right):
        tri = self.triangulator
        pt_l_rect = tri._rectify_point(
            pt_left, tri.cmtx0, tri.dist0, tri.R_rect0, tri.P_rect0
        )
        pt_r_rect = tri._rectify_point(
            pt_right, tri.cmtx1, tri.dist1, tri.R_rect1, tri.P_rect1
        )

        disparity = pt_l_rect[0] - pt_r_rect[0]
        if disparity <= 0:
            print("  ERROR: Negative disparity. Check left/right click order.")
            return None

        pos = tri.triangulate(pt_l_rect, pt_r_rect)
        if pos[2] <= 0:
            print("  ERROR: Triangulated point is behind camera.")
            return None

        err_l, err_r = tri._reprojection_error(pos, pt_l_rect, pt_r_rect)
        reproj = max(err_l, err_r)
        return tuple(pos), reproj, disparity

    def _record_measurement(self):
        if len(self.click_points) < 2:
            return

        tri_out = self._triangulate_click_pair(self.click_points[0], self.click_points[1])
        if tri_out is None:
            return

        (cam_x, cam_y, cam_z), reproj, disparity = tri_out
        cam_scaled = np.array([cam_x, cam_y, cam_z]) * self._tf_scale
        robot_pt = self._tf_R @ cam_scaled + self._tf_t
        robot_x, robot_y, robot_z = float(robot_pt[0]), float(robot_pt[1]), float(robot_pt[2])

        m = {
            "n": len(self.measurements) + 1,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "cam_x_cm": float(cam_x),
            "cam_y_cm": float(cam_y),
            "cam_z_cm": float(cam_z),
            "robot_x_mm": float(robot_x),
            "robot_y_mm": float(robot_y),
            "robot_z_mm": float(robot_z),
            "reproj_px": float(reproj),
            "disparity_px": float(disparity),
            "gt_z_cm": self.gt_depth_z_cm,
        }

        if self.gt_depth_z_cm is not None:
            err_cm = cam_z - self.gt_depth_z_cm
            m["z_error_cm"] = float(err_cm)
            m["z_error_pct"] = float(err_cm / self.gt_depth_z_cm * 100.0)
        else:
            m["z_error_cm"] = None
            m["z_error_pct"] = None

        self.measurements.append(m)

        print(f"\n  ┌─ MEASUREMENT #{m['n']} ─────────────────────────────────────┐")
        print(
            f"  │ Cam  (cm): X={cam_x:+8.2f}  Y={cam_y:+8.2f}  Z={cam_z:+8.2f} │"
        )
        print(
            f"  │ Robot(mm): X={robot_x:+8.1f}  Y={robot_y:+8.1f}  Z={robot_z:+8.1f} │"
        )
        if self.gt_depth_z_cm is not None:
            print(
                f"  │ GT Z={self.gt_depth_z_cm:7.2f} cm  "
                f"Err={m['z_error_cm']:+7.2f} cm ({m['z_error_pct']:+6.2f}%) │"
            )
        else:
            print("  │ GT Z: not set (press 'g' to set)                            │")
        print(f"  │ Reproj={reproj:5.2f}px  Disparity={disparity:6.2f}px                 │")
        print("  └──────────────────────────────────────────────────────────────┘")
        print("  Press 'n' for next point, then click LEFT and RIGHT again.")

    def _prompt_ground_truth(self):
        s = input("\nEnter ground-truth depth Z in cm (blank to clear): ").strip()
        if s == "":
            self.gt_depth_z_cm = None
            print("  Ground-truth Z cleared.")
            return
        try:
            v = float(s)
            if v <= 0:
                raise ValueError
            self.gt_depth_z_cm = v
            print(f"  Ground-truth Z set to {v:.2f} cm")
        except ValueError:
            print("  Invalid value. Ground-truth Z unchanged.")

    def _draw_overlay(self, left_small, right_small):
        scale_x = self.display_width / self.frame_width
        scale_y = self.display_height / self.frame_height

        # Header boxes
        for img, label in ((left_small, "LEFT"), (right_small, "RIGHT")):
            cv2.rectangle(img, (0, 0), (330, 40), (0, 0, 0), -1)
            cv2.putText(
                img,
                f"{label} {'[FROZEN]' if self.frozen else '[LIVE]'}",
                (8, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                img,
                f"{self._fps_display:.1f} fps",
                (8, 33),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 0),
                1,
            )

        # Click stage guidance
        if self.click_stage == 0:
            cv2.rectangle(
                left_small,
                (0, 0),
                (self.display_width - 1, self.display_height - 1),
                (0, 255, 255),
                3,
            )
            cv2.putText(
                left_small,
                "Click target on LEFT image",
                (10, self.display_height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )
        elif self.click_stage == 1:
            cv2.rectangle(
                right_small,
                (0, 0),
                (self.display_width - 1, self.display_height - 1),
                (0, 255, 255),
                3,
            )
            cv2.putText(
                left_small,
                "Click same point on RIGHT image",
                (10, self.display_height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

        # Draw chosen points
        panels = [left_small, right_small]
        labels = ["L", "R"]
        for i, pt in enumerate(self.click_points[:2]):
            px = int(pt[0] * scale_x)
            py = int(pt[1] * scale_y)
            cv2.circle(panels[i], (px, py), 20, (255, 255, 0), 2)
            cv2.drawMarker(panels[i], (px, py), (0, 255, 0), cv2.MARKER_CROSS, 26, 2)
            cv2.putText(
                panels[i],
                labels[i],
                (px + 22, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # Status footer
        gt_text = "none" if self.gt_depth_z_cm is None else f"{self.gt_depth_z_cm:.2f} cm"
        footer = (
            f"GT Z: {gt_text}   |   "
            "SPACE freeze  g set-GT  n/r reset  p summary  q quit"
        )
        cv2.rectangle(
            left_small,
            (0, self.display_height - 28),
            (self.display_width, self.display_height),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            left_small,
            footer,
            (8, self.display_height - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (200, 200, 200),
            1,
        )

        # Last measurement quick display
        if self.measurements:
            m = self.measurements[-1]
            line1 = (
                f"Cam(cm): X={m['cam_x_cm']:+.1f} Y={m['cam_y_cm']:+.1f} Z={m['cam_z_cm']:+.1f}"
            )
            line2 = (
                f"Robot(mm): X={m['robot_x_mm']:+.0f} Y={m['robot_y_mm']:+.0f} Z={m['robot_z_mm']:+.0f}"
            )
            cv2.rectangle(right_small, (0, self.display_height - 52), (self.display_width, self.display_height), (0, 0, 0), -1)
            cv2.putText(
                right_small,
                line1,
                (8, self.display_height - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )
            cv2.putText(
                right_small,
                line2,
                (8, self.display_height - 11),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
            )

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        is_right = x >= self.display_width
        orig_x, orig_y = self.pixel_to_original(x, y, is_right)

        if self.click_stage == 0:
            if is_right:
                return
            self.click_points.append((orig_x, orig_y))
            self.click_stage = 1
            print(f"  LEFT click: ({orig_x}, {orig_y})")
            return

        if self.click_stage == 1:
            if not is_right:
                return
            self.click_points.append((orig_x, orig_y))
            self.click_stage = 2
            print(f"  RIGHT click: ({orig_x}, {orig_y})")
            self._record_measurement()
            return

    def print_summary(self):
        n = len(self.measurements)
        print("\n" + "=" * 72)
        print(f"SUMMARY: {n} measurement(s)")
        if n == 0:
            print("No data yet.")
            print("=" * 72)
            return

        for m in self.measurements:
            cam = (m["cam_x_cm"], m["cam_y_cm"], m["cam_z_cm"])
            rob = (m["robot_x_mm"], m["robot_y_mm"], m["robot_z_mm"])
            print(
                f"#{m['n']:02d} Cam(cm)=({cam[0]:+.2f},{cam[1]:+.2f},{cam[2]:+.2f})  "
                f"Robot(mm)=({rob[0]:+.1f},{rob[1]:+.1f},{rob[2]:+.1f})  "
                f"Reproj={m['reproj_px']:.2f}px"
            )
            if m["z_error_cm"] is not None:
                print(
                    f"    GT_Z={m['gt_z_cm']:.2f}cm  Err={m['z_error_cm']:+.2f}cm "
                    f"({m['z_error_pct']:+.2f}%)"
                )

        z_errs = [abs(m["z_error_cm"]) for m in self.measurements if m["z_error_cm"] is not None]
        if z_errs:
            print(
                f"\nZ error | mean abs={np.mean(z_errs):.2f} cm, "
                f"max abs={np.max(z_errs):.2f} cm"
            )
        print("=" * 72)

    def run(self):
        print("\n" + "=" * 72)
        print("DISTANCE TRIANGULATION VERIFY (Camera + Robot Frame)")
        print("=" * 72)

        if not self.check_calibration():
            return

        try:
            self.triangulator = StereoTriangulator(
                calibration_dir=self.calibration_dir,
                cam_left_id=self.cam_left_id,
                cam_right_id=self.cam_right_id,
            )
        except Exception as e:
            print(f"ERROR initializing triangulator: {e}")
            return

        print(f"Resolution: {self.frame_width}x{self.frame_height}")
        print(f"Baseline: {self.triangulator.get_baseline():.2f} cm")
        print(f"Focal length: {self.triangulator.get_focal_length_px():.1f} px")
        print("\nControls: SPACE freeze | g set GT Z | n/r reset clicks | p summary | q quit")

        try:
            self.triangulator.start_cameras(self.frame_width, self.frame_height)
        except RuntimeError as e:
            print(f"ERROR: {e}")
            return

        cv2.namedWindow("Triangulation Verify Robot Frame")
        cv2.setMouseCallback("Triangulation Verify Robot Frame", self.on_mouse)

        try:
            while True:
                if not self.frozen:
                    frame_l, frame_r = self._capture_pair()
                    if frame_l is None or frame_r is None:
                        continue
                    self.frozen_left = frame_l.copy()
                    self.frozen_right = frame_r.copy()
                else:
                    frame_l, frame_r = self.frozen_left, self.frozen_right
                    if frame_l is None or frame_r is None:
                        continue

                self._fps_display = 30.0 / max(1e-6, (time.time() - self._fps_time))
                self._fps_time = time.time()

                left_small = cv2.resize(frame_l, (self.display_width, self.display_height))
                right_small = cv2.resize(frame_r, (self.display_width, self.display_height))
                self._draw_overlay(left_small, right_small)

                combined = cv2.hconcat([left_small, right_small])
                cv2.imshow("Triangulation Verify Robot Frame", combined)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    self.frozen = not self.frozen
                    print(f"  [{'FROZEN' if self.frozen else 'LIVE'}]")
                elif key in (ord("r"), ord("n")):
                    self.click_points = []
                    self.click_stage = 0
                    print("  Click reset. Ready for LEFT click.")
                elif key == ord("g"):
                    self._prompt_ground_truth()
                elif key == ord("p"):
                    self.print_summary()

        except KeyboardInterrupt:
            pass
        finally:
            self.triangulator.stop_cameras()
            cv2.destroyAllWindows()

        self.print_summary()


def main():
    DistanceRobotFrameVerifier().run()


if __name__ == "__main__":
    main()
