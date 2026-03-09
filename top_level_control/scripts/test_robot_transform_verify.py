"""
Robot Transform Verification Tool
===================================
Verifies the camera-to-robot coordinate transform by clicking on objects
at known physical positions relative to the robot, then comparing the
transformed robot coordinates against tape-measure ground truth.

WHAT THIS DOES:
    1. You click on a point in the LEFT camera, then the SAME point in the RIGHT camera
    2. It triangulates the 3D position in camera coords (cm)
    3. It transforms to robot coords (mm) using the rotation matrix
    4. You compare the robot coords against what you measured with a tape measure

HOW TO USE:
    1. Place an object (e.g. ball on a stick, corner of a box) at a KNOWN position
       relative to the robot base plate. Measure that position with a tape measure.

    2. Run this script. Freeze the frame (SPACE), click the object in LEFT then RIGHT.

    3. Compare the displayed Robot X/Y/Z against your tape-measure values.

    4. If they don't match, adjust CAM_POSE_* values in trajectory_predictor.py
       and press 'u' to reload. Repeat until they match.

ROBOT FRAME (from robot.h):
    Stand behind the robot, looking toward the net.
    Origin = robot base plate center.
      +X = robot's right (across table width)       ±500mm
      +Y = toward net (along table length)           ±350mm
      +Z = UP (above base plate)                     -700 to -1100mm
    Home = (0, 0, -900) mm

CONTROLS:
    SPACE   - Freeze/unfreeze frame (for precise clicking)
    r       - Reset clicks and measure another point
    u       - Reload camera pose from trajectory_predictor.py constants
    p       - Print all measurements summary
    q       - Quit

PREREQUISITE: Run trigger_setup.py first for camera sync.
"""

import cv2
import sys
import os
import math
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracking.stereo_triangulator import StereoTriangulator
from trajectory.trajectory_predictor import (
    TrajectoryPredictor,
    CAM_POSE_X_MM, CAM_POSE_Y_MM, CAM_POSE_Z_MM,
    CAM_POSE_YAW, CAM_POSE_PITCH, CAM_POSE_ROLL,
    ROBOT_HOME, ROBOT_LIMIT_X, ROBOT_LIMIT_Y, ROBOT_LIMIT_Z,
)
from config.camera_config import load_camera_settings


class RobotTransformVerifier:
    """Click-to-verify camera→robot coordinate transform."""

    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.calibration_dir = os.path.join(
            self.script_dir, '..', 'camera_calibration', 'camera_parameters')

        cam = load_camera_settings()
        self.frame_width = cam['frame_width']
        self.frame_height = cam['frame_height']
        self.cam_left_id = cam['camera0']
        self.cam_right_id = cam['camera1']

        self.display_width = 640
        self.display_height = int(
            self.display_width * self.frame_height / self.frame_width)

        self.triangulator = None

        # Predictor (only used for cam_to_robot / robot_to_cam transform)
        self.predictor = TrajectoryPredictor(
            buffer_size=5, min_points=3,
            gravity=981.0, y_down=True, enable_drag=False)

        # State
        self.frozen = False
        self.frozen_left = None
        self.frozen_right = None
        self.click_points = []   # (x, y) in original image coords
        self.click_stage = 0     # 0 = waiting for left, 1 = waiting for right, 2 = done

        # Results
        self.measurements = []

        # FPS
        self._fps_n = 0
        self._fps_t = time.perf_counter()
        self._fps = 0.0

    def check_calibration(self):
        required = ['camera0_intrinsics.dat', 'camera1_intrinsics.dat',
                     'camera0_rot_trans.dat', 'camera1_rot_trans.dat']
        missing = [f for f in required
                   if not os.path.exists(os.path.join(self.calibration_dir, f))]
        if missing:
            print("ERROR: Missing calibration files:", missing)
            return False
        return True

    def verify_trigger_sync(self):
        cap_l = self.triangulator.cap_left
        cap_r = self.triangulator.cap_right
        trig_l = int(cap_l.get(cv2.CAP_PROP_BACKLIGHT))
        trig_r = int(cap_r.get(cv2.CAP_PROP_BACKLIGHT))
        print(f"\n  Trigger: LEFT={trig_l} {'[OK]' if trig_l==1 else '[FAIL]'}  "
              f"RIGHT={trig_r} {'[OK]' if trig_r==1 else '[FAIL]'}")
        if trig_l != 1 or trig_r != 1:
            print("  WARNING: Trigger mode NOT active. Run trigger_setup.py first.")
        return trig_l == 1 and trig_r == 1

    def pixel_to_original(self, x_display, y_display, is_right=False):
        if is_right:
            x_display = x_display - self.display_width
        scale_x = self.frame_width / self.display_width
        scale_y = self.frame_height / self.display_height
        return int(x_display * scale_x), int(y_display * scale_y)

    def _triangulate_click_pair(self, pt_left, pt_right):
        tri = self.triangulator
        pt_l_rect = tri._rectify_point(
            pt_left, tri.cmtx0, tri.dist0, tri.R_rect0, tri.P_rect0)
        pt_r_rect = tri._rectify_point(
            pt_right, tri.cmtx1, tri.dist1, tri.R_rect1, tri.P_rect1)
        disp = pt_l_rect[0] - pt_r_rect[0]
        if disp <= 0:
            print("  ERROR: Negative disparity — left/right points may be swapped")
            return None, None
        pos = tri.triangulate(pt_l_rect, pt_r_rect)
        if pos[2] <= 0:
            print("  ERROR: Point behind camera")
            return None, None
        err_l, err_r = tri._reprojection_error(pos, pt_l_rect, pt_r_rect)
        reproj = max(err_l, err_r)
        return tuple(pos), reproj

    def reload_camera_pose(self):
        """Reload CAM_POSE_* constants from the module (re-import)."""
        import importlib
        import trajectory.trajectory_predictor as tp_mod
        importlib.reload(tp_mod)
        self.predictor.set_camera_pose(
            tp_mod.CAM_POSE_X_MM, tp_mod.CAM_POSE_Y_MM, tp_mod.CAM_POSE_Z_MM,
            tp_mod.CAM_POSE_YAW, tp_mod.CAM_POSE_PITCH, tp_mod.CAM_POSE_ROLL)
        print("  Camera pose reloaded from trajectory_predictor.py")

    # ================================================================
    # MOUSE CALLBACK
    # ================================================================

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        is_right = x >= self.display_width
        orig_x, orig_y = self.pixel_to_original(x, y, is_right)

        if self.click_stage == 0:
            if is_right:
                return
            self.click_points = [(orig_x, orig_y)]
            self.click_stage = 1
            print(f"  LEFT click: ({orig_x}, {orig_y}) — now click SAME point on RIGHT")

        elif self.click_stage == 1:
            if not is_right:
                return
            self.click_points.append((orig_x, orig_y))
            self.click_stage = 2
            print(f"  RIGHT click: ({orig_x}, {orig_y})")

            # Triangulate and transform
            pos, reproj = self._triangulate_click_pair(
                self.click_points[0], self.click_points[1])

            if pos is not None:
                cx, cy, cz = pos
                rx, ry, rz = self.predictor.cam_to_robot(cx, cy, cz)
                in_ws = self.predictor.check_workspace(rx, ry, rz)

                n = len(self.measurements) + 1
                self.measurements.append({
                    'n': n,
                    'cam_x': round(cx, 2), 'cam_y': round(cy, 2), 'cam_z': round(cz, 2),
                    'robot_x': round(rx, 1), 'robot_y': round(ry, 1), 'robot_z': round(rz, 1),
                    'reproj': round(reproj, 2),
                    'in_workspace': in_ws,
                })

                ws_str = "IN WORKSPACE" if in_ws else "OUT OF RANGE"
                print()
                print(f"  ┌─ POINT #{n} ──────────────────────────────────────────┐")
                print(f"  │                                                       │")
                print(f"  │  Camera (cm):                                         │")
                print(f"  │    X = {cx:+8.2f}   (along table length)              │")
                print(f"  │    Y = {cy:+8.2f}   (vertical, down=+)               │")
                print(f"  │    Z = {cz:+8.2f}   (across table width)             │")
                print(f"  │                                                       │")
                print(f"  │  Robot (mm):     ← COMPARE THESE TO TAPE MEASURE     │")
                print(f"  │    X = {rx:+8.1f}   (lateral, robot's right=+)        │")
                print(f"  │    Y = {ry:+8.1f}   (toward net=+)                   │")
                print(f"  │    Z = {rz:+8.1f}   (vertical, up=+)                 │")
                print(f"  │                                                       │")
                print(f"  │  {ws_str:^14}   Reproj: {reproj:.1f}px               │")
                print(f"  │                                                       │")
                print(f"  │  Home ref: ({ROBOT_HOME[0]:.0f}, {ROBOT_HOME[1]:.0f}, "
                      f"{ROBOT_HOME[2]:.0f}) mm                       │")
                print(f"  └───────────────────────────────────────────────────────┘")
                print(f"  'r' = reset and click another point")
            else:
                print("  ERROR: Triangulation failed — try clicking more precisely")

    # ================================================================
    # DISPLAY
    # ================================================================

    def _draw_overlay(self, left_small, right_small):
        h = self.display_height
        scale_x = self.display_width / self.frame_width
        scale_y = self.display_height / self.frame_height

        # Header
        freeze_label = " [FROZEN]" if self.frozen else ""
        for img, label in [(left_small, "LEFT"), (right_small, "RIGHT")]:
            cv2.rectangle(img, (0, 0), (320, 38), (0, 0, 0), -1)
            cv2.putText(img, f"{label} | ROBOT TRANSFORM VERIFY{freeze_label}",
                        (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            cv2.putText(img, f"{self._fps:.0f} fps", (5, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Click instructions
        if self.click_stage == 0:
            cv2.rectangle(left_small, (0, 0),
                          (self.display_width - 1, h - 1), (0, 255, 255), 3)
            cv2.putText(left_small, "Click TARGET on LEFT image",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 2)
        elif self.click_stage == 1:
            cv2.rectangle(right_small, (0, 0),
                          (self.display_width - 1, h - 1), (0, 255, 255), 3)
            cv2.putText(left_small, "Click SAME point on RIGHT image",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 2)

        # Draw click markers
        panels = [left_small, right_small]
        labels = ["L", "R"]
        for i, pt in enumerate(self.click_points[:2]):
            img = panels[i]
            px = int(pt[0] * scale_x)
            py = int(pt[1] * scale_y)
            cv2.circle(img, (px, py), 25, (255, 255, 0), 2)
            cv2.drawMarker(img, (px, py), (0, 255, 0),
                           cv2.MARKER_CROSS, 30, 2)
            cv2.circle(img, (px, py), 3, (0, 0, 255), -1)
            cv2.putText(img, labels[i], (px + 28, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show result on screen
        if self.click_stage >= 2 and self.measurements:
            last = self.measurements[-1]
            rx, ry, rz = last['robot_x'], last['robot_y'], last['robot_z']
            cx, cy, cz = last['cam_x'], last['cam_y'], last['cam_z']
            in_ws = last['in_workspace']

            # Camera coords on left panel
            box_y = h - 55
            cv2.rectangle(left_small, (0, box_y), (self.display_width, h), (0, 0, 0), -1)
            cv2.putText(left_small,
                        f"Cam: X={cx:+.1f} Y={cy:+.1f} Z={cz:.1f} cm",
                        (10, box_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(left_small,
                        f"Reproj: {last['reproj']:.1f}px",
                        (10, box_y + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

            # Robot coords on right panel (BIG, this is the main output)
            box_y = h - 75
            cv2.rectangle(right_small, (0, box_y), (self.display_width, h), (0, 0, 0), -1)
            cv2.putText(right_small,
                        f"Robot: X={rx:+.0f}  Y={ry:+.0f}  Z={rz:+.0f} mm",
                        (10, box_y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            ws_clr = (0, 255, 0) if in_ws else (0, 0, 255)
            ws_txt = "IN WORKSPACE" if in_ws else "OUT OF RANGE"
            cv2.putText(right_small, ws_txt,
                        (10, box_y + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, ws_clr, 1)
            cv2.putText(right_small, "'r'=reset  'u'=reload pose",
                        (10, box_y + 63),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (150, 150, 150), 1)

        # Measurement count
        if self.measurements:
            cv2.putText(right_small,
                        f"Points: {len(self.measurements)}",
                        (self.display_width - 110, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # ================================================================
    # SUMMARY
    # ================================================================

    def print_summary(self):
        if not self.measurements:
            print("\n  No measurements yet.\n")
            return

        bar = "═" * 72
        print(f"\n  {bar}")
        print(f"  ROBOT TRANSFORM VERIFICATION — {len(self.measurements)} points")
        print(f"  {bar}")
        print(f"  {'#':>3}  {'Cam X':>7} {'Cam Y':>7} {'Cam Z':>7}  │  "
              f"{'Rob X':>7} {'Rob Y':>7} {'Rob Z':>7}  │ {'Reproj':>6} {'WS':>3}")
        print(f"  {'─'*3}  {'─'*7} {'─'*7} {'─'*7}  │  "
              f"{'─'*7} {'─'*7} {'─'*7}  │ {'─'*6} {'─'*3}")

        for m in self.measurements:
            ws = "OK" if m['in_workspace'] else "OUT"
            print(f"  {m['n']:3d}  {m['cam_x']:+7.1f} {m['cam_y']:+7.1f} {m['cam_z']:7.1f}  │  "
                  f"{m['robot_x']:+7.0f} {m['robot_y']:+7.0f} {m['robot_z']:+7.0f}  │ "
                  f"{m['reproj']:6.1f} {ws:>3}")

        print(f"  {bar}")
        print(f"  Compare 'Rob X/Y/Z' against tape-measure values.")
        print(f"  If they disagree, update CAM_POSE_* in trajectory_predictor.py")
        print(f"  and press 'u' to reload.\n")

    # ================================================================
    # RUN
    # ================================================================

    def run(self):
        print("\n" + "=" * 60)
        print("  ROBOT TRANSFORM VERIFICATION")
        print("=" * 60)
        print(f"  Click on objects at known positions relative to the robot.")
        print(f"  Compare displayed Robot X/Y/Z against tape-measure values.")
        print(f"  Controls: SPACE=freeze  r=reset  u=reload pose  p=summary  q=quit")
        print("=" * 60)

        # Print current camera pose
        print(f"\n  Current camera pose:")
        print(f"    Position: ({CAM_POSE_X_MM}, {CAM_POSE_Y_MM}, {CAM_POSE_Z_MM}) mm")
        print(f"    Yaw={CAM_POSE_YAW}°  Pitch={CAM_POSE_PITCH}°  Roll={CAM_POSE_ROLL}°")
        if CAM_POSE_X_MM == 0 and CAM_POSE_Y_MM == 0 and CAM_POSE_Z_MM == 0:
            print(f"    ⚠  POSITION IS ALL ZEROS — values are placeholders!")
            print(f"    ⚠  Robot coords will be wrong until you measure and update.")
        print()

        if not self.check_calibration():
            return

        try:
            self.triangulator = StereoTriangulator(
                calibration_dir=self.calibration_dir,
                cam_left_id=self.cam_left_id, cam_right_id=self.cam_right_id)
        except Exception as e:
            print(f"ERROR: {e}")
            return

        try:
            self.triangulator.start_cameras(self.frame_width, self.frame_height)
        except RuntimeError as e:
            print(f"ERROR: {e}")
            return

        self.verify_trigger_sync()

        cv2.namedWindow('Robot Transform Verify')
        cv2.setMouseCallback('Robot Transform Verify', self.on_mouse)

        print("\n  Ready! Freeze frame (SPACE), then click target.\n")

        try:
            while True:
                # Grab frames
                if not self.triangulator.cap_left.grab():
                    continue
                if not self.triangulator.cap_right.grab():
                    continue
                ret_l, frame_l = self.triangulator.cap_left.retrieve()
                ret_r, frame_r = self.triangulator.cap_right.retrieve()
                if not ret_l or not ret_r or frame_l is None or frame_r is None:
                    continue

                # FPS
                self._fps_n += 1
                if self._fps_n % 30 == 0:
                    now = time.perf_counter()
                    self._fps = 30.0 / (now - self._fps_t)
                    self._fps_t = now

                # Use frozen frame if frozen
                if self.frozen:
                    show_l = self.frozen_left.copy()
                    show_r = self.frozen_right.copy()
                else:
                    show_l = cv2.resize(frame_l, (self.display_width, self.display_height))
                    show_r = cv2.resize(frame_r, (self.display_width, self.display_height))

                self._draw_overlay(show_l, show_r)
                combined = np.hstack([show_l, show_r])
                cv2.imshow('Robot Transform Verify', combined)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    if not self.frozen:
                        self.frozen = True
                        self.frozen_left = cv2.resize(
                            frame_l, (self.display_width, self.display_height))
                        self.frozen_right = cv2.resize(
                            frame_r, (self.display_width, self.display_height))
                        print("  [FROZEN] — click target now")
                    else:
                        self.frozen = False
                        print("  [UNFROZEN]")
                elif key == ord('r'):
                    self.click_points = []
                    self.click_stage = 0
                    print("  [RESET] — click next point")
                elif key == ord('u'):
                    self.reload_camera_pose()
                    # Re-transform all existing measurements with new pose
                    for m in self.measurements:
                        rx, ry, rz = self.predictor.cam_to_robot(
                            m['cam_x'], m['cam_y'], m['cam_z'])
                        m['robot_x'] = round(rx, 1)
                        m['robot_y'] = round(ry, 1)
                        m['robot_z'] = round(rz, 1)
                        m['in_workspace'] = self.predictor.check_workspace(rx, ry, rz)
                    if self.measurements:
                        print(f"  Re-transformed {len(self.measurements)} existing points")
                        self.print_summary()
                elif key == ord('p'):
                    self.print_summary()

        except KeyboardInterrupt:
            pass
        finally:
            self.triangulator.stop_cameras()
            cv2.destroyAllWindows()

        if self.measurements:
            self.print_summary()
        print("\nDone!")


def main():
    RobotTransformVerifier().run()


if __name__ == '__main__':
    main()
