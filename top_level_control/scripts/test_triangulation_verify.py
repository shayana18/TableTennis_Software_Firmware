"""
Triangulation Verification Tool
================================
Verifies stereo triangulation accuracy against known physical measurements.
Tests each axis (X, Y, Z) independently to catch anamorphic scale errors
that self-consistent tests (like ball diameter) cannot detect.

Three verification modes:

  [1] DISTANCE  — Click a target at a known depth from cameras.
                   Compares triangulated Z vs tape-measure ground truth.

  [2] LENGTH    — Click endpoints of a known-length object (ruler/rod).
                   Triangulates both, computes 3D distance and per-axis deltas.

  [3] GRAVITY   — Toss ball, fit Y(t) quadratic, extract measured gravity.
                   Cross-checks spatial calibration against physics.

  [4] DIAMETER  — Continuous ball diameter logging to terminal.
                   Shows ball detected, logs diameter + 3D position per frame.
                   Each unbroken detection sequence is a "pass".
                   Tests calibration CONSISTENCY (not absolute accuracy —
                   Z/fx errors cancel in diameter = 2*r_px*Z/fx).

CAMERA: Arducam OV9782 Global Shutter USB Camera

PREREQUISITE: Run trigger_setup.py first to enable external trigger mode.
              Both cameras must be in trigger mode (CAP_PROP_BACKLIGHT=1)
              with STM32 external trigger providing sync pulses.
              Uses grab()/retrieve() pattern via StereoTriangulator to
              ensure both frames come from the same trigger pulse.

UNITS: cm (from checkerboard_box_size_scale in calibration)

CONTROLS:
    1/2/3   - Switch verification mode
    SPACE   - Freeze/unfreeze frame (for precise clicking)
    r       - Reset current measurement clicks
    s       - Save current measurement
    n       - Next measurement (enter new ground truth)
    p       - Print summary report
    w       - Write results to JSON file
    b       - Reset background (gravity mode)
    q       - Quit
"""

import cv2
import sys
import os
import time
import json
import math
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracking.stereo_triangulator import StereoTriangulator
from config.camera_config import load_camera_settings


# ================================================================
# VERIFICATION RESULT CONTAINERS
# ================================================================

class LengthResult:
    """Single length measurement result from 2 triangulated endpoints."""
    def __init__(self, point_a_3d, point_b_3d, reproj_err_a, reproj_err_b):
        self.point_a_3d = point_a_3d
        self.point_b_3d = point_b_3d

        delta = np.array(point_b_3d) - np.array(point_a_3d)
        self.delta_x = delta[0]
        self.delta_y = delta[1]
        self.delta_z = delta[2]
        self.measured_len = float(np.linalg.norm(delta))
        self.reproj_err = max(reproj_err_a, reproj_err_b)

        # Mid-point depth
        mid = (np.array(point_a_3d) + np.array(point_b_3d)) / 2
        self.depth_z = float(mid[2])

    def to_dict(self):
        return {
            'measured_len': round(self.measured_len, 2),
            'delta_x': round(self.delta_x, 2),
            'delta_y': round(self.delta_y, 2),
            'delta_z': round(self.delta_z, 2),
            'depth_z': round(self.depth_z, 2),
            'point_a_3d': [round(v, 2) for v in self.point_a_3d],
            'point_b_3d': [round(v, 2) for v in self.point_b_3d],
            'reproj_err': round(self.reproj_err, 2),
        }


class GravityResult:
    """Single gravity measurement from a tossed ball."""
    GRAVITY_EXPECTED = 981.0  # cm/s^2

    def __init__(self, positions):
        """
        Args:
            positions: list of (x, y, z, t) tuples
        """
        self.n_points = len(positions)
        self.valid = False
        self.measured_g = 0.0
        self.error_pct = 0.0
        self.r_squared = 0.0
        self.duration = 0.0

        if self.n_points < 4:
            return

        arr = np.array(positions)
        t = arr[:, 3] - arr[0, 3]
        self.duration = float(t[-1])

        if self.duration < 0.05:
            return

        y = arr[:, 1]
        coeffs = np.polyfit(t, y, 2)
        a = coeffs[0]
        fitted = np.polyval(coeffs, t)
        ss_res = np.sum((y - fitted) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        self.r_squared = float(1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0)

        # Y(t) = 0.5*g*t^2 + Vy0*t + Y0  =>  a = 0.5*g  =>  g = 2*a
        self.measured_g = float(2.0 * a)
        self.error_pct = float(
            (self.measured_g - self.GRAVITY_EXPECTED) / self.GRAVITY_EXPECTED * 100)
        self.valid = True

    def to_dict(self):
        return {
            'valid': self.valid,
            'n_points': self.n_points,
            'duration': round(self.duration, 4),
            'measured_g': round(self.measured_g, 1),
            'expected_g': self.GRAVITY_EXPECTED,
            'error_pct': round(self.error_pct, 2),
            'r_squared': round(self.r_squared, 4),
        }


# ================================================================
# MAIN VERIFIER
# ================================================================

class TriangulationVerifier:
    """Interactive stereo triangulation verification tool."""

    MODES = {1: 'DISTANCE', 2: 'LENGTH', 3: 'GRAVITY', 4: 'DIAMETER'}

    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.calibration_dir = os.path.join(
            self.script_dir, '..', 'camera_calibration', 'camera_parameters')

        cam_settings = load_camera_settings()
        self.frame_width = cam_settings['frame_width']
        self.frame_height = cam_settings['frame_height']
        self.cam_left_id = cam_settings['camera0']
        self.cam_right_id = cam_settings['camera1']

        # Display
        self.display_width = 640
        self.display_height = int(
            self.display_width * self.frame_height / self.frame_width)

        self.triangulator = None

        # State
        self.mode = 1  # 1=DISTANCE, 2=LENGTH, 3=GRAVITY
        self.frozen = False
        self.frozen_left = None
        self.frozen_right = None

        # Click state (used by DISTANCE and LENGTH modes)
        self.click_points = []  # (x, y) in original image coords
        self.click_stage = 0

        # Results
        self.distance_results = []
        self.length_results = []
        self.gravity_results = []

        # Gravity mode state
        self._throw_active = False
        self._throw_positions = []
        self._lost_count = 0
        self._throw_count = 0

        # Diameter mode state
        self._diam_tracking = False   # currently seeing ball
        self._diam_log = []           # list of dicts for current pass
        self._diam_all_passes = []    # list of completed passes
        self._diam_pass_count = 0
        self._diam_t0 = None          # perf_counter at first detection

        # Sync verification state
        self.trigger_verified = False

        # FPS
        self._fps_timestamps = []
        self._fps_display = 0.0

    # ================================================================
    # SETUP
    # ================================================================

    def check_calibration(self):
        required = ['camera0_intrinsics.dat', 'camera1_intrinsics.dat',
                     'camera0_rot_trans.dat', 'camera1_rot_trans.dat']
        missing = [f for f in required
                   if not os.path.exists(os.path.join(self.calibration_dir, f))]
        if missing:
            print("ERROR: Missing calibration files:")
            for f in missing:
                print(f"  - {f}")
            return False
        return True

    def verify_trigger_sync(self):
        """Verify both cameras are in external trigger mode (CAP_PROP_BACKLIGHT=1).

        Must be called after start_cameras(). Checks that the STM32
        external trigger is driving both cameras so grab()/retrieve()
        returns frames from the same trigger pulse.

        Returns True if both cameras have trigger active.
        """
        cap_l = self.triangulator.cap_left
        cap_r = self.triangulator.cap_right

        trig_l = int(cap_l.get(cv2.CAP_PROP_BACKLIGHT))
        trig_r = int(cap_r.get(cv2.CAP_PROP_BACKLIGHT))

        print(f"\n  Trigger sync check:")
        print(f"    LEFT  CAP_PROP_BACKLIGHT = {trig_l}  "
              f"{'[OK]' if trig_l == 1 else '[FAIL]'}")
        print(f"    RIGHT CAP_PROP_BACKLIGHT = {trig_r}  "
              f"{'[OK]' if trig_r == 1 else '[FAIL]'}")

        if trig_l != 1 or trig_r != 1:
            print("\n  WARNING: Trigger mode NOT active on one or both cameras!")
            print("  Frames may not be synchronized. Run trigger_setup.py first.")
            print("  Continuing anyway — results may be inaccurate.\n")
            self.trigger_verified = False
            return False

        # Quick frame-flow check: grab a few pairs to confirm trigger is delivering
        pairs_ok = 0
        for _ in range(5):
            gl = cap_l.grab()
            gr = cap_r.grab()
            if gl and gr:
                ret_l, _ = cap_l.retrieve()
                ret_r, _ = cap_r.retrieve()
                if ret_l and ret_r:
                    pairs_ok += 1

        if pairs_ok == 0:
            print("\n  WARNING: No frame pairs received!")
            print("  Check that STM32 trigger is running and connected.")
            return False

        print(f"    Frame pairs OK: {pairs_ok}/5")
        print(f"    Sync: grab()/retrieve() pattern ensures same trigger pulse\n")
        self.trigger_verified = True
        return True

    def pixel_to_original(self, x_display, y_display, is_right=False):
        if is_right:
            x_display = x_display - self.display_width
        scale_x = self.frame_width / self.display_width
        scale_y = self.frame_height / self.display_height
        return int(x_display * scale_x), int(y_display * scale_y)

    # ================================================================
    # TRIANGULATION HELPERS
    # ================================================================

    def _triangulate_click_pair(self, pt_left, pt_right):
        """Triangulate a single point from left+right pixel coords.

        Returns (point_3d, reproj_err) or (None, None) on failure.
        """
        tri = self.triangulator
        pt_l_rect = tri._rectify_point(
            pt_left, tri.cmtx0, tri.dist0, tri.R_rect0, tri.P_rect0)
        pt_r_rect = tri._rectify_point(
            pt_right, tri.cmtx1, tri.dist1, tri.R_rect1, tri.P_rect1)

        disp = pt_l_rect[0] - pt_r_rect[0]
        epipolar_err = abs(pt_l_rect[1] - pt_r_rect[1])

        # Diagnostic: show rectified coords and epipolar quality
        print(f"  [DIAG] Raw  L=({pt_left[0]}, {pt_left[1]})  R=({pt_right[0]}, {pt_right[1]})")
        print(f"  [DIAG] Rect L=({pt_l_rect[0]:.1f}, {pt_l_rect[1]:.1f})  "
              f"R=({pt_r_rect[0]:.1f}, {pt_r_rect[1]:.1f})")
        print(f"  [DIAG] Disparity={disp:.1f}px  Epipolar dy={epipolar_err:.1f}px")
        if epipolar_err > 3.0:
            print(f"  [DIAG] *** HIGH EPIPOLAR ERROR ({epipolar_err:.1f}px) — "
                  f"rectification quality is poor, likely bad intrinsics ***")

        if disp <= 0:
            print("  ERROR: Negative disparity — left/right points may be swapped")
            return None, None

        pos = tri.triangulate(pt_l_rect, pt_r_rect)
        if pos[2] <= 0:
            print("  ERROR: Point behind camera")
            return None, None

        err_l, err_r = tri._reprojection_error(pos, pt_l_rect, pt_r_rect)
        reproj = max(err_l, err_r)
        print(f"  [DIAG] Reproj: L={err_l:.2f}px  R={err_r:.2f}px  max={reproj:.2f}px")

        return tuple(pos), reproj

    # ================================================================
    # DISTANCE MODE (simplified: click → get 3D coords)
    # ================================================================

    def _compute_distance_measurement(self):
        """Triangulate from 2 clicks (left + right). Returns (pos_3d, reproj) or None."""
        if len(self.click_points) < 2:
            return None
        return self._triangulate_click_pair(
            self.click_points[0], self.click_points[1])

    # ================================================================
    # LENGTH MODE
    # ================================================================

    def _compute_length_measurement(self):
        """Compute length measurement from 4 clicks. Auto-called on 4th click."""
        if len(self.click_points) < 4:
            return None

        pos_a, reproj_a = self._triangulate_click_pair(
            self.click_points[0], self.click_points[1])
        pos_b, reproj_b = self._triangulate_click_pair(
            self.click_points[2], self.click_points[3])

        if pos_a is None or pos_b is None:
            return None

        return LengthResult(pos_a, pos_b, reproj_a, reproj_b)

    # ================================================================
    # GRAVITY MODE
    # ================================================================

    def _start_gravity_throw(self):
        self._throw_active = True
        self._throw_positions = []
        self._lost_count = 0
        self._throw_count += 1
        print(f"\n  [THROW #{self._throw_count}] tracking...")

    def _end_gravity_throw(self):
        self._throw_active = False
        n = len(self._throw_positions)
        if n < 4:
            print(f"  [END] insufficient data ({n} points)")
            return

        result = GravityResult(self._throw_positions)
        if not result.valid:
            print(f"  [END] invalid throw (duration={result.duration*1000:.0f}ms)")
            return

        self.gravity_results.append(result)
        g_color = "" if abs(result.error_pct) < 5 else " (!)"
        print(f"  [THROW #{self._throw_count}] "
              f"g={result.measured_g:.0f} cm/s^2 "
              f"({result.error_pct:+.1f}%){g_color}  "
              f"R^2={result.r_squared:.4f}  "
              f"{result.n_points} pts  {result.duration*1000:.0f}ms")

    def _warmup_background(self):
        """Learn background for gravity mode."""
        print("\n  Remove ball from view. Learning background (SPACE=skip)...")
        t0, dur = time.time(), 2.0
        while time.time() - t0 < dur:
            if not self.triangulator.cap_left.grab():
                continue
            if not self.triangulator.cap_right.grab():
                continue
            _, fl = self.triangulator.cap_left.retrieve()
            _, fr = self.triangulator.cap_right.retrieve()
            if fl is None or fr is None:
                continue
            self.triangulator.build_background(fl, fr)

            p = min((time.time() - t0) / dur, 1.0)
            dl = cv2.resize(fl, (self.display_width, self.display_height))
            bw = int(p * (self.display_width - 40))
            cv2.rectangle(dl, (20, self.display_height - 30),
                          (20 + bw, self.display_height - 15), (0, 255, 255), -1)
            cv2.putText(dl, f"BG: {p*100:.0f}%", (20, self.display_height - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.imshow('Triangulation Verify', dl)
            k = cv2.waitKey(1) & 0xFF
            if k == ord(' '):
                break
            elif k == ord('q'):
                return False
        print("  Background ready!")
        return True

    # ================================================================
    # MOUSE CALLBACK
    # ================================================================

    def on_mouse(self, event, x, y, flags, param):
        if self.mode == 3:  # gravity mode has no clicks
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        is_right = x >= self.display_width
        orig_x, orig_y = self.pixel_to_original(x, y, is_right)

        if self.mode == 1:
            # DISTANCE: 2 clicks — left then right, auto-triangulate
            if self.click_stage == 0:
                if is_right:
                    return
                print(f"  Target LEFT: ({orig_x}, {orig_y})")
            elif self.click_stage == 1:
                if not is_right:
                    return
                print(f"  Target RIGHT: ({orig_x}, {orig_y})")
                # Auto-triangulate immediately on second click
                self.click_points.append((orig_x, orig_y))
                self.click_stage += 1
                pos, reproj = self._compute_distance_measurement()
                if pos is not None:
                    X, Y, Z = pos
                    n = len(self.distance_results) + 1
                    self.distance_results.append({
                        'n': n, 'X': round(X, 2), 'Y': round(Y, 2),
                        'Z': round(Z, 2), 'reproj': round(reproj, 2),
                    })
                    print(f"\n  ┌─ POINT #{n} ─────────────────────────┐")
                    print(f"  │  X = {X:+8.2f} cm  (left/right)      │")
                    print(f"  │  Y = {Y:+8.2f} cm  (up/down)         │")
                    print(f"  │  Z = {Z:+8.2f} cm  (depth)           │")
                    print(f"  │  Reproj err: {reproj:.1f} px             │")
                    print(f"  └─────────────────────────────────────┘")
                    print(f"  Press 'r' to reset and click another point")
                return  # already appended above
            else:
                return

        elif self.mode == 2:
            # LENGTH: 4 clicks — A left, A right, B left, B right
            if self.click_stage in (0, 2):
                if is_right:
                    return
                label = "A" if self.click_stage == 0 else "B"
                print(f"  Point {label} LEFT: ({orig_x}, {orig_y})")
            elif self.click_stage in (1, 3):
                if not is_right:
                    return
                label = "A" if self.click_stage == 1 else "B"
                print(f"  Point {label} RIGHT: ({orig_x}, {orig_y})")
            else:
                return

            self.click_points.append((orig_x, orig_y))
            self.click_stage += 1

            # Auto-triangulate on 4th click
            if self.click_stage == 4:
                meas = self._compute_length_measurement()
                if meas:
                    n = len(self.length_results) + 1
                    self.length_results.append(meas)
                    Ax, Ay, Az = meas.point_a_3d
                    Bx, By, Bz = meas.point_b_3d
                    print(f"\n  ┌─ LENGTH #{n} ────────────────────────────────────────┐")
                    print(f"  │  A = ({Ax:+8.2f}, {Ay:+8.2f}, {Az:+8.2f})              │")
                    print(f"  │  B = ({Bx:+8.2f}, {By:+8.2f}, {Bz:+8.2f})              │")
                    print(f"  │                                                      │")
                    print(f"  │  3D Distance = {meas.measured_len:8.2f} cm                      │")
                    print(f"  │  dX={meas.delta_x:+7.2f}  dY={meas.delta_y:+7.2f}  dZ={meas.delta_z:+7.2f}      │")
                    print(f"  │  Mid-depth Z = {meas.depth_z:.1f} cm                        │")
                    print(f"  │  Reproj err: {meas.reproj_err:.1f} px                          │")
                    print(f"  └────────────────────────────────────────────────────┘")
                    print(f"  Press 'r' to reset and measure another object")
                else:
                    print("  ERROR: Triangulation failed — try clicking more precisely")
            return

        self.click_points.append((orig_x, orig_y))
        self.click_stage += 1

    # ================================================================
    # DISPLAY OVERLAYS
    # ================================================================

    def _draw_overlay(self, left_small, right_small, result):
        """Draw mode-specific overlay."""
        h = self.display_height
        scale_x = self.display_width / self.frame_width
        scale_y = self.display_height / self.frame_height

        # Mode label
        mode_name = self.MODES[self.mode]
        freeze_label = " [FROZEN]" if self.frozen else ""
        for img, cam_label in [(left_small, "LEFT"), (right_small, "RIGHT")]:
            box_w, box_h = 300, 38
            cv2.rectangle(img, (0, 0), (box_w, box_h), (0, 0, 0), -1)
            cv2.putText(img,
                        f"{cam_label} | {mode_name}{freeze_label}",
                        (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (255, 255, 255), 1)
            cv2.putText(img, f"{self._fps_display:.1f} fps", (5, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        if self.mode == 1:
            self._draw_distance_overlay(left_small, right_small, scale_x, scale_y)
        elif self.mode == 2:
            self._draw_length_overlay(left_small, right_small, scale_x, scale_y)
        elif self.mode == 3:
            self._draw_gravity_overlay(left_small, right_small, result)
        elif self.mode == 4:
            self._draw_diameter_overlay(left_small, right_small, result)

    def _draw_distance_overlay(self, left_small, right_small, scale_x, scale_y):
        h = self.display_height

        # Instructions
        if self.click_stage == 0:
            cv2.rectangle(left_small, (0, 0),
                          (self.display_width - 1, self.display_height - 1),
                          (0, 255, 255), 3)
            cv2.putText(left_small, "Click TARGET on LEFT image",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 2)
        elif self.click_stage == 1:
            cv2.rectangle(right_small, (0, 0),
                          (self.display_width - 1, self.display_height - 1),
                          (0, 255, 255), 3)
            cv2.putText(left_small, "Click SAME point on RIGHT image",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 2)

        # Draw placed points with prominent visuals
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

        # Show result when both points clicked
        if self.click_stage >= 2 and self.distance_results:
            last = self.distance_results[-1]
            X, Y, Z = last['X'], last['Y'], last['Z']

            box_y = h - 55
            cv2.rectangle(left_small, (0, box_y),
                          (self.display_width, h), (0, 0, 0), -1)
            cv2.putText(left_small,
                        f"X={X:+.1f}  Y={Y:+.1f}  Z={Z:.1f} cm",
                        (10, box_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(left_small,
                        f"Reproj: {last['reproj']:.1f}px",
                        (10, box_y + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(right_small, "'r' = reset and click again",
                        (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Show count of measurements taken
        if self.distance_results:
            cv2.putText(right_small,
                        f"Points: {len(self.distance_results)}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 255), 1)

    def _draw_length_overlay(self, left_small, right_small, scale_x, scale_y):
        h = self.display_height

        stage_labels = [
            "Click POINT A on LEFT image",
            "Click POINT A on RIGHT image",
            "Click POINT B on LEFT image",
            "Click POINT B on RIGHT image",
        ]

        if self.click_stage < 4:
            label = stage_labels[self.click_stage]
            if self.click_stage in (0, 2):
                cv2.rectangle(left_small, (0, 0),
                              (self.display_width - 1, self.display_height - 1),
                              (0, 255, 255), 3)
            else:
                cv2.rectangle(right_small, (0, 0),
                              (self.display_width - 1, self.display_height - 1),
                              (0, 255, 255), 3)
            cv2.putText(left_small, label, (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Draw placed points with prominent visuals
        # A = blue (BGR: 255,0,0), B = red (BGR: 0,0,255)
        colors = [(255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255)]
        point_labels = ["A", "A", "B", "B"]
        panels = [left_small, right_small, left_small, right_small]
        for i, pt in enumerate(self.click_points[:4]):
            img = panels[i]
            px = int(pt[0] * scale_x)
            py = int(pt[1] * scale_y)
            cv2.circle(img, (px, py), 22, colors[i], 2)
            cv2.drawMarker(img, (px, py), colors[i],
                           cv2.MARKER_CROSS, 28, 2)
            cv2.circle(img, (px, py), 3, (0, 255, 255), -1)
            cv2.putText(img, point_labels[i], (px + 25, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)

        # Show result when all 4 points clicked
        if self.click_stage >= 4 and self.length_results:
            meas = self.length_results[-1]

            # Line between A and B on left image
            ax = int(self.click_points[0][0] * scale_x)
            ay = int(self.click_points[0][1] * scale_y)
            bx = int(self.click_points[2][0] * scale_x)
            by = int(self.click_points[2][1] * scale_y)
            cv2.line(left_small, (ax, ay), (bx, by), (0, 255, 0), 2)

            box_y = h - 75
            cv2.rectangle(left_small, (0, box_y),
                          (self.display_width, h), (0, 0, 0), -1)
            cv2.putText(left_small,
                        f"3D Distance = {meas.measured_len:.2f} cm",
                        (10, box_y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(left_small,
                        f"dX:{meas.delta_x:+.1f} dY:{meas.delta_y:+.1f} "
                        f"dZ:{meas.delta_z:+.1f}  Depth:{meas.depth_z:.0f}cm",
                        (10, box_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
            cv2.putText(left_small,
                        f"A:({meas.point_a_3d[0]:.1f},{meas.point_a_3d[1]:.1f},"
                        f"{meas.point_a_3d[2]:.1f})  "
                        f"B:({meas.point_b_3d[0]:.1f},{meas.point_b_3d[1]:.1f},"
                        f"{meas.point_b_3d[2]:.1f})",
                        (10, box_y + 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)
            cv2.putText(right_small, "'r' = reset and measure again",
                        (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Show count
        if self.length_results:
            cv2.putText(right_small,
                        f"Measurements: {len(self.length_results)}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 255), 1)

    def _draw_gravity_overlay(self, left_small, right_small, result):
        h = self.display_height

        # Draw detections
        if result['found_3d']:
            x, y, z = result['position_3d']
            cv2.putText(left_small, f"3D: ({x:.1f}, {y:.1f}, {z:.1f})",
                        (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 0), 1)

        for det, img in [(result['left_detection'], left_small),
                         (result['right_detection'], right_small)]:
            if det is not None:
                cx = int(det['center'][0] * self.display_width / self.frame_width)
                cy = int(det['center'][1] * self.display_height / self.frame_height)
                r = max(5, int(np.sqrt(det['area'] / np.pi)
                               * self.display_width / self.frame_width))
                cv2.circle(img, (cx, cy), r, (0, 255, 0), 2)

        # Throw state
        if self._throw_active:
            n = len(self._throw_positions)
            cv2.putText(left_small, f"TRACKING: {n} pts",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
        else:
            wu = self.triangulator.warmup_status()
            if not wu['ready']:
                cv2.putText(left_small, f"Warmup {wu['progress']*100:.0f}%",
                            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 1)
            else:
                cv2.putText(left_small, "Toss ball to measure gravity",
                            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (200, 200, 200), 1)

        # Latest gravity result
        if self.gravity_results:
            last = self.gravity_results[-1]
            g_color = ((0, 255, 0) if abs(last.error_pct) < 5
                       else (0, 200, 255) if abs(last.error_pct) < 10
                       else (0, 0, 255))
            cv2.putText(right_small,
                        f"Last: g={last.measured_g:.0f} ({last.error_pct:+.1f}%)",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, g_color, 1)

    # ================================================================
    # DIAMETER MODE
    # ================================================================

    def _update_diameter_tracking(self, result):
        """Track ball diameter continuously. Logs to terminal on each detection.

        Each unbroken sequence of detections is a 'pass'. When the ball
        leaves view (20 consecutive lost frames), the pass ends and a
        summary is printed.
        """
        LOST_THRESHOLD = 20

        if result['found_3d'] and result.get('left_detection') is not None:
            det = result['left_detection']
            X, Y, Z = result['position_3d']

            # Diameter from pinhole model
            radius_px = np.sqrt(det['area'] / np.pi)
            fx = self.triangulator.cmtx0[0, 0]
            diameter_cm = 2.0 * (radius_px * abs(Z)) / fx

            t_now = time.perf_counter()

            if not self._diam_tracking:
                # Start new pass
                self._diam_tracking = True
                self._diam_pass_count += 1
                self._diam_t0 = t_now
                self._diam_log = []
                print(f"\n  ┌─ PASS #{self._diam_pass_count} ─────────────"
                      f"──────────────────────────────────────────────┐")
                print(f"  │ {'#':>4}  {'t(s)':>7}  {'Diam':>7}  "
                      f"{'X':>8}  {'Y':>8}  {'Z':>8}  "
                      f"{'r_px':>5}  {'Area':>6}  {'Disp':>5} │")
                print(f"  │ {'─'*4}  {'─'*7}  {'─'*7}  "
                      f"{'─'*8}  {'─'*8}  {'─'*8}  "
                      f"{'─'*5}  {'─'*6}  {'─'*5} │")

            self._diam_lost_count = 0
            t_rel = t_now - self._diam_t0
            n = len(self._diam_log) + 1
            disp = result.get('disparity') or 0

            entry = {
                't': round(t_rel, 4),
                'diameter_cm': round(diameter_cm, 3),
                'X': round(X, 2), 'Y': round(Y, 2), 'Z': round(Z, 2),
                'radius_px': round(radius_px, 1),
                'area': round(det['area'], 0),
                'disparity': round(disp, 1),
            }
            self._diam_log.append(entry)

            print(f"  │ {n:4d}  {t_rel:7.3f}  {diameter_cm:7.2f}  "
                  f"{X:8.1f}  {Y:8.1f}  {Z:8.1f}  "
                  f"{radius_px:5.1f}  {det['area']:6.0f}  {disp:5.1f} │")

        elif self._diam_tracking:
            if not hasattr(self, '_diam_lost_count'):
                self._diam_lost_count = 0
            self._diam_lost_count += 1
            if self._diam_lost_count >= LOST_THRESHOLD:
                self._end_diameter_pass()

    def _end_diameter_pass(self):
        """End current diameter pass and print summary."""
        self._diam_tracking = False
        if not self._diam_log:
            return

        diams = [e['diameter_cm'] for e in self._diam_log]
        depths = [e['Z'] for e in self._diam_log]
        n = len(diams)
        mean_d = np.mean(diams)
        std_d = np.std(diams)
        mean_z = np.mean(depths)
        duration = self._diam_log[-1]['t']

        print(f"  └─ END PASS #{self._diam_pass_count}: "
              f"{n} pts  {duration*1000:.0f}ms  "
              f"diam={mean_d:.2f}+/-{std_d:.2f}cm  "
              f"Z_avg={mean_z:.1f}cm ──┘")

        self._diam_all_passes.append({
            'pass': self._diam_pass_count,
            'n_points': n,
            'duration': round(duration, 4),
            'mean_diameter': round(mean_d, 3),
            'std_diameter': round(std_d, 3),
            'min_diameter': round(min(diams), 3),
            'max_diameter': round(max(diams), 3),
            'mean_depth': round(mean_z, 1),
            'min_depth': round(min(depths), 1),
            'max_depth': round(max(depths), 1),
            'points': self._diam_log,
        })

    def _draw_diameter_overlay(self, left_small, right_small, result):
        """Draw diameter mode overlay."""
        h = self.display_height

        # Draw detection circles
        for det, img in [(result.get('left_detection'), left_small),
                         (result.get('right_detection'), right_small)]:
            if det is not None:
                cx = int(det['center'][0] * self.display_width / self.frame_width)
                cy = int(det['center'][1] * self.display_height / self.frame_height)
                r = max(5, int(np.sqrt(det['area'] / np.pi)
                               * self.display_width / self.frame_width))
                cv2.circle(img, (cx, cy), r, (0, 255, 0), 2)
                cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)

        if result.get('found_3d') and result.get('left_detection') is not None:
            det = result['left_detection']
            X, Y, Z = result['position_3d']
            radius_px = np.sqrt(det['area'] / np.pi)
            fx = self.triangulator.cmtx0[0, 0]
            diam = 2.0 * (radius_px * abs(Z)) / fx

            box_y = h - 65
            cv2.rectangle(left_small, (0, box_y),
                          (self.display_width, h), (0, 0, 0), -1)
            cv2.putText(left_small, f"Diam: {diam:.2f} cm  Z: {Z:.1f} cm",
                        (10, box_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0), 2)
            cv2.putText(left_small,
                        f"3D: ({X:.1f}, {Y:.1f}, {Z:.1f})  r={radius_px:.0f}px",
                        (10, box_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (200, 200, 200), 1)

            if self._diam_tracking:
                cv2.putText(left_small,
                            f"PASS #{self._diam_pass_count}: "
                            f"{len(self._diam_log)} pts",
                            (10, box_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 255, 255), 1)
        else:
            wu = self.triangulator.warmup_status()
            if not wu['ready']:
                cv2.putText(left_small, f"Warmup {wu['progress']*100:.0f}%",
                            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 1)
            else:
                cv2.putText(left_small,
                            "Show ball to cameras — logging to terminal",
                            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (200, 200, 200), 1)

        # Show last pass summary
        if self._diam_all_passes and not self._diam_tracking:
            last = self._diam_all_passes[-1]
            cv2.putText(right_small,
                        f"Last: {last['mean_diameter']:.2f}cm "
                        f"+/-{last['std_diameter']:.2f}  "
                        f"Z={last['mean_depth']:.0f}cm",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 0), 1)

    # ================================================================
    # REPORTING
    # ================================================================

    def print_summary(self):
        print("\n" + "=" * 65)
        print("  TRIANGULATION VERIFICATION SUMMARY")
        print("=" * 65)

        # --- DISTANCE (3D point coords) ---
        if self.distance_results:
            print(f"\n  DISTANCE (3D points) — {len(self.distance_results)} points")
            print(f"  {'#':>4}  {'X(cm)':>8}  {'Y(cm)':>8}  {'Z(cm)':>8}  {'Reproj':>7}")
            print(f"  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*7}")
            for r in self.distance_results:
                print(f"  {r['n']:4d}  {r['X']:+8.2f}  {r['Y']:+8.2f}  "
                      f"{r['Z']:8.2f}  {r['reproj']:7.1f}px")

        # --- LENGTH ---
        if self.length_results:
            print(f"\n  LENGTH — {len(self.length_results)} measurements")
            print(f"  {'#':>4}  {'3D Dist':>8}  {'dX':>8}  {'dY':>8}  "
                  f"{'dZ':>8}  {'Depth':>6}  {'Reproj':>7}")
            print(f"  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*8}  "
                  f"{'─'*8}  {'─'*6}  {'─'*7}")
            for i, r in enumerate(self.length_results, 1):
                print(f"  {i:4d}  {r.measured_len:8.2f}  {r.delta_x:+8.2f}  "
                      f"{r.delta_y:+8.2f}  {r.delta_z:+8.2f}  "
                      f"{r.depth_z:6.0f}  {r.reproj_err:7.1f}px")

        # --- GRAVITY ---
        if self.gravity_results:
            print(f"\n  GRAVITY (dynamic) — {len(self.gravity_results)} throws")
            print(f"  {'#':>4}  {'g(cm/s2)':>9}  {'Err(%)':>7}  "
                  f"{'R^2':>7}  {'Pts':>4}  {'Dur(ms)':>8}")
            print(f"  {'─'*4}  {'─'*9}  {'─'*7}  {'─'*7}  {'─'*4}  {'─'*8}")
            for i, r in enumerate(self.gravity_results, 1):
                print(f"  {i:4d}  {r.measured_g:9.0f}  {r.error_pct:+7.1f}  "
                      f"{r.r_squared:7.4f}  {r.n_points:4d}  "
                      f"{r.duration*1000:8.0f}")

            g_vals = [r.measured_g for r in self.gravity_results]
            g_errs = [r.error_pct for r in self.gravity_results]
            print(f"\n  Gravity: mean={np.mean(g_vals):.0f} cm/s^2  "
                  f"std={np.std(g_vals):.0f}  expected=981")
            print(f"  Error: mean={np.mean(g_errs):+.1f}%  "
                  f"std={np.std(g_errs):.1f}%")

            # Implied lateral scale factor
            alpha = 981.0 / np.mean(g_vals) if np.mean(g_vals) > 0 else 0
            print(f"  Implied lateral scale (alpha): {alpha:.3f}  "
                  f"(1.000 = perfect)")

        # --- OVERALL VERDICT ---
        # --- DIAMETER ---
        if self._diam_all_passes:
            print(f"\n  DIAMETER (consistency) — {len(self._diam_all_passes)} passes")
            print(f"  {'Pass':>5}  {'Pts':>4}  {'Diam':>7}  {'Std':>6}  "
                  f"{'Z_avg':>6}  {'Z_range':>12}")
            print(f"  {'─'*5}  {'─'*4}  {'─'*7}  {'─'*6}  "
                  f"{'─'*6}  {'─'*12}")
            for p in self._diam_all_passes:
                print(f"  {p['pass']:5d}  {p['n_points']:4d}  "
                      f"{p['mean_diameter']:7.2f}  {p['std_diameter']:6.2f}  "
                      f"{p['mean_depth']:6.1f}  "
                      f"{p['min_depth']:.0f}-{p['max_depth']:.0f}cm")

            all_means = [p['mean_diameter'] for p in self._diam_all_passes]
            all_depths = [p['mean_depth'] for p in self._diam_all_passes]
            print(f"\n  Overall diameter: {np.mean(all_means):.2f} +/- "
                  f"{np.std(all_means):.2f} cm  "
                  f"(across {len(all_means)} passes)")
            print(f"  Depth range: {min(all_depths):.0f} - "
                  f"{max(all_depths):.0f} cm")
            print(f"  NOTE: Diameter = 2*r_px*Z/fx. Scale errors in fx and Z")
            print(f"        cancel, so this tests CONSISTENCY, not absolute accuracy.")

        print(f"\n  {'─'*60}")
        all_ok = True
        if self.distance_results:
            print(f"  3D points: {len(self.distance_results)} recorded  "
                  f"(compare against tape measurements)")

        if self.length_results:
            print(f"  Length measurements: {len(self.length_results)} recorded  "
                  f"(compare 3D distances against tape measurements)")

        if self.gravity_results:
            mean_g_err = abs(np.mean([r.error_pct for r in self.gravity_results]))
            g_ok = mean_g_err < 10
            print(f"  Gravity accuracy: {'PASS' if g_ok else 'FAIL'}  "
                  f"(mean err {mean_g_err:.1f}%, threshold 10%)")
            all_ok = all_ok and g_ok

        if self._diam_all_passes:
            all_means = [p['mean_diameter'] for p in self._diam_all_passes]
            diam_std = np.std(all_means) if len(all_means) > 1 else 0
            d_ok = diam_std < 0.5  # consistent within 0.5cm across passes
            print(f"  Diameter consistency: {'PASS' if d_ok else 'FAIL'}  "
                  f"(cross-pass std {diam_std:.2f}cm, threshold 0.5cm)")
            all_ok = all_ok and d_ok

        total = (len(self.distance_results) + len(self.length_results) +
                 len(self.gravity_results) + len(self._diam_all_passes))
        if total == 0:
            print("  No measurements recorded.")
        else:
            print(f"\n  VERDICT: {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
        print("=" * 65)

    def save_json(self):
        """Save all results to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.script_dir, f"verify_{timestamp}.json")

        data = {
            'timestamp': timestamp,
            'calibration_dir': os.path.abspath(self.calibration_dir),
            'resolution': f"{self.frame_width}x{self.frame_height}",
            'trigger_sync_verified': self.trigger_verified,
            'distance_results': self.distance_results,
            'length_results': [r.to_dict() for r in self.length_results],
            'gravity_results': [r.to_dict() for r in self.gravity_results],
            'diameter_passes': self._diam_all_passes,
        }

        # Summary stats
        if self.distance_results:
            data['distance_summary'] = {'n': len(self.distance_results)}
        if self.length_results:
            errs = [r.error_pct for r in self.length_results]
            data['length_summary'] = {
                'n': len(errs),
                'mean_error_pct': round(float(np.mean(errs)), 2),
                'std_error_pct': round(float(np.std(errs)), 2),
                'max_abs_error_pct': round(float(max(abs(e) for e in errs)), 2),
            }
        if self.gravity_results:
            g_vals = [r.measured_g for r in self.gravity_results]
            data['gravity_summary'] = {
                'n': len(g_vals),
                'mean_g': round(float(np.mean(g_vals)), 1),
                'std_g': round(float(np.std(g_vals)), 1),
                'mean_error_pct': round(float(np.mean(
                    [r.error_pct for r in self.gravity_results])), 2),
                'implied_alpha': round(float(981.0 / np.mean(g_vals)), 3)
                if np.mean(g_vals) > 0 else None,
            }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n  Results saved to: {filename}")
        return filename

    # ================================================================
    # MAIN LOOP
    # ================================================================

    def run(self):
        print("\n" + "=" * 65)
        print("  TRIANGULATION VERIFICATION TOOL")
        print("=" * 65)

        if not self.check_calibration():
            return

        try:
            self.triangulator = StereoTriangulator(
                calibration_dir=self.calibration_dir,
                cam_left_id=self.cam_left_id,
                cam_right_id=self.cam_right_id)
        except Exception as e:
            print(f"ERROR initializing triangulator: {e}")
            return

        print(f"\n  Resolution: {self.frame_width}x{self.frame_height}")
        print(f"  Baseline: {self.triangulator.get_baseline():.2f} cm")
        print(f"  Focal length: {self.triangulator.get_focal_length_px():.1f} px")

        # Calibration quality diagnostics
        tri = self.triangulator
        fx0, fy0 = tri.cmtx0[0, 0], tri.cmtx0[1, 1]
        fx1, fy1 = tri.cmtx1[0, 0], tri.cmtx1[1, 1]
        d0 = tri.dist0.flatten()
        d1 = tri.dist1.flatten()
        fx_diff_pct = abs(fx0 - fx1) / min(fx0, fx1) * 100
        print(f"\n  --- Calibration Quality Check ---")
        print(f"  cam0: fx={fx0:.1f}  fy={fy0:.1f}  k1={d0[0]:.3f}  k2={d0[1]:.3f}  k3={d0[4]:.3f}")
        print(f"  cam1: fx={fx1:.1f}  fy={fy1:.1f}  k1={d1[0]:.3f}  k2={d1[1]:.3f}  k3={d1[4]:.3f}")
        print(f"  fx difference: {fx_diff_pct:.1f}% (same lens model should be <3%)")
        if fx_diff_pct > 5.0:
            print(f"  *** WARNING: {fx_diff_pct:.1f}% focal length difference is suspicious! ***")
            print(f"  *** This usually means one camera's intrinsics are poorly calibrated ***")
            print(f"  *** Recalibrate with more images (30+) covering full frame ***")
        if len(d0) >= 5 and abs(d0[4]) > 0.3:
            print(f"  *** WARNING: cam0 k3={d0[4]:.3f} is very high — possible overfitting ***")
        if len(d1) >= 5 and abs(d1[4]) > 0.3:
            print(f"  *** WARNING: cam1 k3={d1[4]:.3f} is very high — possible overfitting ***")

        # Show rectified focal length vs raw (large ratio = heavy zoom from alpha=0)
        rect_f = tri.P_rect0[0, 0]
        raw_f = (fx0 + fx1) / 2
        zoom = rect_f / raw_f
        print(f"  Rectified focal: {rect_f:.0f}px  Raw avg: {raw_f:.0f}px  Zoom: {zoom:.2f}x")
        if zoom > 1.5:
            print(f"  Note: {zoom:.1f}x zoom amplifies click errors — "
                  f"1px click error → {zoom:.1f}px rectified error")

        print("\n  MODES:")
        print("    1 = DISTANCE  (verify Z depth)")
        print("    2 = LENGTH    (verify X/Y/Z scale)")
        print("    3 = GRAVITY   (dynamic cross-check)")
        print("    4 = DIAMETER  (continuous ball diameter logging)")
        print("\n  CONTROLS:")
        print("    1/2/3/4 = switch mode   SPACE = freeze   r = reset")
        print("    n = new measurement     s = save result  p = print summary")
        print("    w = write JSON          b = reset bg     q = quit")
        print("=" * 65)

        try:
            self.triangulator.start_cameras(self.frame_width, self.frame_height)
        except RuntimeError as e:
            print(f"ERROR: {e}")
            return

        self.verify_trigger_sync()

        cv2.namedWindow('Triangulation Verify')
        cv2.setMouseCallback('Triangulation Verify', self.on_mouse)

        # Initial warmup for gravity mode background
        initial_warmup_done = False

        try:
            while True:
                # FPS
                t_now = time.perf_counter()
                self._fps_timestamps.append(t_now)
                if len(self._fps_timestamps) > 30:
                    self._fps_timestamps.pop(0)
                if len(self._fps_timestamps) >= 2:
                    elapsed = self._fps_timestamps[-1] - self._fps_timestamps[0]
                    if elapsed > 0:
                        self._fps_display = (
                            (len(self._fps_timestamps) - 1) / elapsed)

                # Capture
                if not self.frozen or self.mode in (3, 4):
                    result = self.triangulator.update()
                    if result['left_frame'] is None or result['right_frame'] is None:
                        continue
                    if not self.frozen:
                        self.frozen_left = result['left_frame'].copy()
                        self.frozen_right = result['right_frame'].copy()
                else:
                    # Frozen in click modes — reuse stored frames
                    result = {
                        'left_frame': self.frozen_left,
                        'right_frame': self.frozen_right,
                        'left_detection': None,
                        'right_detection': None,
                        'found_3d': False,
                        'position_3d': None,
                        'reject_reason': None,
                    }

                # --- Gravity mode throw tracking ---
                if self.mode == 3:
                    if result['found_3d']:
                        if not self._throw_active:
                            self._start_gravity_throw()
                        x, y, z = result['position_3d']
                        t_stamp = time.perf_counter()
                        self._throw_positions.append(
                            (float(x), float(y), float(z), t_stamp))
                        self._lost_count = 0
                    elif self._throw_active:
                        self._lost_count += 1
                        if (self._lost_count >= 20 and
                                len(self._throw_positions) > 0):
                            self._end_gravity_throw()

                # --- Diameter mode continuous tracking ---
                if self.mode == 4:
                    self._update_diameter_tracking(result)

                # Display
                left_small = cv2.resize(
                    result['left_frame'],
                    (self.display_width, self.display_height))
                right_small = cv2.resize(
                    result['right_frame'],
                    (self.display_width, self.display_height))

                self._draw_overlay(left_small, right_small, result)
                combined = cv2.hconcat([left_small, right_small])
                cv2.imshow('Triangulation Verify', combined)

                # Keys
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break

                elif key == ord('1'):
                    self.mode = 1
                    self.click_points = []
                    self.click_stage = 0
                    self.frozen = False
                    print("\n  [MODE] DISTANCE — click point to get 3D coords")
                    print("  Freeze (SPACE), click on LEFT, then same point on RIGHT")

                elif key == ord('2'):
                    self.mode = 2
                    self.click_points = []
                    self.click_stage = 0
                    self.frozen = False
                    print("\n  [MODE] LENGTH — click endpoints of any object")
                    print("  Freeze (SPACE), click A on LEFT, A on RIGHT, B on LEFT, B on RIGHT")
                    print("  3D distance auto-prints. Then measure the object to compare.")

                elif key == ord('3'):
                    self.mode = 3
                    self.click_points = []
                    self.click_stage = 0
                    self.frozen = False
                    self._throw_active = False
                    self._throw_positions = []
                    self._lost_count = 0
                    print("\n  [MODE] GRAVITY — dynamic cross-check")
                    if not initial_warmup_done:
                        self._warmup_background()
                        initial_warmup_done = True

                elif key == ord('4'):
                    # End any in-progress diameter pass before switching
                    if self._diam_tracking:
                        self._end_diameter_pass()
                    self.mode = 4
                    self.click_points = []
                    self.click_stage = 0
                    self.frozen = False
                    self._diam_tracking = False
                    self._diam_log = []
                    print("\n  [MODE] DIAMETER — continuous ball diameter logging")
                    print("  Show ball to cameras. Each detection pass logs to terminal.")
                    print("  Copy-paste terminal output for analysis.")
                    if not initial_warmup_done:
                        self._warmup_background()
                        initial_warmup_done = True

                elif key == ord(' '):
                    if self.mode not in (3, 4):
                        self.frozen = not self.frozen
                        print(f"  [{'FROZEN' if self.frozen else 'LIVE'}]")

                elif key == ord('r'):
                    self.click_points = []
                    self.click_stage = 0
                    if self.mode == 3:
                        self._throw_active = False
                        self._throw_positions = []
                        self._lost_count = 0
                    elif self.mode == 4:
                        if self._diam_tracking:
                            self._end_diameter_pass()
                        self._diam_tracking = False
                        self._diam_log = []
                    print("  [RESET]")

                elif key == ord('n'):
                    if self.mode in (1, 2):
                        self.click_points = []
                        self.click_stage = 0
                        print("  Ready — click next point")
                    elif self.mode == 3:
                        print("  (Gravity mode: just toss the ball)")

                elif key == ord('s'):
                    self._save_current()

                elif key == ord('p'):
                    self.print_summary()

                elif key == ord('w'):
                    self.save_json()

                elif key == ord('b'):
                    if self.mode in (3, 4):
                        if self._diam_tracking:
                            self._end_diameter_pass()
                        self.triangulator.reset_background()
                        self._throw_active = False
                        self._throw_positions = []
                        self._lost_count = 0
                        self._diam_tracking = False
                        self._diam_log = []
                        print("  [BG RESET]")
                        self._warmup_background()
                        initial_warmup_done = True

        except KeyboardInterrupt:
            pass
        finally:
            # End any in-progress diameter pass
            if self._diam_tracking:
                self._end_diameter_pass()
            self.triangulator.stop_cameras()
            cv2.destroyAllWindows()

        if (self.distance_results or self.length_results or
                self.gravity_results or self._diam_all_passes):
            self.print_summary()
            self.save_json()
        print("\nDone!")

    def _save_current(self):
        """Save the current measurement based on mode."""
        if self.mode == 1:
            print("  (Mode 1 auto-saves on click — press 'p' to see all points)")

        elif self.mode == 2:
            print("  (Mode 2 auto-saves on 4th click — press 'p' to see all)")

        elif self.mode == 3:
            print("  (Gravity measurements auto-save after each throw)")


def main():
    TriangulationVerifier().run()


if __name__ == '__main__':
    main()
