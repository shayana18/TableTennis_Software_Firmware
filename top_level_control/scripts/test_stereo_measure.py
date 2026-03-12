"""
Stereo Object Measurement Tool
===============================
Measure real-world object sizes using stereo calibration.

Two modes:
  [A] AUTO MODE  - Measures detected ball diameter automatically
                   Uses triangulated depth + pixel radius + focal length
  [M] CLICK MODE - Click two points on LEFT, then matching points on RIGHT
                   Triangulates both and computes 3D distance between them

CAMERA: Arducam OV9782 Global Shutter USB Camera
        1MP, 100fps @ 1280x800 MJPG

UNITS: cm (from checkerboard_box_size_scale in calibration)

CONTROLS:
    q     - Quit
    m     - Toggle between AUTO and CLICK mode
    r     - Reset click points (in click mode)
    s     - Save current measurement to log
    p     - Print all saved measurements
    SPACE - Freeze/unfreeze frame (for precise clicking)
"""

import cv2
import sys
import os
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracking.stereo_triangulator import StereoTriangulator
from config.camera_config import load_camera_settings


class StereoMeasureTool:
    """Stereo measurement tool with auto ball sizing and click-to-measure."""

    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.calibration_dir = os.path.join(
            self.script_dir, '..', 'camera_calibration', 'camera_parameters')
        self.thresholds_stereo = os.path.join(
            self.script_dir, '..', 'config', 'ball_thresholds_stereo.json')
        self.thresholds_single = os.path.join(
            self.script_dir, '..', 'config', 'ball_thresholds.json')

        # Camera settings (from calibration_settings.yaml)
        cam_settings = load_camera_settings()
        self.frame_width = cam_settings['frame_width']
        self.frame_height = cam_settings['frame_height']
        self.cam_left_id = cam_settings['camera0']
        self.cam_right_id = cam_settings['camera1']

        # Display
        self.display_width = 640
        self.display_height = 400

        # Components
        self.triangulator = None

        # Mode: 'auto' or 'click'
        self.mode = 'auto'
        self.frozen = False
        self.frozen_left = None
        self.frozen_right = None

        # Click mode state
        # Collect 4 clicks: point A on left, point A on right, point B on left, point B on right
        self.click_points = []  # list of (x, y) in ORIGINAL image coords
        self.click_stage = 0   # 0-3: which click we're expecting

        # Measurements log
        self.measurements = []

        # Latest auto measurement
        self.auto_diameter_cm = None
        self.auto_position_3d = None

        self.load_config()

    def load_config(self):
        """Compute display dimensions from loaded camera settings."""
        self.display_height = int(self.display_width * self.frame_height / self.frame_width)

    def load_thresholds(self):
        """No-op: MOG2 detection doesn't use HSV/LAB thresholds."""
        pass

    def check_calibration(self):
        """Check if calibration files exist."""
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

    def start_cameras(self):
        """Open and configure cameras via StereoTriangulator (CAP_DSHOW + buffer)."""
        self.triangulator.start_cameras(self.frame_width, self.frame_height)

    def pixel_to_original(self, x_display, y_display, is_right=False):
        """Convert display coordinates back to original image coordinates."""
        # Display is two images side by side, each display_width x display_height
        if is_right:
            x_display = x_display - self.display_width
        scale_x = self.frame_width / self.display_width
        scale_y = self.frame_height / self.display_height
        return int(x_display * scale_x), int(y_display * scale_y)

    def on_mouse(self, event, x, y, flags, param):
        """Mouse callback for click mode."""
        if self.mode != 'click' or event != cv2.EVENT_LBUTTONDOWN:
            return

        is_right = x >= self.display_width
        orig_x, orig_y = self.pixel_to_original(x, y, is_right)

        # Validate click is in expected panel
        if self.click_stage in (0, 2):
            # Expect left image click
            if is_right:
                return
            label = "A" if self.click_stage == 0 else "B"
            print(f"  Point {label} LEFT: ({orig_x}, {orig_y})")
        elif self.click_stage in (1, 3):
            # Expect right image click
            if not is_right:
                return
            label = "A" if self.click_stage == 1 else "B"
            print(f"  Point {label} RIGHT: ({orig_x}, {orig_y})")

        self.click_points.append((orig_x, orig_y))
        self.click_stage += 1

    def compute_auto_measurement(self, result):
        """Compute ball diameter from stereo detection."""
        self.auto_diameter_cm = None
        self.auto_position_3d = None

        if not result['found_3d']:
            return

        det = result['left_detection']
        if det is None:
            return

        X, Y, Z = result['position_3d']
        self.auto_position_3d = (X, Y, Z)

        # Compute pixel radius from contour area (BallDetector gives area, not radius)
        radius_px = np.sqrt(det['area'] / np.pi)
        fx = self.triangulator.cmtx0[0, 0]  # focal length in pixels

        # Physical diameter = 2 * (radius_px * Z) / fx
        # This is the pinhole model: pixel_size / focal_length = real_size / depth
        diameter_cm = 2.0 * (radius_px * abs(Z)) / fx
        self.auto_diameter_cm = diameter_cm

    def compute_click_measurement(self):
        """Compute 3D distance between two clicked point pairs."""
        if len(self.click_points) < 4:
            return None

        pt_a_left = self.click_points[0]
        pt_a_right = self.click_points[1]
        pt_b_left = self.click_points[2]
        pt_b_right = self.click_points[3]

        # Undistort + rectify pixel coords before triangulation
        tri = self.triangulator
        pt_a_left_u = tri._rectify_point(pt_a_left, tri.cmtx0, tri.dist0, tri.R_rect0, tri.P_rect0)
        pt_a_right_u = tri._rectify_point(pt_a_right, tri.cmtx1, tri.dist1, tri.R_rect1, tri.P_rect1)
        pt_b_left_u = tri._rectify_point(pt_b_left, tri.cmtx0, tri.dist0, tri.R_rect0, tri.P_rect0)
        pt_b_right_u = tri._rectify_point(pt_b_right, tri.cmtx1, tri.dist1, tri.R_rect1, tri.P_rect1)

        # Triangulate both points
        pos_a = tri.triangulate(pt_a_left_u, pt_a_right_u)
        pos_b = tri.triangulate(pt_b_left_u, pt_b_right_u)

        # 3D Euclidean distance
        distance = np.linalg.norm(pos_a - pos_b)

        return {
            'point_a_3d': pos_a,
            'point_b_3d': pos_b,
            'distance_cm': distance,
            'pt_a_left': pt_a_left,
            'pt_a_right': pt_a_right,
            'pt_b_left': pt_b_left,
            'pt_b_right': pt_b_right,
        }

    def draw_auto_overlay(self, left_small, right_small, result):
        """Draw auto measurement overlay."""
        h = self.display_height

        if result['found_3d'] and self.auto_diameter_cm is not None:
            X, Y, Z = self.auto_position_3d
            diam = self.auto_diameter_cm

            # Measurement box at bottom
            box_y = h - 70
            cv2.rectangle(left_small, (0, box_y), (self.display_width, h), (0, 0, 0), -1)
            cv2.putText(left_small, f"Diameter: {diam:.2f} cm", (10, box_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(left_small, f"Depth: {Z:.1f} cm", (10, box_y + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(left_small, f"3D: ({X:.1f}, {Y:.1f}, {Z:.1f})", (10, box_y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # Draw detection circles on display (scaled)
            for det, img in [(result['left_detection'], left_small),
                             (result['right_detection'], right_small)]:
                if det is not None:
                    cx = int(det['center'][0] * self.display_width / self.frame_width)
                    cy = int(det['center'][1] * self.display_height / self.frame_height)
                    radius_px = np.sqrt(det['area'] / np.pi)
                    r = int(radius_px * self.display_width / self.frame_width)
                    cv2.circle(img, (cx, cy), max(r, 3), (0, 255, 0), 2)
                    cv2.circle(img, (cx, cy), 2, (0, 0, 255), -1)
                    # Show pixel radius
                    cv2.putText(img, f"r={radius_px:.0f}px", (cx + r + 5, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        else:
            cv2.putText(left_small, "Point ball at cameras to measure", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    def draw_click_overlay(self, left_small, right_small):
        """Draw click mode overlay with instructions and points."""
        h = self.display_height
        scale_x = self.display_width / self.frame_width
        scale_y = self.display_height / self.frame_height

        # Instructions
        stage_labels = [
            "Click POINT A on LEFT image",
            "Click POINT A on RIGHT image",
            "Click POINT B on LEFT image",
            "Click POINT B on RIGHT image",
        ]

        if self.click_stage < 4:
            label = stage_labels[self.click_stage]
            # Highlight the target panel
            if self.click_stage in (0, 2):
                cv2.rectangle(left_small, (0, 0),
                              (self.display_width - 1, self.display_height - 1), (0, 255, 255), 3)
            else:
                cv2.rectangle(right_small, (0, 0),
                              (self.display_width - 1, self.display_height - 1), (0, 255, 255), 3)

            cv2.putText(left_small, label, (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # Draw placed points
        colors = [(255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255)]  # A=blue, B=red
        point_labels = ["A", "A", "B", "B"]
        panels = [left_small, right_small, left_small, right_small]

        for i, pt in enumerate(self.click_points):
            img = panels[i]
            px = int(pt[0] * scale_x)
            py = int(pt[1] * scale_y)
            cv2.drawMarker(img, (px, py), colors[i], cv2.MARKER_CROSS, 20, 2)
            cv2.putText(img, point_labels[i], (px + 12, py - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)

        # If all 4 points collected, compute and show measurement
        if self.click_stage >= 4:
            meas = self.compute_click_measurement()
            if meas:
                dist = meas['distance_cm']
                pa = meas['point_a_3d']
                pb = meas['point_b_3d']

                # Draw line between A and B on left image (scaled)
                ax = int(meas['pt_a_left'][0] * scale_x)
                ay = int(meas['pt_a_left'][1] * scale_y)
                bx = int(meas['pt_b_left'][0] * scale_x)
                by = int(meas['pt_b_left'][1] * scale_y)
                cv2.line(left_small, (ax, ay), (bx, by), (0, 255, 0), 2)

                # Measurement box
                box_y = h - 70
                cv2.rectangle(left_small, (0, box_y), (self.display_width, h), (0, 0, 0), -1)
                cv2.putText(left_small, f"Distance: {dist:.2f} cm", (10, box_y + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                cv2.putText(left_small,
                            f"A: ({pa[0]:.1f}, {pa[1]:.1f}, {pa[2]:.1f})", (10, box_y + 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
                cv2.putText(left_small,
                            f"B: ({pb[0]:.1f}, {pb[1]:.1f}, {pb[2]:.1f})", (10, box_y + 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

                cv2.putText(right_small, "Press 'r' to reset, 's' to save", (10, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    def run(self):
        """Main loop."""
        print("\n" + "=" * 60)
        print("STEREO OBJECT MEASUREMENT TOOL")
        print("=" * 60)

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

        self.load_thresholds()

        print(f"\nResolution: {self.frame_width}x{self.frame_height} MJPG")
        print(f"Baseline: {self.triangulator.get_baseline():.2f} cm")
        print(f"Focal length: {self.triangulator.get_focal_length_px():.1f} px")

        print("\nCONTROLS:")
        print("  m     - Toggle AUTO / CLICK mode")
        print("  SPACE - Freeze/unfreeze frame")
        print("  r     - Reset click points")
        print("  s     - Save measurement")
        print("  p     - Print saved measurements")
        print("  q     - Quit")
        print("=" * 60)

        try:
            self.start_cameras()
        except RuntimeError as e:
            print(f"ERROR: {e}")
            return

        cv2.namedWindow('Stereo Measure')
        cv2.setMouseCallback('Stereo Measure', self.on_mouse)

        # FPS tracking
        fps_timestamps = []
        fps_display = 0.0

        try:
            while True:
                # FPS
                t_now = time.perf_counter()
                fps_timestamps.append(t_now)
                if len(fps_timestamps) > 30:
                    fps_timestamps.pop(0)
                if len(fps_timestamps) >= 2:
                    elapsed = fps_timestamps[-1] - fps_timestamps[0]
                    if elapsed > 0:
                        fps_display = (len(fps_timestamps) - 1) / elapsed

                # Capture (or use frozen frames)
                if not self.frozen:
                    result = self.triangulator.update()
                    if result['left_frame'] is None or result['right_frame'] is None:
                        continue
                    self.frozen_left = result['left_frame'].copy()
                    self.frozen_right = result['right_frame'].copy()

                    # Auto measurement
                    if self.mode == 'auto':
                        self.compute_auto_measurement(result)
                else:
                    # Re-detect on frozen frames using BallDetector directly
                    best_l, _, _, _ = self.triangulator.detector_left.detect(self.frozen_left)
                    best_r, _, _, _ = self.triangulator.detector_right.detect(self.frozen_right)
                    result = {
                        'left_frame': self.frozen_left,
                        'right_frame': self.frozen_right,
                        'left_detection': best_l,
                        'right_detection': best_r,
                        'found_3d': False,
                        'position_3d': None,
                    }
                    # Triangulate on frozen (with rectification)
                    if best_l is not None and best_r is not None:
                        tri = self.triangulator
                        pt_l = tri._rectify_point(best_l['center'], tri.cmtx0, tri.dist0, tri.R_rect0, tri.P_rect0)
                        pt_r = tri._rectify_point(best_r['center'], tri.cmtx1, tri.dist1, tri.R_rect1, tri.P_rect1)
                        disp = pt_l[0] - pt_r[0]
                        if disp > 0:
                            pos = tri.triangulate(pt_l, pt_r)
                            if pos[2] > 0:
                                result['found_3d'] = True
                                result['position_3d'] = tuple(pos)
                                result['disparity'] = disp
                    if self.mode == 'auto':
                        self.compute_auto_measurement(result)

                # Resize for display
                left_small = cv2.resize(result['left_frame'], (self.display_width, self.display_height))
                right_small = cv2.resize(result['right_frame'], (self.display_width, self.display_height))

                # Mode-specific overlay
                if self.mode == 'auto':
                    self.draw_auto_overlay(left_small, right_small, result)
                else:
                    self.draw_click_overlay(left_small, right_small)

                # Top info bar on each panel
                mode_label = "AUTO" if self.mode == 'auto' else "CLICK"
                freeze_label = " [FROZEN]" if self.frozen else ""
                for img, cam_label in [(left_small, "LEFT"), (right_small, "RIGHT")]:
                    box_w, box_h = 280, 38
                    overlay = img[0:box_h, 0:box_w].copy()
                    cv2.rectangle(img, (0, 0), (box_w, box_h), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.3, img[0:box_h, 0:box_w], 0.7, 0, img[0:box_h, 0:box_w])
                    cv2.putText(img,
                                f"{cam_label} | {self.frame_width}x{self.frame_height} | {mode_label}{freeze_label}",
                                (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                    cv2.putText(img, f"{fps_display:.1f} fps", (5, 32),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

                combined = cv2.hconcat([left_small, right_small])
                cv2.imshow('Stereo Measure', combined)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break

                elif key == ord('m'):
                    self.mode = 'click' if self.mode == 'auto' else 'auto'
                    self.click_points = []
                    self.click_stage = 0
                    print(f"\n[MODE] Switched to {self.mode.upper()} mode")
                    if self.mode == 'click':
                        print("  Click point A on LEFT, then A on RIGHT,")
                        print("  then point B on LEFT, then B on RIGHT.")

                elif key == ord(' '):
                    self.frozen = not self.frozen
                    print(f"\n[{'FROZEN' if self.frozen else 'LIVE'}]")

                elif key == ord('r'):
                    self.click_points = []
                    self.click_stage = 0
                    print("\n[RESET] Click points cleared")

                elif key == ord('s'):
                    if self.mode == 'auto' and self.auto_diameter_cm is not None:
                        entry = {
                            'type': 'auto_diameter',
                            'diameter_cm': round(self.auto_diameter_cm, 3),
                            'position_3d': [round(v, 2) for v in self.auto_position_3d],
                        }
                        self.measurements.append(entry)
                        print(f"\n[SAVED] Ball diameter: {entry['diameter_cm']:.2f} cm "
                              f"at depth {entry['position_3d'][2]:.1f} cm")

                    elif self.mode == 'click' and self.click_stage >= 4:
                        meas = self.compute_click_measurement()
                        if meas:
                            entry = {
                                'type': 'click_distance',
                                'distance_cm': round(meas['distance_cm'], 3),
                                'point_a_3d': [round(v, 2) for v in meas['point_a_3d']],
                                'point_b_3d': [round(v, 2) for v in meas['point_b_3d']],
                            }
                            self.measurements.append(entry)
                            print(f"\n[SAVED] Distance: {entry['distance_cm']:.2f} cm")
                    else:
                        print("\n[ERROR] No measurement to save")

                elif key == ord('p'):
                    self.print_measurements()

        except KeyboardInterrupt:
            pass

        finally:
            self.triangulator.stop_cameras()
            cv2.destroyAllWindows()

        self.print_measurements()
        print("\nDone!")

    def print_measurements(self):
        """Print all saved measurements."""
        if not self.measurements:
            print("\nNo measurements saved.")
            return

        print("\n" + "=" * 60)
        print("SAVED MEASUREMENTS")
        print("=" * 60)
        for i, m in enumerate(self.measurements, 1):
            if m['type'] == 'auto_diameter':
                print(f"  #{i} [AUTO]  Diameter: {m['diameter_cm']:.2f} cm  "
                      f"(at Z={m['position_3d'][2]:.1f} cm)")
            else:
                print(f"  #{i} [CLICK] Distance: {m['distance_cm']:.2f} cm  "
                      f"A={m['point_a_3d']}  B={m['point_b_3d']}")

        # Stats for repeated measurements
        auto_diams = [m['diameter_cm'] for m in self.measurements if m['type'] == 'auto_diameter']
        if len(auto_diams) > 1:
            print(f"\n  Auto diameter stats (n={len(auto_diams)}):")
            print(f"    Mean:  {np.mean(auto_diams):.2f} cm")
            print(f"    Std:   {np.std(auto_diams):.2f} cm")
            print(f"    Range: {min(auto_diams):.2f} - {max(auto_diams):.2f} cm")

        click_dists = [m['distance_cm'] for m in self.measurements if m['type'] == 'click_distance']
        if len(click_dists) > 1:
            print(f"\n  Click distance stats (n={len(click_dists)}):")
            print(f"    Mean:  {np.mean(click_dists):.2f} cm")
            print(f"    Std:   {np.std(click_dists):.2f} cm")
            print(f"    Range: {min(click_dists):.2f} - {max(click_dists):.2f} cm")

        print("=" * 60)


def main():
    tool = StereoMeasureTool()
    tool.run()


if __name__ == '__main__':
    main()
