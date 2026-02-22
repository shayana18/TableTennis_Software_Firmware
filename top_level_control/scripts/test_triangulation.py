"""
Test Triangulation - Stereo 3D Ball Tracking

Tests stereo calibration by detecting the ball in both cameras
and triangulating to get 3D position. Uses MOG2 background
subtraction for detection (lighting-invariant, no manual tuning).

CAMERA: Arducam OV9782 Global Shutter USB Camera
        1MP, 100fps @ 1280x800 MJPG

FEATURES:
- MOG2 background subtraction detection
- Debug mode shows foreground masks for both cameras
- Measurement logging with statistics

CONTROLS:
    q - Quit
    s - Save current 3D measurement
    r - Reset/clear measurements
    d - Toggle debug view (show fg masks)
    b - Reset background model

UNITS:
    All measurements are in the SAME UNITS as checkerboard_box_size_scale
    - X: Horizontal (+ right, - left) from camera center
    - Y: Vertical (+ down, - up) from camera center
    - Z: Depth (distance from cameras)
"""

import cv2
import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracking.stereo_triangulator import StereoTriangulator
from config.camera_config import load_camera_settings, configure_camera


class TriangulationTester:
    """Full-featured triangulation tester with debug mode."""

    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.calibration_dir = os.path.join(self.script_dir, '..', 'camera_calibration', 'camera_parameters')

        cam_settings = load_camera_settings()
        self.frame_width = cam_settings['frame_width']
        self.frame_height = cam_settings['frame_height']

        self.triangulator = None
        self.measurements = []
        self.show_debug = False

    def load_config(self):
        """Load camera IDs and resolution from camera_config."""
        cam_settings = load_camera_settings()
        self.cam_left_id = cam_settings['camera0']
        self.cam_right_id = cam_settings['camera1']
        self.frame_width = cam_settings['frame_width']
        self.frame_height = cam_settings['frame_height']
        print(f"Loaded config: Left=ID{self.cam_left_id}, Right=ID{self.cam_right_id}")
        print(f"  Resolution: {self.frame_width}x{self.frame_height}")

    def check_calibration(self):
        """Check if calibration files exist."""
        required_files = [
            'camera0_intrinsics.dat',
            'camera1_intrinsics.dat',
            'camera0_rot_trans.dat',
            'camera1_rot_trans.dat'
        ]

        missing = []
        for f in required_files:
            path = os.path.join(self.calibration_dir, f)
            if not os.path.exists(path):
                missing.append(f)

        if missing:
            print("\n" + "=" * 60)
            print("ERROR: Missing calibration files!")
            print("=" * 60)
            for f in missing:
                print(f"  - {f}")
            print("\nRun calibration first:")
            print("  cd camera_calibration")
            print("  python calibrate.py calibration_settings.yaml")
            print("=" * 60)
            return False

        return True

    def create_debug_view(self, frame_left, frame_right):
        """Create debug visualization with fg masks for both cameras."""
        result_left = self.triangulator.tracker_left.detect(frame_left, return_debug=True)
        result_right = self.triangulator.tracker_right.detect(frame_right, return_debug=True)

        h, w = 120, 160

        def make_mask_vis(mask, label):
            if mask is None:
                return np.zeros((h, w, 3), dtype=np.uint8)
            m = cv2.resize(mask, (w, h))
            m = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
            cv2.putText(m, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            return m

        left_fg = make_mask_vis(
            result_left.get('debug', {}).get('fg_mask'), "L FG Mask")
        left_final = make_mask_vis(result_left.get('mask'), "L Final")

        right_fg = make_mask_vis(
            result_right.get('debug', {}).get('fg_mask'), "R FG Mask")
        right_final = make_mask_vis(result_right.get('mask'), "R Final")

        top_row = cv2.hconcat([left_fg, left_final])
        bottom_row = cv2.hconcat([right_fg, right_final])
        debug_view = cv2.vconcat([top_row, bottom_row])

        return debug_view

    def print_controls(self):
        """Print control instructions."""
        print("\n" + "=" * 60)
        print("STEREO TRIANGULATION TEST - MOG2 (Arducam OV9782)")
        print("=" * 60)
        print("\nVerify calibration by holding ball at known distances.")
        print(f"Resolution: {self.frame_width}x{self.frame_height} (MJPG)")
        print("\nCONTROLS:")
        print("  q - Quit")
        print("  s - Save current 3D measurement")
        print("  r - Reset/clear measurements")
        print("  d - Toggle debug view (show fg masks)")
        print("  b - Reset background model")
        print("=" * 60)

    def start_cameras_with_arducam_config(self):
        """Start cameras with Arducam OV9782 configuration."""
        self.triangulator.cap_left = cv2.VideoCapture(self.cam_left_id)
        self.triangulator.cap_right = cv2.VideoCapture(self.cam_right_id)

        if not self.triangulator.cap_left.isOpened():
            raise RuntimeError(f"Failed to open left camera (ID: {self.cam_left_id})")
        if not self.triangulator.cap_right.isOpened():
            raise RuntimeError(f"Failed to open right camera (ID: {self.cam_right_id})")

        print("\nConfiguring cameras (Arducam OV9782 MJPG mode):")
        s_left = configure_camera(
            self.triangulator.cap_left, self.frame_width, self.frame_height)
        s_right = configure_camera(
            self.triangulator.cap_right, self.frame_width, self.frame_height)

        print(f"  LEFT:  {s_left['width']}x{s_left['height']} "
              f"@ {s_left['fps']:.0f}fps ({s_left['fourcc']})")
        print(f"  RIGHT: {s_right['width']}x{s_right['height']} "
              f"@ {s_right['fps']:.0f}fps ({s_right['fourcc']})")

    def run(self):
        """Main run loop."""
        self.print_controls()

        self.load_config()

        if not self.check_calibration():
            return

        try:
            self.triangulator = StereoTriangulator(
                calibration_dir=self.calibration_dir,
                cam_left_id=self.cam_left_id,
                cam_right_id=self.cam_right_id
            )
        except Exception as e:
            print(f"\nERROR initializing triangulator: {e}")
            return

        try:
            self.start_cameras_with_arducam_config()
            print("\nCameras started successfully!")
        except RuntimeError as e:
            print(f"\nERROR: {e}")
            return

        print("\n--- LIVE TRACKING ---")
        print("(Move ball in front of cameras)")

        try:
            while True:
                result = self.triangulator.update()

                if result['left_frame'] is None or result['right_frame'] is None:
                    continue

                left_vis, right_vis = self.triangulator.draw_results(result)

                # Warmup status overlay
                warmup = self.triangulator.warmup_status()
                if not warmup['left_ready']:
                    cv2.putText(left_vis, f"Warming up... {warmup['left_progress']*100:.0f}%",
                               (10, left_vis.shape[0] - 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                if not warmup['right_ready']:
                    cv2.putText(right_vis, f"Warming up... {warmup['right_progress']*100:.0f}%",
                               (10, right_vis.shape[0] - 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                display_width = 640
                display_height = int(display_width * self.frame_height / self.frame_width)
                left_vis = cv2.resize(left_vis, (display_width, display_height))
                right_vis = cv2.resize(right_vis, (display_width, display_height))

                status = "TRACKING" if result['found_3d'] else "SEARCHING..."
                color = (0, 255, 0) if result['found_3d'] else (0, 0, 255)
                cv2.putText(left_vis, status, (10, display_height - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                l_det = "L: OK" if result['left_detection']['found'] else "L: --"
                r_det = "R: OK" if result['right_detection']['found'] else "R: --"
                cv2.putText(left_vis, l_det, (10, display_height - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(right_vis, r_det, (10, display_height - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                cv2.putText(left_vis, "q:quit d:debug s:save b:reset-bg",
                           (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

                combined = cv2.hconcat([left_vis, right_vis])

                if self.show_debug:
                    debug_view = self.create_debug_view(
                        result['left_frame'], result['right_frame'])
                    cv2.imshow('Debug Masks', debug_view)

                cv2.imshow('Stereo Triangulation', combined)

                if result['found_3d']:
                    X, Y, Z = result['position_3d']
                    disp = result['disparity']
                    print(f"\r3D: X={X:7.1f}  Y={Y:7.1f}  Z={Z:7.1f}  (disp={disp:5.1f}px)", end='')

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break

                elif key == ord('s'):
                    if result['found_3d']:
                        X, Y, Z = result['position_3d']
                        self.measurements.append((X, Y, Z))
                        print(f"\n\n[SAVED] #{len(self.measurements)}: X={X:.1f}, Y={Y:.1f}, Z={Z:.1f}")
                    else:
                        print("\n\n[ERROR] No ball detected - cannot save")

                elif key == ord('r'):
                    self.measurements = []
                    print("\n\n[RESET] Cleared all measurements")

                elif key == ord('d'):
                    self.show_debug = not self.show_debug
                    if not self.show_debug:
                        cv2.destroyWindow('Debug Masks')
                    print(f"\n[DEBUG] Debug view {'ENABLED' if self.show_debug else 'DISABLED'}")

                elif key == ord('b'):
                    self.triangulator.reset_background()
                    print("\n[BG RESET] Background model reset, warming up...")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")

        finally:
            self.triangulator.stop_cameras()
            cv2.destroyAllWindows()

        if self.measurements:
            print("\n" + "=" * 60)
            print("MEASUREMENT SUMMARY")
            print("=" * 60)
            for i, (X, Y, Z) in enumerate(self.measurements, 1):
                print(f"  #{i}: X={X:7.1f}  Y={Y:7.1f}  Z={Z:7.1f}")

            Z_values = [m[2] for m in self.measurements]
            print(f"\nDepth (Z) Statistics:")
            print(f"  Min:  {min(Z_values):7.1f}")
            print(f"  Max:  {max(Z_values):7.1f}")
            print(f"  Mean: {sum(Z_values)/len(Z_values):7.1f}")
            print(f"  Range: {max(Z_values)-min(Z_values):7.1f}")
            print("=" * 60)

        print("\nDone!")


def main():
    tester = TriangulationTester()
    tester.run()


if __name__ == '__main__':
    main()
