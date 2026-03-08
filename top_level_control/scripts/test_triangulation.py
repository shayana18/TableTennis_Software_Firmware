"""
Test Triangulation - Stereo 3D Ball Tracking

Tests stereo calibration by detecting the ball in both cameras
and triangulating to get 3D position.

DETECTION: Uses shared BallDetector (tracking/ball_detector.py)
           MOG2 background subtraction — lighting invariant

CONTROLS:
    q - Quit
    s - Save current 3D measurement
    r - Reset/clear measurements
    d - Toggle debug view (show fg masks + candidates)
    b - Reset background model
    SPACE - Skip background warmup

VERIFICATION:
    1. Hold ball at known distance (measure with tape)
    2. Press 's' to save measurement
    3. Z should match tape measure within ~5%
    4. Hold ball still — Z jitter should be < 2-3 units at close range
"""

import cv2
import sys
import os
import numpy as np
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracking.stereo_triangulator import StereoTriangulator
from config.camera_config import load_camera_settings


class TriangulationTester:
    """Full-featured triangulation tester with debug mode."""

    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.calibration_dir = os.path.join(
            self.script_dir, '..', 'camera_calibration', 'camera_parameters')

        cam_settings = load_camera_settings()
        self.frame_width = cam_settings['frame_width']
        self.frame_height = cam_settings['frame_height']
        self.cam_left_id = cam_settings['camera0']
        self.cam_right_id = cam_settings['camera1']

        self.triangulator = None
        self.measurements = []
        self.show_debug = False
        self.reject_counts = {}

    def check_calibration(self):
        """Check if calibration files exist."""
        required_files = [
            'camera0_intrinsics.dat', 'camera1_intrinsics.dat',
            'camera0_rot_trans.dat', 'camera1_rot_trans.dat'
        ]
        missing = [f for f in required_files
                   if not os.path.exists(os.path.join(self.calibration_dir, f))]

        if missing:
            print("\nERROR: Missing calibration files!")
            for f in missing:
                print(f"  - {f}")
            print("\nRun calibration first.")
            return False
        return True

    def build_background_phase(self):
        """
        Explicit background learning phase.
        Shows preview and learns background for 3 seconds.
        """
        print("\n  Remove ball from view. Learning background...")
        print("  Press SPACE to skip.\n")

        t_start = time.time()
        duration = 3.0

        while time.time() - t_start < duration:
            # grab/retrieve for sync
            if not self.triangulator.cap_left.grab():
                continue
            if not self.triangulator.cap_right.grab():
                continue
            _, frame_l = self.triangulator.cap_left.retrieve()
            _, frame_r = self.triangulator.cap_right.retrieve()

            if frame_l is None or frame_r is None:
                continue

            self.triangulator.build_background(frame_l, frame_r)

            # Show progress
            elapsed = time.time() - t_start
            progress = min(elapsed / duration, 1.0)
            disp_l = cv2.resize(frame_l, (480, 300))
            disp_r = cv2.resize(frame_r, (480, 300))
            cv2.putText(disp_l, "LEFT", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(disp_r, "RIGHT", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            bar_w = int(progress * 440)
            cv2.rectangle(disp_l, (20, 270), (20 + bar_w, 285), (0, 255, 255), -1)
            cv2.rectangle(disp_l, (20, 270), (460, 285), (200, 200, 200), 1)
            cv2.putText(disp_l, f"Background: {progress*100:.0f}%", (20, 265),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            combined = np.hstack([disp_l, disp_r])
            cv2.imshow('Stereo Triangulation', combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                print("  Skipped.")
                break
            elif key == ord('q'):
                return False

        print("  Background ready!\n")
        return True

    def create_debug_view(self, result):
        """Create debug visualization with fg masks and all candidates."""
        h, w = 180, 240

        def make_mask_vis(mask, label, det, cands, rejected):
            if mask is None:
                return np.zeros((h, w, 3), dtype=np.uint8)
            m = cv2.resize(mask, (w, h))
            m = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
            cv2.putText(m, label, (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            # Scale factor for drawing
            if result['left_frame'] is not None:
                fh, fw = result['left_frame'].shape[:2]
                sx, sy = w / fw, h / fh
            else:
                sx, sy = 1, 1

            # Draw rejected in red
            for rej in rejected:
                rx = int(rej['center'][0] * sx)
                ry = int(rej['center'][1] * sy)
                cv2.drawMarker(m, (rx, ry), (0, 0, 200),
                               cv2.MARKER_TILTED_CROSS, 5, 1)

            # Draw candidates in yellow
            for cand in cands:
                cx = int(cand['center'][0] * sx)
                cy = int(cand['center'][1] * sy)
                cv2.circle(m, (cx, cy), 4, (0, 255, 255), 1)

            # Draw best in green
            if det is not None:
                dx = int(det['center'][0] * sx)
                dy = int(det['center'][1] * sy)
                cv2.circle(m, (dx, dy), 6, (0, 255, 0), 2)

            info = f"P:{len(cands)} R:{len(rejected)}"
            cv2.putText(m, info, (5, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            return m

        left_mask = make_mask_vis(
            result['left_mask'], "LEFT MASK",
            result['left_detection'],
            result['left_all_candidates'],
            result['left_rejected'])

        right_mask = make_mask_vis(
            result['right_mask'], "RIGHT MASK",
            result['right_detection'],
            result['right_all_candidates'],
            result['right_rejected'])

        debug_view = np.hstack([left_mask, right_mask])

        # Rejection stats bar at bottom
        if self.reject_counts:
            stats_img = np.zeros((25, debug_view.shape[1], 3), dtype=np.uint8)
            stats_text = "  ".join(f"{k}:{v}" for k, v in
                                  sorted(self.reject_counts.items(),
                                         key=lambda x: -x[1])[:5])
            cv2.putText(stats_img, f"Rejects: {stats_text}", (5, 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 255), 1)
            debug_view = np.vstack([debug_view, stats_img])

        return debug_view

    def run(self):
        """Main run loop."""
        print("\n" + "=" * 60)
        print("STEREO TRIANGULATION TEST")
        print("=" * 60)
        print(f"  Left=ID{self.cam_left_id}, Right=ID{self.cam_right_id}")
        print(f"  Resolution: {self.frame_width}x{self.frame_height}")
        print("\n  q:quit  s:save  r:reset  d:debug  b:reset-bg")
        print("=" * 60)

        if not self.check_calibration():
            return

        try:
            self.triangulator = StereoTriangulator(
                calibration_dir=self.calibration_dir,
                cam_left_id=self.cam_left_id,
                cam_right_id=self.cam_right_id)
        except Exception as e:
            print(f"\nERROR initializing triangulator: {e}")
            return

        try:
            self.triangulator.start_cameras(self.frame_width, self.frame_height)
        except RuntimeError as e:
            print(f"\nERROR: {e}")
            return

        # Background learning phase
        if not self.build_background_phase():
            self.triangulator.stop_cameras()
            cv2.destroyAllWindows()
            return

        print("--- LIVE TRACKING ---")
        print("(Move ball in front of cameras)\n")

        fps_counter = 0
        fps_timer = time.perf_counter()
        actual_fps = 0

        try:
            while True:
                result = self.triangulator.update()

                if result['left_frame'] is None or result['right_frame'] is None:
                    continue

                # Track rejection reasons
                if result['reject_reason']:
                    reason = result['reject_reason'].split('(')[0]
                    self.reject_counts[reason] = self.reject_counts.get(reason, 0) + 1

                # FPS
                fps_counter += 1
                if fps_counter % 30 == 0:
                    actual_fps = 30.0 / (time.perf_counter() - fps_timer)
                    fps_timer = time.perf_counter()

                left_vis, right_vis = self.triangulator.draw_results(result)

                # Resize for display
                dw = 640
                dh = int(dw * self.frame_height / self.frame_width)
                left_vis = cv2.resize(left_vis, (dw, dh))
                right_vis = cv2.resize(right_vis, (dw, dh))

                # Status bar
                status = "TRACKING" if result['found_3d'] else "SEARCHING..."
                color = (0, 255, 0) if result['found_3d'] else (0, 0, 255)
                cv2.putText(left_vis, f"FPS:{actual_fps:.0f} | {status}",
                            (10, dh - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                l_ok = result['left_detection'] is not None
                r_ok = result['right_detection'] is not None
                cv2.putText(right_vis, f"L:{'OK' if l_ok else '--'} R:{'OK' if r_ok else '--'}",
                            (10, dh - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                cv2.putText(left_vis, "q:quit d:debug s:save b:reset-bg",
                            (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

                combined = np.hstack([left_vis, right_vis])
                cv2.imshow('Stereo Triangulation', combined)

                if self.show_debug:
                    debug_view = self.create_debug_view(result)
                    cv2.imshow('Debug', debug_view)

                # Terminal output
                if result['found_3d']:
                    X, Y, Z = result['position_3d']
                    disp = result['disparity']
                    reproj = result['reproj_err']
                    print(f"\r  3D: X={X:7.1f} Y={Y:7.1f} Z={Z:7.1f}  "
                          f"disp={disp:5.1f}px  reproj={reproj:4.1f}px", end='')
                elif result['reject_reason']:
                    print(f"\r  REJECTED: {result['reject_reason']:40s}", end='')

                # Key handling
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break

                elif key == ord('s'):
                    if result['found_3d']:
                        X, Y, Z = result['position_3d']
                        self.measurements.append((X, Y, Z))
                        print(f"\n\n  [SAVED] #{len(self.measurements)}: "
                              f"X={X:.1f}, Y={Y:.1f}, Z={Z:.1f}")
                    else:
                        print("\n\n  [ERROR] No ball detected — cannot save")

                elif key == ord('r'):
                    self.measurements = []
                    self.reject_counts = {}
                    print("\n\n  [RESET] Cleared all measurements")

                elif key == ord('d'):
                    self.show_debug = not self.show_debug
                    if not self.show_debug:
                        cv2.destroyWindow('Debug')
                    print(f"\n  [DEBUG] {'ON' if self.show_debug else 'OFF'}")

                elif key == ord('b'):
                    self.triangulator.reset_background()
                    self.reject_counts = {}
                    print("\n  [BG RESET] Warming up...")
                    self.build_background_phase()

        except KeyboardInterrupt:
            print("\n\nInterrupted.")

        finally:
            self.triangulator.stop_cameras()
            cv2.destroyAllWindows()

        # Print summary
        if self.measurements:
            print("\n\n" + "=" * 60)
            print("MEASUREMENT SUMMARY")
            print("=" * 60)
            for i, (X, Y, Z) in enumerate(self.measurements, 1):
                print(f"  #{i}: X={X:7.1f}  Y={Y:7.1f}  Z={Z:7.1f}")

            Z_values = [m[2] for m in self.measurements]
            print(f"\n  Depth (Z): mean={np.mean(Z_values):.1f}  "
                  f"std={np.std(Z_values):.1f}  "
                  f"range={max(Z_values)-min(Z_values):.1f}")

            if len(self.measurements) > 1:
                X_values = [m[0] for m in self.measurements]
                Y_values = [m[1] for m in self.measurements]
                print(f"  X: mean={np.mean(X_values):.1f}  std={np.std(X_values):.1f}")
                print(f"  Y: mean={np.mean(Y_values):.1f}  std={np.std(Y_values):.1f}")
            print("=" * 60)

        if self.reject_counts:
            print("\nRejection breakdown:")
            for reason, count in sorted(self.reject_counts.items(), key=lambda x: -x[1]):
                print(f"  {reason}: {count}")

        print("\nDone!")


def main():
    tester = TriangulationTester()
    tester.run()


if __name__ == '__main__':
    main()