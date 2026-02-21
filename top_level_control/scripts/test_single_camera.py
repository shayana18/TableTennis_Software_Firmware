"""
Test Single Camera - MOG2 Ball Detection

Test ball detection on a single camera using MOG2 background subtraction.
The detector is lighting-invariant and requires no manual threshold tuning.

CAMERA: Arducam OV9782 Global Shutter USB Camera
        1MP, 100fps @ 1280x800 MJPG

CONTROLS:
    q - Quit
    d - Toggle debug view (show foreground mask)
    b - Reset background model
    c - Cycle camera (0, 1, 2...)
"""

import cv2
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracking.ball_tracker import EnhancedBallTracker
from config.camera_config import load_camera_settings, configure_camera


class SingleCameraTester:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        cam_settings = load_camera_settings()
        self.camera_id = cam_settings['camera0']
        self.cap = None
        self.tracker = EnhancedBallTracker()

        self.frame_width = cam_settings['frame_width']
        self.frame_height = cam_settings['frame_height']

        self.show_debug = False

    def start_camera(self):
        """Open camera with Arducam configuration."""
        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            print(f"Failed to open camera {self.camera_id}")
            return False

        settings = configure_camera(self.cap, self.frame_width, self.frame_height)

        print(f"\nCamera {self.camera_id} opened:")
        print(f"  Resolution: {settings['width']}x{settings['height']}")
        print(f"  FPS: {settings['fps']:.0f}")
        print(f"  FOURCC: {settings['fourcc']}")

        if not settings['settings_match']:
            print("  WARNING: Resolution mismatch!")

        return True

    def run(self):
        """Main loop."""
        print("\n" + "=" * 60)
        print("SINGLE CAMERA TEST - MOG2 Detection (Arducam OV9782)")
        print("=" * 60)
        print(f"\nResolution: {self.frame_width}x{self.frame_height} (MJPG)")
        print("\nCONTROLS:")
        print("  q - Quit")
        print("  d - Toggle debug view (foreground mask)")
        print("  b - Reset background model")
        print("  c - Cycle camera ID")
        print("=" * 60)

        if not self.start_camera():
            return

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                result = self.tracker.detect(frame, return_debug=self.show_debug)

                vis = frame.copy()

                # Warmup progress
                if not self.tracker.is_ready():
                    progress = self.tracker.get_warmup_progress()
                    cv2.putText(vis, f"Warming up... {progress*100:.0f}%", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                elif result['found']:
                    center = result['center']
                    radius = int(result['radius'])
                    cv2.circle(vis, center, radius, (0, 255, 0), 2)
                    cv2.circle(vis, center, 3, (0, 0, 255), -1)
                    cv2.putText(vis, f"Found: ({center[0]}, {center[1]})", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(vis, "No ball detected", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.putText(vis, f"Camera: {self.camera_id}", (10, vis.shape[0] - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(vis, "q:quit d:debug b:reset-bg c:cycle-cam", (10, vis.shape[0] - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

                display_width = 720
                display_height = int(display_width * self.frame_height / self.frame_width)
                vis = cv2.resize(vis, (display_width, display_height))

                cv2.imshow('Single Camera Test', vis)

                if self.show_debug and 'debug' in result and result['debug']:
                    debug = result['debug']
                    h, w = 150, 200

                    masks = []
                    for mask, label in [
                        (debug.get('fg_mask'), 'FG Mask'),
                        (result['mask'], 'Final Mask')
                    ]:
                        if mask is not None:
                            m = cv2.resize(mask, (w, h))
                            m = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
                            cv2.putText(m, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            masks.append(m)

                    if masks:
                        debug_view = cv2.hconcat(masks)
                        cv2.imshow('Debug Masks', debug_view)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break

                elif key == ord('d'):
                    self.show_debug = not self.show_debug
                    if not self.show_debug:
                        cv2.destroyWindow('Debug Masks')
                    print(f"\n[DEBUG] {'Enabled' if self.show_debug else 'Disabled'}")

                elif key == ord('b'):
                    self.tracker.reset_background()
                    print("\n[BG RESET] Background model reset, warming up...")

                elif key == ord('c'):
                    self.camera_id = (self.camera_id + 1) % 5
                    self.tracker.reset_background()
                    self.start_camera()

        except KeyboardInterrupt:
            pass

        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()

        print("\nDone!")


def main():
    tester = SingleCameraTester()
    tester.run()


if __name__ == '__main__':
    main()
