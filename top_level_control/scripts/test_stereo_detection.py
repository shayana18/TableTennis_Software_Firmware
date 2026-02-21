"""
Test Stereo Detection - Ball Detection on Both Cameras

Test ball tracking on both cameras using MOG2 background subtraction.
Detection is lighting-invariant and requires no manual threshold tuning.

CAMERA: Arducam OV9782 Global Shutter USB Camera
        1MP, 100fps @ 1280x800 MJPG

CONTROLS:
    q - Quit
    l - Cycle lighting mode
    d - Toggle debug view (foreground masks for both cameras)
    b - Reset background model
"""

import cv2
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracking.ball_tracker import EnhancedBallTracker
from config.camera_config import (
    load_camera_settings, configure_camera,
    CAMERA_LEFT_ID, CAMERA_RIGHT_ID, FRAME_WIDTH, FRAME_HEIGHT
)


class ArducamStereoCapture:
    """
    Stereo camera capture optimized for Arducam OV9782 cameras.
    All defaults come from camera_config.py.
    """

    def __init__(self, cam_left_id=None, cam_right_id=None):
        self.cam_left_id = cam_left_id if cam_left_id is not None else CAMERA_LEFT_ID
        self.cam_right_id = cam_right_id if cam_right_id is not None else CAMERA_RIGHT_ID
        self.cap_left = None
        self.cap_right = None

    def start_cameras(self, width=None, height=None):
        """Open and configure cameras. Defaults from camera_config.py."""
        if width is None:
            width = FRAME_WIDTH
        if height is None:
            height = FRAME_HEIGHT
        self.cap_left = cv2.VideoCapture(self.cam_left_id)
        self.cap_right = cv2.VideoCapture(self.cam_right_id)

        if not self.cap_left.isOpened():
            raise RuntimeError(f"Failed to open left camera (ID: {self.cam_left_id})")
        if not self.cap_right.isOpened():
            raise RuntimeError(f"Failed to open right camera (ID: {self.cam_right_id})")

        print("\nConfiguring cameras (Arducam OV9782 MJPG mode):")
        s_left = configure_camera(self.cap_left, width, height)
        s_right = configure_camera(self.cap_right, width, height)

        print(f"  LEFT:  {s_left['width']}x{s_left['height']} @ {s_left['fps']:.0f}fps ({s_left['fourcc']})")
        print(f"  RIGHT: {s_right['width']}x{s_right['height']} @ {s_right['fps']:.0f}fps ({s_right['fourcc']})")

        if not s_left['settings_match'] or not s_right['settings_match']:
            print("  WARNING: Some camera settings don't match requested values.")

        print(f"\nCameras started: Left=ID{self.cam_left_id}, Right=ID{self.cam_right_id}")

    def read(self):
        """Read frames from both cameras."""
        ret_left, frame_left = self.cap_left.read()
        ret_right, frame_right = self.cap_right.read()
        return ret_left, frame_left, ret_right, frame_right

    def stop_cameras(self):
        """Release cameras."""
        if self.cap_left:
            self.cap_left.release()
        if self.cap_right:
            self.cap_right.release()
        print("Cameras stopped")


class StereoDetectorIndependent:
    """Stereo detector with independent trackers per camera."""

    def __init__(self, config):
        self.config = config
        self.capture = None

        self.tracker_left = EnhancedBallTracker()
        self.tracker_right = EnhancedBallTracker()

    def start_cameras(self, width, height):
        """Start cameras."""
        cam_left_id = self.config.get('camera_left', {}).get('id', 0)
        cam_right_id = self.config.get('camera_right', {}).get('id', 1)

        self.capture = ArducamStereoCapture(cam_left_id, cam_right_id)
        self.capture.start_cameras(width, height)

    def read_frames(self):
        """Read frames from cameras."""
        return self.capture.read()

    def stop_cameras(self):
        """Stop cameras."""
        if self.capture:
            self.capture.stop_cameras()

    def reset_background(self):
        """Reset MOG2 background model on both trackers."""
        self.tracker_left.reset_background()
        self.tracker_right.reset_background()

    def warmup_status(self):
        """Return warmup progress for both trackers."""
        return {
            'left_ready': self.tracker_left.is_ready(),
            'right_ready': self.tracker_right.is_ready(),
            'left_progress': self.tracker_left.get_warmup_progress(),
            'right_progress': self.tracker_right.get_warmup_progress()
        }


class LightingNormalizer:
    """Lighting normalization techniques."""

    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.clahe_strong = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))

        self.mode = 0
        self.mode_names = ["OFF", "CLAHE", "CLAHE Strong", "White Balance", "Gamma Auto", "Full Pipeline"]

    def cycle_mode(self):
        self.mode = (self.mode + 1) % len(self.mode_names)
        return self.mode_names[self.mode]

    def get_mode_name(self):
        return self.mode_names[self.mode]

    def normalize(self, frame):
        if frame is None:
            return None
        if self.mode == 0:
            return frame
        elif self.mode == 1:
            return self._apply_clahe(frame)
        elif self.mode == 2:
            return self._apply_clahe(frame, strong=True)
        elif self.mode == 3:
            return self._apply_white_balance(frame)
        elif self.mode == 4:
            return self._apply_gamma(frame)
        elif self.mode == 5:
            result = self._apply_white_balance(frame)
            result = self._apply_clahe(result)
            return self._apply_gamma(result)
        return frame

    def _apply_clahe(self, frame, strong=False):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe_strong.apply(l) if strong else self.clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def _apply_white_balance(self, frame):
        result = frame.astype(np.float32)
        for i in range(3):
            avg = np.mean(result[:, :, i])
            if avg > 0:
                result[:, :, i] *= 128.0 / avg
        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_gamma(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        if brightness < 80:
            gamma = 0.6
        elif brightness > 180:
            gamma = 1.5
        else:
            return frame
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(frame, table)


def create_debug_view(frame_left, frame_right, tracker_left, tracker_right):
    """Create debug visualization showing fg masks for both cameras."""
    h, w = 120, 160

    result_left = tracker_left.detect(frame_left, return_debug=True)
    result_right = tracker_right.detect(frame_right, return_debug=True)

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


def main():
    cam_settings = load_camera_settings()
    config = {
        'camera_left': {'id': cam_settings['camera0']},
        'camera_right': {'id': cam_settings['camera1']},
        'frame_width': cam_settings['frame_width'],
        'frame_height': cam_settings['frame_height'],
    }

    FRAME_W = cam_settings['frame_width']
    FRAME_H = cam_settings['frame_height']

    print("\n" + "=" * 60)
    print("STEREO DETECTION TEST - MOG2 (Arducam OV9782)")
    print("=" * 60)
    print(f"\nCamera: Arducam OV9782 Global Shutter")
    print(f"  Left ID:  {cam_settings['camera0']}")
    print(f"  Right ID: {cam_settings['camera1']}")
    print(f"\nResolution: {FRAME_W}x{FRAME_H} (MJPG)")

    print("\nCONTROLS:")
    print("  q - Quit")
    print("  l - Cycle lighting mode")
    print("  d - Toggle debug view (show fg masks)")
    print("  b - Reset background model")
    print("=" * 60)

    detector = StereoDetectorIndependent(config)
    lighting = LightingNormalizer()

    try:
        detector.start_cameras(FRAME_W, FRAME_H)
    except Exception as e:
        print(f"\nERROR: {e}")
        return

    show_debug = False

    fps_timestamps = []
    fps_display = 0.0

    try:
        while True:
            t_now = time.perf_counter()
            fps_timestamps.append(t_now)
            if len(fps_timestamps) > 30:
                fps_timestamps.pop(0)
            if len(fps_timestamps) >= 2:
                elapsed = fps_timestamps[-1] - fps_timestamps[0]
                if elapsed > 0:
                    fps_display = (len(fps_timestamps) - 1) / elapsed

            ret_left, frame_left_raw, ret_right, frame_right_raw = detector.read_frames()

            if not ret_left or not ret_right:
                continue

            frame_left = lighting.normalize(frame_left_raw)
            frame_right = lighting.normalize(frame_right_raw)

            result_left = detector.tracker_left.detect(frame_left)
            result_right = detector.tracker_right.detect(frame_right)

            left_vis = frame_left.copy()
            right_vis = frame_right.copy()

            # Warmup status
            warmup = detector.warmup_status()
            if not warmup['left_ready']:
                cv2.putText(left_vis, f"Warming up... {warmup['left_progress']*100:.0f}%",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            elif result_left['found']:
                center = result_left['center']
                radius = int(result_left['radius'])
                cv2.circle(left_vis, center, radius, (0, 255, 0), 2)
                cv2.circle(left_vis, center, 3, (0, 0, 255), -1)
                cv2.putText(left_vis, f"L: ({center[0]}, {center[1]})", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(left_vis, "L: No ball", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if not warmup['right_ready']:
                cv2.putText(right_vis, f"Warming up... {warmup['right_progress']*100:.0f}%",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            elif result_right['found']:
                center = result_right['center']
                radius = int(result_right['radius'])
                cv2.circle(right_vis, center, radius, (0, 255, 0), 2)
                cv2.circle(right_vis, center, 3, (0, 0, 255), -1)
                cv2.putText(right_vis, f"R: ({center[0]}, {center[1]})", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(right_vis, "R: No ball", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            both_found = result_left['found'] and result_right['found']
            status = "BOTH DETECTED" if both_found else "Need both cameras"
            color = (0, 255, 0) if both_found else (0, 0, 255)
            cv2.putText(left_vis, status, (10, left_vis.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.putText(left_vis, f"Light: {lighting.get_mode_name()}", (10, left_vis.shape[0] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            cv2.putText(right_vis, "q:quit l:light d:debug b:reset-bg", (10, right_vis.shape[0] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            display_width = 640
            display_height = int(display_width * FRAME_H / FRAME_W)
            left_small = cv2.resize(left_vis, (display_width, display_height))
            right_small = cv2.resize(right_vis, (display_width, display_height))

            fps_text = f"{fps_display:.1f} fps"
            cv2.putText(left_small, fps_text, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(right_small, fps_text, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            combined = cv2.hconcat([left_small, right_small])
            cv2.imshow('Stereo Detection', combined)

            if show_debug:
                debug_view = create_debug_view(
                    frame_left, frame_right,
                    detector.tracker_left, detector.tracker_right
                )
                if debug_view is not None:
                    cv2.imshow('Debug Masks', debug_view)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('l'):
                mode = lighting.cycle_mode()
                print(f"[LIGHTING] {mode}")

            elif key == ord('d'):
                show_debug = not show_debug
                if not show_debug:
                    cv2.destroyWindow('Debug Masks')
                print(f"[DEBUG] {'ON' if show_debug else 'OFF'}")

            elif key == ord('b'):
                detector.reset_background()
                print("[BG RESET] Background model reset, warming up...")

    except KeyboardInterrupt:
        print("\n\nInterrupted")

    finally:
        detector.stop_cameras()
        cv2.destroyAllWindows()

    print("\nDone!")


if __name__ == '__main__':
    main()
