"""
Stereo Camera Feed - Raw View
==============================
Opens both cameras side-by-side with NO detection or processing.
Displays actual measured FPS and resolution overlay on each feed.

Use this to verify camera settings, frame rate, and sync before running
detection or triangulation scripts.

CAMERA: Arducam OV9782 Global Shutter USB Camera

CONTROLS:
    q - Quit
"""

import cv2
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.camera_config import load_camera_settings, configure_camera


def main():
    cam_settings = load_camera_settings()
    cam_left_id = cam_settings['camera0']
    cam_right_id = cam_settings['camera1']
    CAMERA_WIDTH = cam_settings['frame_width']
    CAMERA_HEIGHT = cam_settings['frame_height']

    print("=" * 60)
    print("STEREO CAMERA FEED (Raw View)")
    print("=" * 60)
    print(f"  Left camera:  ID {cam_left_id}")
    print(f"  Right camera: ID {cam_right_id}")
    print(f"  Target: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ 100fps MJPG")
    print(f"\nOpening cameras...")

    cap_left = cv2.VideoCapture(cam_left_id, cv2.CAP_DSHOW)
    cap_right = cv2.VideoCapture(cam_right_id, cv2.CAP_DSHOW)

    if not cap_left.isOpened() or not cap_right.isOpened():
        print("ERROR: Failed to open one or both cameras")
        return

    cap_left.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap_right.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    s_left = configure_camera(cap_left, CAMERA_WIDTH, CAMERA_HEIGHT)
    s_right = configure_camera(cap_right, CAMERA_WIDTH, CAMERA_HEIGHT)

    print(f"  LEFT:  {s_left['width']}x{s_left['height']} @ {s_left['fps']:.0f}fps ({s_left['fourcc']})")
    print(f"  RIGHT: {s_right['width']}x{s_right['height']} @ {s_right['fps']:.0f}fps ({s_right['fourcc']})")

    match = (s_left['width'] == s_right['width'] and
             s_left['height'] == s_right['height'] and
             s_left['fourcc'] == s_right['fourcc'])
    if match:
        print(f"\n  OK: Both cameras matched at {s_left['width']}x{s_left['height']} {s_left['fourcc']}")
    else:
        print(f"\n  WARNING: Camera settings do NOT match!")

    print(f"\nPress 'q' to quit.")

    # Display sizing
    display_w = 640
    display_h = int(display_w * CAMERA_HEIGHT / CAMERA_WIDTH)

    # FPS tracking
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

            # grab()/retrieve() ensures both frames come from the same trigger pulse
            if not cap_left.grab() or not cap_right.grab():
                continue
            ret_left, frame_left = cap_left.retrieve()
            ret_right, frame_right = cap_right.retrieve()

            if not ret_left or not ret_right:
                continue

            left_small = cv2.resize(frame_left, (display_w, display_h))
            right_small = cv2.resize(frame_right, (display_w, display_h))

            # FPS + actual resolution overlay on each feed
            fps_text = f"{fps_display:.1f} fps"
            for img, label, s in [(left_small, "LEFT", s_left), (right_small, "RIGHT", s_right)]:
                box_w, box_h = 220, 38
                overlay = img[0:box_h, 0:box_w].copy()
                cv2.rectangle(img, (0, 0), (box_w, box_h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.3, img[0:box_h, 0:box_w], 0.7, 0, img[0:box_h, 0:box_w])
                cv2.putText(img, f"{label} | {s['width']}x{s['height']} {s['fourcc']}", (5, 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
                cv2.putText(img, fps_text, (5, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            combined = cv2.hconcat([left_small, right_small])
            cv2.imshow('Stereo Feed', combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass

    finally:
        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()

    print("Done!")


if __name__ == '__main__':
    main()
