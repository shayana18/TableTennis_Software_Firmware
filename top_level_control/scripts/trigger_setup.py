"""
Camera Trigger Setup — Run Once After Plugging In Cameras
==========================================================

This script enables external trigger mode and manual exposure on both
ArduCam OV9782 cameras. Run this ONCE every time you plug in / reconnect
the cameras, BEFORE running any other scripts (calibration, detection,
triangulation, trajectory prediction, etc.)

What it does:
  1. Opens both cameras via DirectShow
  2. Sets MJPG codec, resolution, FPS
  3. Enables external trigger mode (CAP_PROP_BACKLIGHT = 1)
  4. Sets manual exposure to -6 (shortest, for max trigger rate)
  5. Verifies all settings were accepted
  6. Runs a quick 3-second stream test to confirm frames are flowing
  7. Keeps cameras open briefly so settings persist in the UVC driver

After this script finishes, the trigger settings are "baked in" to the
Windows UVC driver for these cameras. Other scripts can then open the
cameras normally and the trigger mode will still be active.
"""

import cv2
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.camera_config import (
    CAMERA_LEFT_ID, CAMERA_RIGHT_ID, FRAME_WIDTH, FRAME_HEIGHT, FPS, FOURCC
)

EXPOSURE = -6


def setup_camera(cam_id, label):
    """
    Open and fully configure one ArduCam OV9782 camera.

    Returns:
        (cap, settings, all_ok) — VideoCapture, readback dict, success bool
    """
    print(f"\n  [{label}] Opening camera ID {cam_id}...")
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"  [{label}] ERROR: Could not open camera {cam_id}")
        return None, None, False

    # --- Codec, resolution, FPS ---
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # --- Trigger mode ---
    cap.set(cv2.CAP_PROP_BACKLIGHT, 1)

    # --- Manual exposure ---
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)

    # Also try DirectShow-specific auto exposure value
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)

    # --- Read back ---
    settings = {
        'width':         int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height':        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps':           int(cap.get(cv2.CAP_PROP_FPS)),
        'backlight':     int(cap.get(cv2.CAP_PROP_BACKLIGHT)),
        'auto_exposure': cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
        'exposure':      int(cap.get(cv2.CAP_PROP_EXPOSURE)),
    }

    # --- Print results ---
    all_ok = True

    res_ok = (settings['width'] == FRAME_WIDTH and settings['height'] == FRAME_HEIGHT)
    trig_ok = (settings['backlight'] == 1)
    exp_ok = (settings['exposure'] == EXPOSURE)

    print(f"  [{label}] Resolution:  {settings['width']}x{settings['height']}", end="")
    if res_ok:
        print("  [OK]")
    else:
        print(f"  [WARN: expected {FRAME_WIDTH}x{FRAME_HEIGHT}]")
        all_ok = False

    print(f"  [{label}] FPS:         {settings['fps']}")

    print(f"  [{label}] Trigger:     {settings['backlight']}", end="")
    if trig_ok:
        print("  [ON]")
    else:
        print("  [FAIL — not accepted]")
        all_ok = False

    print(f"  [{label}] Exposure:    {settings['exposure']}", end="")
    if exp_ok:
        print("  [OK]")
    else:
        print(f"  [WARN: expected {EXPOSURE}]")
        all_ok = False

    print(f"  [{label}] Auto-exp:    {settings['auto_exposure']}")

    return cap, settings, all_ok


def stream_test(cap_l, cap_r, duration=3.0):
    """
    Quick stream test: grab/retrieve frame pairs for a few seconds.
    Confirms frames are flowing and measures effective FPS.
    """
    print(f"\n  Running {duration:.0f}s stream test...")

    # Flush buffers
    for _ in range(10):
        cap_l.grab()
        cap_r.grab()

    frame_count = 0
    t_start = time.perf_counter()

    while time.perf_counter() - t_start < duration:
        gl = cap_l.grab()
        gr = cap_r.grab()

        if gl and gr:
            ret_l, f_l = cap_l.retrieve()
            ret_r, f_r = cap_r.retrieve()
            if ret_l and ret_r:
                frame_count += 1

                # Show preview every 10th frame
                if frame_count % 10 == 0:
                    disp_l = cv2.resize(f_l, (320, 200))
                    disp_r = cv2.resize(f_r, (320, 200))
                    cv2.putText(disp_l, "LEFT", (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(disp_r, "RIGHT", (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    combined = np.hstack([disp_l, disp_r])
                    elapsed = time.perf_counter() - t_start
                    cv2.putText(combined, f"Stream test: {elapsed:.1f}s  Pairs: {frame_count}",
                                (10, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    cv2.imshow("Camera Setup", combined)
                    cv2.waitKey(1)

    elapsed = time.perf_counter() - t_start
    fps = frame_count / elapsed if elapsed > 0 else 0

    cv2.destroyAllWindows()

    print(f"  Frames captured: {frame_count} in {elapsed:.1f}s")
    print(f"  Effective stereo FPS: {fps:.1f}")

    return frame_count > 0, fps


def main():
    print("=" * 64)
    print("  ARDUCAM OV9782 — CAMERA TRIGGER SETUP")
    print("  Run this once after plugging in cameras")
    print("=" * 64)
    print(f"\n  Config from camera_config.py:")
    print(f"    LEFT camera:  ID {CAMERA_LEFT_ID}")
    print(f"    RIGHT camera: ID {CAMERA_RIGHT_ID}")
    print(f"    Resolution:   {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"    FPS:          {FPS}")
    print(f"    Codec:        {FOURCC}")
    print(f"    Exposure:     {EXPOSURE}")

    # --- Setup both cameras ---
    cap_l, settings_l, ok_l = setup_camera(CAMERA_LEFT_ID, "LEFT")
    cap_r, settings_r, ok_r = setup_camera(CAMERA_RIGHT_ID, "RIGHT")

    if cap_l is None or cap_r is None:
        print("\n  FATAL: Could not open one or both cameras.")
        print("  Check USB connections and camera IDs in camera_config.py")
        if cap_l:
            cap_l.release()
        if cap_r:
            cap_r.release()
        sys.exit(1)

    # --- Check trigger mode ---
    if not ok_l or not ok_r:
        print("\n  WARNING: Some settings were not accepted.")
        print("  The trigger mode may need to be set via AMCAP once.")
        print("  After setting it in AMCAP, it usually persists and")
        print("  this script can maintain it on subsequent runs.")

    # --- Stream test ---
    streaming, fps = stream_test(cap_l, cap_r, duration=3.0)

    # --- Keep cameras open briefly to let settings persist ---
    # Some UVC drivers reset settings when the camera is released immediately
    print("\n  Holding cameras open for 2s to persist settings...")
    time.sleep(2.0)

    # --- Release ---
    cap_l.release()
    cap_r.release()

    # --- Final report ---
    print(f"\n{'='*64}")
    print("  SETUP COMPLETE")
    print(f"{'='*64}")

    if streaming and ok_l and ok_r:
        print(f"  LEFT:   Trigger ON, Exposure {settings_l['exposure']}, "
              f"{settings_l['width']}x{settings_l['height']}")
        print(f"  RIGHT:  Trigger ON, Exposure {settings_r['exposure']}, "
              f"{settings_r['width']}x{settings_r['height']}")
        print(f"  Stereo FPS: {fps:.1f}")
        print(f"\n  Cameras are ready. You can now run:")
        print(f"    python camera_calibration/calib.py calibration_settings.yaml")
        print(f"    python scripts/test_triangulation.py")
        print(f"    python -m trajectory.test_trajectory_prediction")
    elif streaming:
        print(f"  Frames are flowing but some settings may not have applied.")
        print(f"  Check trigger mode if triangulation accuracy is poor.")
    else:
        print(f"  NO FRAMES received. Check:")
        print(f"    - STM32 trigger is running (80 Hz, >= 2us pulse)")
        print(f"    - Trigger wire connected to both camera F pins")
        print(f"    - Both G pins to GND")
        print(f"    - Cameras are on separate USB controllers")

    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()