"""
Camera Sync Verification Script
================================
ArduCam OV9782 x2 with STM32 External Trigger

Configure cameras externally (AMCAP etc.) BEFORE running this.
This script ONLY runs verification tests:
  0. PREVIEW — confirm both cameras streaming
  1. FREEZE TEST — confirm trigger mode is active
  2. TIMESTAMP TEST — measure inter-camera grab delay (200 pairs)
  3. VISUAL TEST — capture frame pairs to check sync

Uses grab()/retrieve() for accurate sync measurement.

Usage:
  python verify_sync.py
"""

import cv2
import numpy as np
import time
import os

# ====================================================================
#  SETTINGS — EDIT THESE
# ====================================================================
CAM_LEFT   = 0
CAM_RIGHT  = 2
RES_W      = 640
RES_H      = 480
FPS        = 100
# ====================================================================


def open_camera(cam_id, name):
    """Open camera with MJPG, print actual info."""
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RES_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RES_H)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ret, test_frame = cap.read()
    if ret:
        actual_h, actual_w = test_frame.shape[:2]
    else:
        actual_w, actual_h = -1, -1

    rb_fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    rb_fourcc_str = "".join([chr((rb_fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

    print(f"  {name} (device {cam_id}): actual={actual_w}x{actual_h} fourcc={rb_fourcc_str} "
          f"trigger={'ON' if int(cap.get(cv2.CAP_PROP_BACKLIGHT))==1 else 'OFF'} "
          f"exposure={int(cap.get(cv2.CAP_PROP_EXPOSURE))}")
    return cap


def grab_stereo_pair(cap_l, cap_r):
    """Grab both then retrieve both — ensures same trigger pulse."""
    if not cap_l.grab() or not cap_r.grab():
        return False, None, None
    ret_l, f_l = cap_l.retrieve()
    ret_r, f_r = cap_r.retrieve()
    if not ret_l or not ret_r:
        return False, None, None
    return True, f_l, f_r


def draw_header(img, text):
    cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def main():
    print("=" * 64)
    print("  CAMERA SYNC VERIFICATION")
    print("  Configure cameras externally before running this.")
    print("=" * 64)

    os.makedirs("sync_captures", exist_ok=True)

    print()
    cap_l = open_camera(CAM_LEFT, "LEFT")
    cap_r = open_camera(CAM_RIGHT, "RIGHT")

    # ============================================================
    # TEST 0: PREVIEW
    # ============================================================
    print(f"\n{'='*64}")
    print("  TEST 0: PREVIEW — verify both cameras streaming")
    print("  Press SPACE to continue, 'q' to quit.")
    print(f"{'='*64}")

    while True:
        ok, f_l, f_r = grab_stereo_pair(cap_l, cap_r)
        if not ok:
            continue
        disp_l = cv2.resize(f_l, (480, 300))
        disp_r = cv2.resize(f_r, (480, 300))
        draw_header(disp_l, "LEFT")
        draw_header(disp_r, "RIGHT")
        combined = np.hstack([disp_l, disp_r])
        cv2.putText(combined, "PREVIEW — SPACE to continue, 'q' to quit",
                    (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("Sync Verification", combined)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            break
        elif key == ord('q'):
            cap_l.release(); cap_r.release(); cv2.destroyAllWindows(); return

    # ============================================================
    # TEST 1: FREEZE TEST
    # ============================================================
    print(f"\n{'='*64}")
    print("  TEST 1: FREEZE TEST — is trigger mode active?")
    print("  1. DISCONNECT the STM32 trigger wire")
    print("  2. Press SPACE")
    print("  3. Cameras FREEZE = trigger active, keep streaming = NOT active")
    print(f"{'='*64}")

    while True:
        ok, f_l, f_r = grab_stereo_pair(cap_l, cap_r)
        if ok:
            disp_l = cv2.resize(f_l, (480, 300))
            disp_r = cv2.resize(f_r, (480, 300))
            draw_header(disp_l, "LEFT")
            draw_header(disp_r, "RIGHT")
            combined = np.hstack([disp_l, disp_r])
            cv2.putText(combined, "DISCONNECT trigger wire, then press SPACE",
                        (80, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow("Sync Verification", combined)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            break
        elif key == ord('q'):
            cap_l.release(); cap_r.release(); cv2.destroyAllWindows(); return

    print("\n  Reading frames for 3 seconds with no trigger...")
    frames_received = 0
    t_start = time.perf_counter()
    timeout = 3.0

    while time.perf_counter() - t_start < timeout:
        if cap_l.grab() and cap_r.grab():
            cap_l.retrieve()
            cap_r.retrieve()
            frames_received += 1
        elapsed = time.perf_counter() - t_start
        blank = np.zeros((300, 960, 3), dtype=np.uint8)
        bar_w = int((elapsed / timeout) * 900)
        cv2.rectangle(blank, (30, 140), (30 + bar_w, 160), (0, 255, 255), -1)
        cv2.putText(blank, f"Checking... {elapsed:.1f}s / {timeout:.0f}s   Frames: {frames_received}",
                    (250, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Sync Verification", blank)
        cv2.waitKey(1)

    print(f"\n  Frames received in {timeout:.0f}s: {frames_received}")
    if frames_received < 5:
        print("  *** PASS — Trigger mode ACTIVE (cameras froze) ***")
    else:
        print(f"  *** FAIL — Cameras still streaming at ~{frames_received/timeout:.0f} fps ***")
        print("  Trigger mode NOT active.")

    print("  Reconnect trigger wire, press SPACE to continue.")
    while True:
        ok, f_l, f_r = grab_stereo_pair(cap_l, cap_r)
        if ok:
            disp_l = cv2.resize(f_l, (480, 300))
            disp_r = cv2.resize(f_r, (480, 300))
            draw_header(disp_l, "LEFT")
            draw_header(disp_r, "RIGHT")
            combined = np.hstack([disp_l, disp_r])
            cv2.putText(combined, "Reconnect trigger, press SPACE",
                        (150, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow("Sync Verification", combined)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            break
        elif key == ord('q'):
            cap_l.release(); cap_r.release(); cv2.destroyAllWindows(); return

    # ============================================================
    # TEST 2: TIMESTAMP TEST
    # ============================================================
    print(f"\n{'='*64}")
    print("  TEST 2: TIMESTAMP TEST — measuring grab delta (200 pairs)")
    print(f"{'='*64}")

    for _ in range(30):
        cap_l.grab(); cap_r.grab()

    grab_deltas_ms  = []
    pair_times_ms   = []
    frame_intervals = []
    last_pair_time  = None

    for i in range(200):
        t0 = time.perf_counter()
        grabbed_l = cap_l.grab()
        t1 = time.perf_counter()
        grabbed_r = cap_r.grab()
        t2 = time.perf_counter()
        ret_l, f_l = cap_l.retrieve()
        ret_r, f_r = cap_r.retrieve()
        t3 = time.perf_counter()

        if grabbed_l and grabbed_r and ret_l and ret_r:
            grab_deltas_ms.append((t2 - t1) * 1000.0)
            pair_times_ms.append((t3 - t0) * 1000.0)
            if last_pair_time is not None:
                frame_intervals.append((t0 - last_pair_time) * 1000.0)
            last_pair_time = t0

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/200 captured...")

    grab_deltas_ms  = np.array(grab_deltas_ms)
    pair_times_ms   = np.array(pair_times_ms)
    frame_intervals = np.array(frame_intervals) if frame_intervals else np.array([0])
    effective_fps   = 1000.0 / np.mean(frame_intervals) if np.mean(frame_intervals) > 0 else 0

    print(f"\n  {'─'*50}")
    print(f"  GRAB DELTA (sync metric)")
    print(f"  {'─'*50}")
    print(f"    Mean:   {np.mean(grab_deltas_ms):7.3f} ms")
    print(f"    Std:    {np.std(grab_deltas_ms):7.3f} ms")
    print(f"    Median: {np.median(grab_deltas_ms):7.3f} ms")
    print(f"    Min:    {np.min(grab_deltas_ms):7.3f} ms")
    print(f"    Max:    {np.max(grab_deltas_ms):7.3f} ms")
    print(f"    P95:    {np.percentile(grab_deltas_ms, 95):7.3f} ms")
    print(f"  {'─'*50}")
    print(f"  FRAME TIMING")
    print(f"  {'─'*50}")
    print(f"    Pair time (grab+decode): {np.mean(pair_times_ms):7.2f} ms avg")
    print(f"    Frame interval:          {np.mean(frame_intervals):7.2f} ms avg")
    print(f"    Effective stereo FPS:    {effective_fps:7.1f}")
    print(f"  {'─'*50}")

    ball_speed_ms = 10.0
    error_mean_mm = ball_speed_ms * np.mean(grab_deltas_ms)
    error_p95_mm  = ball_speed_ms * np.percentile(grab_deltas_ms, 95)

    print(f"\n  ERROR ESTIMATE (ball @ {ball_speed_ms:.0f} m/s):")
    print(f"    Mean delta -> {error_mean_mm:5.1f} mm")
    print(f"    P95 delta  -> {error_p95_mm:5.1f} mm")

    if np.mean(grab_deltas_ms) < 1.0:
        verdict = "EXCELLENT (< 1ms)"
    elif np.mean(grab_deltas_ms) < 2.0:
        verdict = "GOOD (< 2ms)"
    elif np.mean(grab_deltas_ms) < 5.0:
        verdict = "ACCEPTABLE (< 5ms)"
    else:
        verdict = "POOR — debug trigger"
    print(f"  VERDICT: {verdict}")

    print(f"\n  Press SPACE for visual test, 'q' to quit.")
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord(' '):
            break
        elif key == ord('q'):
            cap_l.release(); cap_r.release(); cv2.destroyAllWindows(); return

    # ============================================================
    # TEST 3: VISUAL SYNC TEST
    # ============================================================
    print(f"\n{'='*64}")
    print("  TEST 3: VISUAL TEST — swing fast object across both views")
    print("  'c' = capture pair, 'q' = done")
    print(f"{'='*64}")

    capture_count = 0

    while True:
        ok, f_l, f_r = grab_stereo_pair(cap_l, cap_r)
        if not ok:
            continue

        disp_l = cv2.resize(f_l, (480, 300))
        disp_r = cv2.resize(f_r, (480, 300))

        for y in range(0, 300, 30):
            cv2.line(disp_l, (0, y), (480, y), (0, 60, 0), 1)
            cv2.line(disp_r, (0, y), (480, y), (0, 60, 0), 1)
        for x in range(0, 480, 48):
            cv2.line(disp_l, (x, 0), (x, 300), (0, 60, 0), 1)
            cv2.line(disp_r, (x, 0), (x, 300), (0, 60, 0), 1)

        draw_header(disp_l, "LEFT")
        draw_header(disp_r, "RIGHT")

        combined = np.hstack([disp_l, disp_r])
        cv2.putText(combined, f"'c' = capture ({capture_count} saved)  'q' = done",
                    (200, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("Sync Verification", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            capture_count += 1
            cv2.imwrite(f"sync_captures/pair{capture_count:03d}_left.png", f_l)
            cv2.imwrite(f"sync_captures/pair{capture_count:03d}_right.png", f_r)

            gray_l = cv2.cvtColor(cv2.resize(f_l, (480, 300)), cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(cv2.resize(f_r, (480, 300)), cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray_l, gray_r)
            diff_color = cv2.applyColorMap(diff * 3, cv2.COLORMAP_JET)
            mean_diff = np.mean(diff)
            cv2.putText(diff_color, f"DIFF #{capture_count} mean={mean_diff:.1f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            diff_combined = np.hstack([
                cv2.resize(disp_l, (320, 200)),
                cv2.resize(diff_color, (320, 200)),
                cv2.resize(disp_r, (320, 200))
            ])
            cv2.imshow(f"Capture #{capture_count}", diff_combined)
            print(f"  Pair #{capture_count} saved (mean diff: {mean_diff:.1f})")

        elif key == ord('q'):
            break

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*64}")
    print("  SUMMARY")
    print(f"{'='*64}")
    print(f"  Grab delta mean:      {np.mean(grab_deltas_ms):.3f} ms")
    print(f"  Grab delta P95:       {np.percentile(grab_deltas_ms, 95):.3f} ms")
    print(f"  Effective stereo FPS: {effective_fps:.1f}")
    print(f"  Error @ 10 m/s:       {error_mean_mm:.1f} mm (mean), {error_p95_mm:.1f} mm (P95)")
    print(f"  Pairs captured:       {capture_count}")
    print(f"  VERDICT:              {verdict}")
    print(f"{'='*64}\n")

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()