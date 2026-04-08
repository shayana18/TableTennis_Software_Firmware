"""
Live Triangulation in Robot Frame
==================================
Detects ball using the production MOG2 pipeline (same BallDetector as
test_integration_simple.py), triangulates in camera frame, transforms
to robot frame via the points-based R/t matrix, and prints live
robot-frame XYZ to terminal.

Use this to verify that triangulated positions match physical
measurements in the robot coordinate system.

CONTROLS:
    q     - Quit
    s     - Save current measurement
    r     - Reset saved measurements
    b     - Reset background model
    d     - Toggle debug view
    SPACE - Skip background warmup
"""

import csv
import cv2
import sys
import os
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracking.stereo_triangulator import StereoTriangulator
from config.camera_config import load_camera_settings
from comm_function.points_based_transform import load_points_based_transform, cam_to_robot


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    calibration_dir = os.path.join(script_dir, '..', 'camera_calibration', 'camera_parameters')

    cam = load_camera_settings()
    frame_w = cam['frame_width']
    frame_h = cam['frame_height']

    # Load camera→robot transform
    tf = load_points_based_transform()
    R = tf["rotation"]
    t_vec = tf["translation"]
    scale = tf["camera_scale_to_robot_units"]

    print("\n" + "=" * 65)
    print("  LIVE TRIANGULATION — ROBOT FRAME (mm)")
    print("=" * 65)
    print(f"  Cameras: L={cam['camera0']}  R={cam['camera1']}  {frame_w}x{frame_h}")
    print(f"  Transform scale: {scale:.4f}")
    print(f"  Controls: q=quit s=save r=reset b=bg d=debug")
    print("=" * 65)

    # Check calibration
    for f in ['camera0_intrinsics.dat', 'camera1_intrinsics.dat',
              'camera0_rot_trans.dat', 'camera1_rot_trans.dat']:
        if not os.path.exists(os.path.join(calibration_dir, f)):
            print(f"ERROR: Missing {f}")
            return

    tri = StereoTriangulator(
        calibration_dir=calibration_dir,
        cam_left_id=cam['camera0'],
        cam_right_id=cam['camera1'],
    )
    tri.start_cameras(frame_w, frame_h)

    # Background warmup
    print("\n  Remove ball. Learning background (SPACE=skip)...")
    t0 = time.time()
    warmup_s = 2.5
    while time.time() - t0 < warmup_s:
        if not tri.cap_left.grab():
            continue
        if not tri.cap_right.grab():
            continue
        _, fl = tri.cap_left.retrieve()
        _, fr = tri.cap_right.retrieve()
        if fl is None or fr is None:
            continue
        tri.build_background(fl, fr)
        prog = min((time.time() - t0) / warmup_s, 1.0)
        vis = cv2.resize(fl, (480, 300))
        bw = int(prog * 440)
        cv2.rectangle(vis, (20, 270), (20 + bw, 285), (0, 255, 255), -1)
        cv2.putText(vis, f"BG: {prog*100:.0f}%", (20, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.imshow("Robot Frame Triangulation", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            break
        if key == ord('q'):
            tri.stop_cameras()
            cv2.destroyAllWindows()
            return
    print("  Background ready.\n")

    measurements = []
    show_debug = False
    fps_count = 0
    fps_time = time.perf_counter()
    fps = 0.0
    t_start = time.perf_counter()
    frame_num = 0

    # --- Throw tracking for CSV export ---
    csv_dir = os.path.join(script_dir, "triangulation_csvs")
    os.makedirs(csv_dir, exist_ok=True)
    throw_count = 0
    throw_frames = []          # frames in the current throw
    last_detection_time = None  # perf_counter time of last 3D detection
    THROW_GAP_S = 0.3           # 300ms without detection = end of throw

    CSV_COLUMNS = [
        "frame", "time_s", "rob_x", "rob_y", "rob_z",
        "cam_x", "cam_y", "cam_z", "disparity", "reproj_err",
    ]

    def save_throw_csv(throw_id, frames):
        if not frames:
            return
        path = os.path.join(csv_dir, f"throw_{throw_id:03d}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for fr in frames:
                writer.writerow(fr)
        print(f"\n  [CSV] Throw #{throw_id} ({len(frames)} frames) -> {path}")

    print("--- LIVE ---  (move ball in front of cameras)\n")
    print(f"  {'Frame':>6} {'Time(s)':>8} {'Robot X':>9} {'Robot Y':>9} {'Robot Z':>9}  "
          f"{'Cam X':>8} {'Cam Y':>8} {'Cam Z':>8} {'Disp':>6} {'Reproj':>7}")
    print(f"  {'─'*6} {'─'*8} {'─'*9} {'─'*9} {'─'*9}  "
          f"{'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*7}")

    try:
        while True:
            result = tri.update()
            if result['left_frame'] is None:
                continue

            frame_num += 1
            fps_count += 1
            if fps_count % 30 == 0:
                fps = 30.0 / max(1e-6, time.perf_counter() - fps_time)
                fps_time = time.perf_counter()

            t_now = time.perf_counter() - t_start

            # Transform to robot frame if 3D found
            rob_pos = None
            if result['found_3d']:
                cx, cy, cz = result['position_3d']
                reproj = result.get('reproj_err', 0)
                if reproj > 4.0:
                    print(f"  {frame_num:>6} {t_now:>8.3f}  REJECTED: reproj({reproj:.2f}px)")
                else:
                    rx, ry, rz = cam_to_robot(R, t_vec, scale, cx, cy, cz)
                    rob_pos = (rx, ry, rz)
                    disp = result.get('disparity', 0)
                    print(f"  {frame_num:>6} {t_now:>8.3f} {rx:>+9.1f} {ry:>+9.1f} {rz:>+9.1f}  "
                          f"{cx:>+8.2f} {cy:>+8.2f} {cz:>+8.2f} {disp:>6.1f} {reproj:>7.2f}px")
            elif result.get('reject_reason'):
                print(f"  {frame_num:>6} {t_now:>8.3f}  REJECTED: {result['reject_reason']}")

            # --- Throw tracking: detect gaps and save CSV ---
            if rob_pos is not None:
                # If there was a gap, save the previous throw first
                if (last_detection_time is not None
                        and (time.perf_counter() - last_detection_time) > THROW_GAP_S
                        and throw_frames):
                    throw_count += 1
                    save_throw_csv(throw_count, throw_frames)
                    throw_frames = []

                cx, cy, cz = result['position_3d']
                throw_frames.append({
                    "frame":      frame_num,
                    "time_s":     round(t_now, 4),
                    "rob_x":      round(rob_pos[0], 1),
                    "rob_y":      round(rob_pos[1], 1),
                    "rob_z":      round(rob_pos[2], 1),
                    "cam_x":      round(cx, 2),
                    "cam_y":      round(cy, 2),
                    "cam_z":      round(cz, 2),
                    "disparity":  round(result.get('disparity', 0), 1),
                    "reproj_err": round(result.get('reproj_err', 0), 2),
                })
                last_detection_time = time.perf_counter()
            else:
                # No detection — check if a throw just ended
                if (last_detection_time is not None
                        and (time.perf_counter() - last_detection_time) > THROW_GAP_S
                        and throw_frames):
                    throw_count += 1
                    save_throw_csv(throw_count, throw_frames)
                    throw_frames = []

            # Visualization
            left_vis, right_vis = tri.draw_results(result)

            dw = 640
            dh = int(dw * frame_h / frame_w)
            left_s = cv2.resize(left_vis, (dw, dh))
            right_s = cv2.resize(right_vis, (dw, dh))

            # Overlay robot-frame coords
            status = "TRACKING" if result['found_3d'] else "SEARCHING"
            color = (0, 255, 0) if result['found_3d'] else (0, 0, 255)
            cv2.putText(left_s, f"FPS:{fps:.0f} {status}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if rob_pos is not None:
                cv2.putText(left_s,
                    f"Robot(mm): X={rob_pos[0]:+.0f} Y={rob_pos[1]:+.0f} Z={rob_pos[2]:+.0f}",
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            cv2.putText(left_s, "q:quit s:save r:reset b:bg d:debug",
                        (10, dh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (150, 150, 150), 1)

            combined = cv2.hconcat([left_s, right_s])
            cv2.imshow("Robot Frame Triangulation", combined)

            # Keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if rob_pos is not None:
                    cx, cy, cz = result['position_3d']
                    measurements.append({
                        'n': len(measurements) + 1,
                        'rob_x': round(rob_pos[0], 1),
                        'rob_y': round(rob_pos[1], 1),
                        'rob_z': round(rob_pos[2], 1),
                        'cam_x': round(cx, 2),
                        'cam_y': round(cy, 2),
                        'cam_z': round(cz, 2),
                        'reproj': round(result.get('reproj_err', 0), 2),
                        'disp': round(result.get('disparity', 0), 1),
                    })
                    m = measurements[-1]
                    print(f"\n  [SAVED #{m['n']}] "
                          f"Robot(mm): X={m['rob_x']:+.1f} Y={m['rob_y']:+.1f} Z={m['rob_z']:+.1f}  "
                          f"Cam(cm): Z={m['cam_z']:.1f}  Reproj={m['reproj']:.2f}px")
                else:
                    print("\n  [ERROR] No ball detected")
            elif key == ord('r'):
                measurements = []
                print("\n  [RESET] Cleared measurements")
            elif key == ord('b'):
                tri.reset_background()
                print("\n  [BG RESET] Learning...")
                t0 = time.time()
                while time.time() - t0 < 2.0:
                    if not tri.cap_left.grab():
                        continue
                    if not tri.cap_right.grab():
                        continue
                    _, fl = tri.cap_left.retrieve()
                    _, fr = tri.cap_right.retrieve()
                    if fl is not None and fr is not None:
                        tri.build_background(fl, fr)
                    if cv2.waitKey(1) & 0xFF == ord(' '):
                        break
                print("  Background ready.")
            elif key == ord('d'):
                show_debug = not show_debug
                if not show_debug:
                    cv2.destroyWindow('Debug')
                print(f"\n  [DEBUG] {'ON' if show_debug else 'OFF'}")

            if show_debug:
                # Show foreground masks
                masks = []
                for mask, label in [(result.get('left_mask'), 'LEFT'),
                                    (result.get('right_mask'), 'RIGHT')]:
                    if mask is not None:
                        m = cv2.resize(mask, (320, 200))
                        m = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
                        cv2.putText(m, label, (5, 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        masks.append(m)
                if len(masks) == 2:
                    cv2.imshow('Debug', cv2.hconcat(masks))

    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    finally:
        # Save any in-progress throw
        if throw_frames:
            throw_count += 1
            save_throw_csv(throw_count, throw_frames)
            throw_frames = []
        tri.stop_cameras()
        cv2.destroyAllWindows()

    # Summary
    if measurements:
        print("\n\n" + "=" * 70)
        print("MEASUREMENT SUMMARY (Robot Frame, mm)")
        print("=" * 70)
        print(f"  {'#':>3} {'Rob_X':>8} {'Rob_Y':>8} {'Rob_Z':>8}  "
              f"{'Cam_Z':>7} {'Reproj':>7}")
        print(f"  {'─'*3} {'─'*8} {'─'*8} {'─'*8}  {'─'*7} {'─'*7}")
        for m in measurements:
            print(f"  {m['n']:>3} {m['rob_x']:>+8.1f} {m['rob_y']:>+8.1f} {m['rob_z']:>+8.1f}  "
                  f"{m['cam_z']:>7.1f} {m['reproj']:>7.2f}px")

        xs = [m['rob_x'] for m in measurements]
        ys = [m['rob_y'] for m in measurements]
        zs = [m['rob_z'] for m in measurements]
        print(f"\n  X: mean={np.mean(xs):+.1f}  std={np.std(xs):.1f}mm")
        print(f"  Y: mean={np.mean(ys):+.1f}  std={np.std(ys):.1f}mm")
        print(f"  Z: mean={np.mean(zs):+.1f}  std={np.std(zs):.1f}mm")
        print("=" * 70)

    print("\nDone!")


if __name__ == '__main__':
    main()
