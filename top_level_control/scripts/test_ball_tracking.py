"""
Ball Tracking — Robot Follows Ball in Real-Time
=================================================
Triangulates ball position using stereo cameras, transforms to robot frame,
and sends the position directly to the robot as a move target.

Detection: MOG2 (from StereoTriangulator) scored with weighted criteria:
  - MOG2 detection quality (reproj, circularity, area) = 80% weight
  - HSV orange color match = 20% weight
  Combined score must exceed threshold to accept.

CONTROLS:
    g     - Toggle tracking gate ON/OFF
    h     - Send robot HOME
    b     - Reset background model
    d     - Toggle HSV debug window (live trackbars + mask view)
    q     - Quit (sends HOME first)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from comm_function.points_based_transform import load_points_based_transform, cam_to_robot
from comm_function.transmit_over_uart import UartComm
from config.camera_config import load_camera_settings
from tracking.stereo_triangulator import StereoTriangulator

# ================================================================
# TUNABLE PARAMETERS
# ================================================================
SEND_INTERVAL_MS = 15       # Minimum ms between UART sends
REPROJ_THRESHOLD = 50.0     # Max reproj error (px) to accept
MOVE_TIME_S      = 0.12     # Time given to robot for each move
Y_SAFETY_OFFSET  = 0.0   # mm — offset Y so robot stays behind the ball
FIXED_Z          = -800   # mm — lock robot Z to this value (ignores triangulated Z)

# Scoring weights (must sum to 1.0)
W_MOG2           = 0.9      # Weight for MOG2 detection quality (reproj, circularity, area)
W_HSV            = 0.1     # Weight for HSV orange color match

# Score threshold — detection must exceed this to be accepted
MIN_SCORE        = 0.4      # 0-1 range. Higher = stricter.

# HSV orange range (tune via debug trackbars, press 'd')
HSV_H_LOW  = 4
HSV_H_HIGH = 21
HSV_S_LOW  = 113
HSV_S_HIGH = 253
HSV_V_LOW  = 0
HSV_V_HIGH = 255

# Morphological kernels for HSV mask
HSV_KERNEL_OPEN  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
HSV_KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
# ================================================================


def _print(*args, **kwargs):
    print("from planner in terminal", *args, **kwargs)


def compute_hsv_score(frame_bgr, detection, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi):
    """Compute what fraction of the detection region is orange in HSV.
    Returns (score 0-1, hsv_mask)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([h_lo, s_lo, v_lo]), np.array([h_hi, s_hi, v_hi]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, HSV_KERNEL_OPEN)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, HSV_KERNEL_CLOSE)

    if detection is None:
        return 0.0, mask

    cx, cy = int(detection["center"][0]), int(detection["center"][1])
    r = max(5, int(np.sqrt(detection.get("area", 100) / np.pi)))
    h, w = mask.shape[:2]
    y1, y2 = max(0, cy - r), min(h, cy + r)
    x1, x2 = max(0, cx - r), min(w, cx + r)

    roi = mask[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0, mask

    return float(np.count_nonzero(roi)) / roi.size, mask


def compute_mog2_score(detection, reproj):
    """Score the MOG2 detection quality (0-1) based on reproj, circularity, area."""
    if detection is None:
        return 0.0

    score = 0.0

    # Reproj: 0px = 1.0, threshold px = 0.0
    reproj_score = max(0.0, 1.0 - reproj / REPROJ_THRESHOLD)
    score += reproj_score * 0.4

    # Circularity: 1.0 = perfect circle
    circ = detection.get("circularity", 0)
    score += min(1.0, circ) * 0.4

    # Area: score peaks at ~400px² (typical ball), drops for very small/large
    area = detection.get("area", 0)
    if 100 <= area <= 1200:
        area_score = 1.0 - abs(area - 400) / 800
        score += max(0.0, area_score) * 0.2

    return score


def main():
    parser = argparse.ArgumentParser(description="Ball tracking — robot follows ball")
    parser.add_argument("--port", default=os.environ.get("STM32_UART_PORT"),
                        help="UART port (or set STM32_UART_PORT)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--home-ack-timeout", type=float, default=30.0)
    parser.add_argument("--warmup-s", type=float, default=2.0)
    parser.add_argument("--quiet-uart", action="store_true")
    args = parser.parse_args()

    if not args.port:
        parser.error("UART port required. Pass --port or set STM32_UART_PORT.")

    cam = load_camera_settings()
    frame_w = cam["frame_width"]
    frame_h = cam["frame_height"]

    calibration_dir = os.path.join(PARENT_DIR, "camera_calibration", "camera_parameters")

    tf = load_points_based_transform()
    R = tf["rotation"]
    t_vec = tf["translation"]
    scale = tf["camera_scale_to_robot_units"]
    _print(f"Loaded transform (scale={scale})")

    for f in ["camera0_intrinsics.dat", "camera1_intrinsics.dat",
              "camera0_rot_trans.dat", "camera1_rot_trans.dat"]:
        if not os.path.exists(os.path.join(calibration_dir, f)):
            _print(f"ERROR: Missing {f}")
            return

    try:
        triangulator = StereoTriangulator(
            calibration_dir=calibration_dir,
            cam_left_id=cam["camera0"],
            cam_right_id=cam["camera1"],
        )
    except Exception as e:
        _print(f"ERROR init triangulator: {e}")
        return

    uart = UartComm(port=args.port, baud_rate=args.baud, verbose=not args.quiet_uart)

    _print("\n" + "=" * 60)
    _print(" BALL TRACKING — MOG2 (80%) + HSV (20%)")
    _print("=" * 60)
    _print(f"Cameras: L={cam['camera0']} R={cam['camera1']}  {frame_w}x{frame_h}")
    _print(f"Send: {SEND_INTERVAL_MS}ms  Reproj: {REPROJ_THRESHOLD}px  MinScore: {MIN_SCORE}")
    _print(f"Weights: MOG2={W_MOG2}  HSV={W_HSV}")
    _print("")
    _print("Controls: g=gate  h=home  b=bg  d=HSV-debug  q=quit")
    _print("=" * 60)

    # --- Home robot ---
    try:
        uart.open()
        uart.clear_input_buffer()
        uart.send_home()
        timeout = None if args.home_ack_timeout == 0 else args.home_ack_timeout
        ack = uart.wait_for_home_confirmation(timeout_s=timeout)
    except Exception as e:
        _print(f"[UART] ERROR: {e}")
        uart.close()
        return
    if ack is None:
        _print("[UART] Home ACK timeout")
        uart.close()
        return
    _print("Robot homed. Press 'g' to start tracking.")

    try:
        triangulator.start_cameras(frame_w, frame_h)
    except Exception as e:
        _print(f"ERROR starting cameras: {e}")
        uart.close()
        return

    # --- Background warmup ---
    warmup_s = max(0.0, args.warmup_s)
    if warmup_s > 0:
        _print("Remove ball. Learning background (SPACE=skip)...")
        t0 = time.time()
        while time.time() - t0 < warmup_s:
            if not triangulator.cap_left.grab():
                continue
            if not triangulator.cap_right.grab():
                continue
            _, fl = triangulator.cap_left.retrieve()
            _, fr = triangulator.cap_right.retrieve()
            if fl is None or fr is None:
                continue
            triangulator.build_background(fl, fr)
            vis = cv2.resize(fl, (640, int(640 * frame_h / frame_w)))
            progress = min((time.time() - t0) / warmup_s, 1.0)
            h = vis.shape[0]
            bw = int(progress * (vis.shape[1] - 40))
            cv2.rectangle(vis, (20, h - 30), (20 + bw, h - 15), (0, 255, 255), -1)
            cv2.putText(vis, f"BG: {progress*100:.0f}%", (20, h - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.imshow("Ball Tracking", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                break
            if key == ord("q"):
                triangulator.stop_cameras()
                uart.close()
                cv2.destroyAllWindows()
                return
        _print("Background ready.")

    # --- HSV debug state ---
    hsv_debug = False
    h_lo, h_hi = HSV_H_LOW, HSV_H_HIGH
    s_lo, s_hi = HSV_S_LOW, HSV_S_HIGH
    v_lo, v_hi = HSV_V_LOW, HSV_V_HIGH

    def create_hsv_trackbars():
        cv2.namedWindow("HSV Tuning", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("HSV Tuning", 400, 300)
        cv2.createTrackbar("H Low", "HSV Tuning", h_lo, 179, lambda x: None)
        cv2.createTrackbar("H High", "HSV Tuning", h_hi, 179, lambda x: None)
        cv2.createTrackbar("S Low", "HSV Tuning", s_lo, 255, lambda x: None)
        cv2.createTrackbar("S High", "HSV Tuning", s_hi, 255, lambda x: None)
        cv2.createTrackbar("V Low", "HSV Tuning", v_lo, 255, lambda x: None)
        cv2.createTrackbar("V High", "HSV Tuning", v_hi, 255, lambda x: None)

    # --- Main loop ---
    _print("--- LIVE ---  Gate OFF. Press 'g' to start tracking.\n")

    gate = False
    last_send_time = 0.0
    fps_time = time.time()
    fps = 0.0
    frame_count = 0
    send_interval_s = SEND_INTERVAL_MS / 1000.0

    try:
        while True:
            for line in uart.poll_status_lines():
                print(f"[UART][RX] {line}")

            result = triangulator.update()
            if result["left_frame"] is None:
                continue

            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30.0 / max(1e-6, time.time() - fps_time)
                fps_time = time.time()

            # Read HSV trackbars if debug on
            if hsv_debug:
                h_lo = cv2.getTrackbarPos("H Low", "HSV Tuning")
                h_hi = cv2.getTrackbarPos("H High", "HSV Tuning")
                s_lo = cv2.getTrackbarPos("S Low", "HSV Tuning")
                s_hi = cv2.getTrackbarPos("S High", "HSV Tuning")
                v_lo = cv2.getTrackbarPos("V Low", "HSV Tuning")
                v_hi = cv2.getTrackbarPos("V High", "HSV Tuning")

            robot_pos = None
            total_score = 0.0
            hsv_mask_l = None

            if gate and result["found_3d"]:
                reproj = result.get("reproj_err", 0)
                det_l = result.get("left_detection")
                det_r = result.get("right_detection")

                if reproj <= REPROJ_THRESHOLD and det_l is not None and det_r is not None:
                    # HARD GATE: both cameras must detect orange ball
                    # This rejects hand/stick/clothing — only the ball is orange
                    both_orange = (det_l.get("is_orange", False) and
                                   det_r.get("is_orange", False))

                    # MOG2 quality score (average of both cameras)
                    mog2_l = compute_mog2_score(det_l, reproj)
                    mog2_r = compute_mog2_score(det_r, reproj)
                    mog2_score = (mog2_l + mog2_r) / 2.0

                    # HSV orange score (average of both cameras)
                    hsv_l, hsv_mask_l = compute_hsv_score(
                        result["left_frame"], det_l, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi)
                    hsv_r, _ = compute_hsv_score(
                        result["right_frame"], det_r, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi)
                    hsv_score = (hsv_l + hsv_r) / 2.0

                    # Combined weighted score
                    total_score = W_MOG2 * mog2_score + W_HSV * hsv_score

                    if both_orange and total_score >= MIN_SCORE:
                        cx, cy, cz = result["position_3d"]
                        rx, ry, rz = cam_to_robot(R, t_vec, scale, cx, cy, cz)
                        ry_send = ry + Y_SAFETY_OFFSET
                        robot_pos = (rx, ry, rz)

                        now = time.perf_counter()
                        if (now - last_send_time) >= send_interval_s:
                            try:
                                uart.send_intercept(
                                    x_mm=rx, y_mm=ry_send, z_mm=FIXED_Z,
                                    vx_mm_s=0.0, vy_mm_s=0.0, vz_mm_s=0.0,
                                    intercept_time_s=MOVE_TIME_S,
                                    time_sent_s=now,
                                    timestamp_s=now,
                                )
                                last_send_time = now
                                print(f"\r  [SEND] X={rx:+7.0f} Y={ry:+7.0f} Z={rz:+7.0f}  "
                                      f"score={total_score:.2f} reproj={reproj:.1f}px", end="")
                            except Exception as e:
                                _print(f"[UART] Send failed: {e}")

            # --- Visualization ---
            left_vis, right_vis = triangulator.draw_results(result)

            dw = 640
            dh = int(dw * frame_h / frame_w)
            left_s = cv2.resize(left_vis, (dw, dh))
            right_s = cv2.resize(right_vis, (dw, dh))

            gate_str = "ON" if gate else "OFF"
            clr = (0, 255, 0) if gate else (0, 0, 255)
            cv2.putText(left_s, f"FPS:{fps:.0f}  Gate:{gate_str}  Score:{total_score:.2f}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 1)

            if robot_pos is not None:
                cv2.putText(left_s,
                    f"Robot(mm): X={robot_pos[0]:+.0f} Y={robot_pos[1]:+.0f} Z={robot_pos[2]:+.0f}",
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.putText(left_s, "g=gate h=home b=bg d=HSV-debug q=quit",
                        (10, dh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (150, 150, 150), 1)

            cv2.imshow("Ball Tracking", cv2.hconcat([left_s, right_s]))

            # HSV debug mask
            if hsv_debug and result["left_frame"] is not None:
                if hsv_mask_l is None:
                    _, hsv_mask_l = compute_hsv_score(
                        result["left_frame"], None, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi)
                mask_vis = cv2.cvtColor(hsv_mask_l, cv2.COLOR_GRAY2BGR)
                mask_vis = cv2.resize(mask_vis, (dw, dh))
                cv2.putText(mask_vis,
                    f"HSV H=[{h_lo}-{h_hi}] S=[{s_lo}-{s_hi}] V=[{v_lo}-{v_hi}]",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                cv2.imshow("HSV Mask", mask_vis)

            # --- Keys ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                _print("\n[QUIT] Sending home...")
                try:
                    uart.send_home()
                except Exception:
                    pass
                break
            elif key == ord("g"):
                gate = not gate
                _print(f"\n[GATE] {'ON — tracking active' if gate else 'OFF'}")
            elif key == ord("h"):
                _print("\n[HOME] Sending home...")
                try:
                    uart.send_home()
                except Exception:
                    pass
            elif key == ord("d"):
                hsv_debug = not hsv_debug
                if hsv_debug:
                    create_hsv_trackbars()
                    _print(f"\n[HSV DEBUG] ON — adjust trackbars")
                else:
                    cv2.destroyWindow("HSV Tuning")
                    cv2.destroyWindow("HSV Mask")
                    _print(f"\n[HSV DEBUG] OFF — H=[{h_lo}-{h_hi}] S=[{s_lo}-{s_hi}] V=[{v_lo}-{v_hi}]")
            elif key == ord("b"):
                _print("\n[BG RESET] Learning...")
                t0 = time.time()
                while time.time() - t0 < 2.0:
                    if not triangulator.cap_left.grab():
                        continue
                    if not triangulator.cap_right.grab():
                        continue
                    _, fl = triangulator.cap_left.retrieve()
                    _, fr = triangulator.cap_right.retrieve()
                    if fl is not None and fr is not None:
                        triangulator.build_background(fl, fr)
                    if cv2.waitKey(1) & 0xFF == ord(" "):
                        break
                _print("Background ready.")

    except KeyboardInterrupt:
        _print("\n[CTRL-C] Sending home...")
        try:
            uart.send_home()
        except Exception:
            pass
    finally:
        triangulator.stop_cameras()
        cv2.destroyAllWindows()
        uart.close()

    _print("\nDone!")


if __name__ == "__main__":
    main()
