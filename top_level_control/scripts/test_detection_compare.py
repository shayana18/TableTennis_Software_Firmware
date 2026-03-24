"""
Ball Detection Comparison — 3 Modes Side by Side
==================================================
Compare different detection approaches to find which tracks the ball
most robustly while rejecting false positives (hand, stick, etc.)

MODES (press 1/2/3 to switch):
  [1] MOG2 + Orange + Tight Filters (current ball_tracking approach)
      - Background subtraction finds moving objects
      - Requires orange color in HSV
      - Tight circularity (>0.5) and area (100-1200) filters

  [2] HSV-Only (no background subtraction)
      - Detects orange blobs directly in HSV color space
      - Works regardless of motion — ball can be stationary
      - Sensitive to lighting and other orange objects

  [3] MOG2 + HSV Combined (intersection)
      - Background subtraction AND HSV color mask
      - Only pixels that are BOTH moving AND orange pass
      - Most restrictive — fewest false positives

CONTROLS:
    1/2/3 - Switch detection mode
    b     - Reset background (modes 1 & 3)
    d     - Toggle debug masks view
    q     - Quit
    SPACE - Skip background warmup
"""

import cv2
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.camera_config import load_camera_settings

# ================================================================
# TUNABLE PARAMETERS
# ================================================================
# MOG2
MOG2_HISTORY       = 300
MOG2_VAR_THRESHOLD = 40
MOG2_LEARN_RATE    = 0.002
MOG2_BG_LEARN_RATE = 0.05

# Contour filters
MIN_AREA        = 100
MAX_AREA        = 1200
MIN_CIRCULARITY = 0.5

# HSV orange range (tune these if lighting changes)
HSV_LOW  = np.array([3, 100, 100])
HSV_HIGH = np.array([25, 255, 255])

# Morphological kernels
KERNEL_OPEN  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
# ================================================================

MODE_NAMES = {1: "MOG2+Orange+Tight", 2: "HSV-Only", 3: "MOG2+HSV Combined"}


def find_best_ball(contours, frame_bgr=None, require_orange=False):
    """Score contours and return best ball candidate."""
    best = None
    best_score = -1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < MIN_CIRCULARITY:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # Check orange if required and frame provided
        is_orange = False
        if frame_bgr is not None:
            mask_temp = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask_temp, [cnt], -1, 255, -1)
            mean_bgr = cv2.mean(frame_bgr, mask=mask_temp)[:3]
            pixel = np.uint8([[list(mean_bgr)]])
            hsv_val = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
            hue, sat, val = int(hsv_val[0]), int(hsv_val[1]), int(hsv_val[2])
            is_orange = (3 <= hue <= 25 and sat > 100 and val > 100)

        if require_orange and not is_orange:
            continue

        # Score: circularity + area nearness to typical ball size (~400px²)
        score = circularity
        if is_orange:
            score += 0.3

        if score > best_score:
            best_score = score
            best = {
                "center": (cx, cy),
                "area": area,
                "circularity": circularity,
                "score": score,
                "contour": cnt,
                "is_orange": is_orange,
            }

    return best


def detect_mode1(frame, bg_sub):
    """Mode 1: MOG2 + Orange + Tight Filters."""
    fg_mask = bg_sub.apply(frame, learningRate=MOG2_LEARN_RATE)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, KERNEL_OPEN)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, KERNEL_CLOSE)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = find_best_ball(contours, frame_bgr=frame, require_orange=True)
    return best, fg_mask


def detect_mode2(frame):
    """Mode 2: HSV-Only (no MOG2)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, KERNEL_OPEN)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, KERNEL_CLOSE)

    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = find_best_ball(contours, frame_bgr=frame, require_orange=False)
    return best, color_mask


def detect_mode3(frame, bg_sub):
    """Mode 3: MOG2 + HSV Combined (intersection)."""
    # MOG2 motion mask
    fg_mask = bg_sub.apply(frame, learningRate=MOG2_LEARN_RATE)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, KERNEL_OPEN)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, KERNEL_CLOSE)

    # HSV color mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, KERNEL_OPEN)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, KERNEL_CLOSE)

    # Intersection: must be BOTH moving AND orange
    combined_mask = cv2.bitwise_and(fg_mask, color_mask)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, KERNEL_CLOSE)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = find_best_ball(contours, frame_bgr=frame, require_orange=False)
    return best, combined_mask, fg_mask, color_mask


def draw_detection(vis, best, label, color=(0, 255, 0)):
    """Draw detection circle and info on frame."""
    if best is not None:
        cx, cy = int(best["center"][0]), int(best["center"][1])
        r = max(8, int(np.sqrt(best["area"] / np.pi)))
        cv2.circle(vis, (cx, cy), r, color, 2)
        cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)
        info = f"A:{best['area']:.0f} C:{best['circularity']:.2f}"
        if best["is_orange"]:
            info += " [O]"
        cv2.putText(vis, info, (cx + r + 5, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.putText(vis, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        cv2.putText(vis, f"{label} — no ball", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return vis


def main():
    cam = load_camera_settings()
    frame_w, frame_h = cam["frame_width"], cam["frame_height"]

    cap = cv2.VideoCapture(cam["camera0"], cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)
    cap.set(cv2.CAP_PROP_FPS, 100)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    # MOG2 background subtractors (separate for mode 1 and mode 3)
    bg_sub1 = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY, varThreshold=MOG2_VAR_THRESHOLD, detectShadows=False)
    bg_sub3 = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY, varThreshold=MOG2_VAR_THRESHOLD, detectShadows=False)

    print("\n" + "=" * 60)
    print("  BALL DETECTION COMPARISON — 3 Modes")
    print("=" * 60)
    print("  1 = MOG2 + Orange + Tight Filters")
    print("  2 = HSV-Only (no background subtraction)")
    print("  3 = MOG2 + HSV Combined (intersection)")
    print("  d = toggle debug masks | b = reset bg | q = quit")
    print("=" * 60)

    # Background warmup
    print("\nRemove ball. Learning background (SPACE=skip)...")
    t0 = time.time()
    while time.time() - t0 < 2.5:
        ret, frame = cap.read()
        if not ret:
            continue
        bg_sub1.apply(frame, learningRate=MOG2_BG_LEARN_RATE)
        bg_sub3.apply(frame, learningRate=MOG2_BG_LEARN_RATE)
        vis = cv2.resize(frame, (640, int(640 * frame_h / frame_w)))
        prog = min((time.time() - t0) / 2.5, 1.0)
        cv2.putText(vis, f"BG: {prog*100:.0f}%", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Detection Compare", vis)
        if cv2.waitKey(1) & 0xFF == ord(" "):
            break
    print("Background ready.\n")

    mode = 1
    show_debug = False
    fps_time = time.time()
    fps = 0.0
    fc = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        fc += 1
        if fc % 30 == 0:
            fps = 30.0 / max(1e-6, time.time() - fps_time)
            fps_time = time.time()

        vis = frame.copy()
        debug_mask = None
        debug_extra1 = None
        debug_extra2 = None

        if mode == 1:
            best, mask = detect_mode1(frame, bg_sub1)
            # Still feed mode 3 bg_sub to keep it current
            bg_sub3.apply(frame, learningRate=MOG2_LEARN_RATE)
            draw_detection(vis, best, f"[1] MOG2+Orange+Tight  FPS:{fps:.0f}")
            debug_mask = mask

        elif mode == 2:
            best, mask = detect_mode2(frame)
            # Feed both bg_subs to keep them current
            bg_sub1.apply(frame, learningRate=MOG2_LEARN_RATE)
            bg_sub3.apply(frame, learningRate=MOG2_LEARN_RATE)
            draw_detection(vis, best, f"[2] HSV-Only  FPS:{fps:.0f}")
            debug_mask = mask

        elif mode == 3:
            best, combined, fg, color = detect_mode3(frame, bg_sub3)
            # Still feed mode 1 bg_sub
            bg_sub1.apply(frame, learningRate=MOG2_LEARN_RATE)
            draw_detection(vis, best, f"[3] MOG2+HSV Combined  FPS:{fps:.0f}")
            debug_mask = combined
            debug_extra1 = fg
            debug_extra2 = color

        # Display
        dw = 640
        dh = int(dw * frame_h / frame_w)
        vis_s = cv2.resize(vis, (dw, dh))

        if show_debug and debug_mask is not None:
            mask_color = cv2.cvtColor(debug_mask, cv2.COLOR_GRAY2BGR)
            mask_s = cv2.resize(mask_color, (dw, dh))
            cv2.putText(mask_s, f"Mask [{MODE_NAMES[mode]}]", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if mode == 3 and debug_extra1 is not None and debug_extra2 is not None:
                # Show MOG2 mask, HSV mask, and combined
                fg_s = cv2.resize(cv2.cvtColor(debug_extra1, cv2.COLOR_GRAY2BGR), (dw // 2, dh // 2))
                hsv_s = cv2.resize(cv2.cvtColor(debug_extra2, cv2.COLOR_GRAY2BGR), (dw // 2, dh // 2))
                cv2.putText(fg_s, "MOG2", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.putText(hsv_s, "HSV", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                extra_row = cv2.hconcat([fg_s, hsv_s])
                mask_col = cv2.vconcat([mask_s, extra_row])
                display = cv2.hconcat([vis_s, cv2.resize(mask_col, (dw, dh + dh // 2))])
            else:
                display = cv2.hconcat([vis_s, mask_s])
        else:
            display = vis_s

        cv2.imshow("Detection Compare", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("1"):
            mode = 1
            print(f"\n[MODE 1] MOG2 + Orange + Tight Filters")
        elif key == ord("2"):
            mode = 2
            print(f"\n[MODE 2] HSV-Only")
        elif key == ord("3"):
            mode = 3
            print(f"\n[MODE 3] MOG2 + HSV Combined")
        elif key == ord("d"):
            show_debug = not show_debug
            print(f"\n[DEBUG] {'ON' if show_debug else 'OFF'}")
        elif key == ord("b"):
            bg_sub1 = cv2.createBackgroundSubtractorMOG2(
                history=MOG2_HISTORY, varThreshold=MOG2_VAR_THRESHOLD, detectShadows=False)
            bg_sub3 = cv2.createBackgroundSubtractorMOG2(
                history=MOG2_HISTORY, varThreshold=MOG2_VAR_THRESHOLD, detectShadows=False)
            print("\n[BG RESET] Learning...")
            t0 = time.time()
            while time.time() - t0 < 2.0:
                ret, f = cap.read()
                if ret:
                    bg_sub1.apply(f, learningRate=MOG2_BG_LEARN_RATE)
                    bg_sub3.apply(f, learningRate=MOG2_BG_LEARN_RATE)
                if cv2.waitKey(1) & 0xFF == ord(" "):
                    break
            print("Background ready.")

    cap.release()
    cv2.destroyAllWindows()
    print("\nDone!")


if __name__ == "__main__":
    main()
