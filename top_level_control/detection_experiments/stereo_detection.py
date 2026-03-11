"""
Stereo Ball Detection Experiment

Runs MOG2 + contour-based ball detection on BOTH cameras simultaneously.
Prints actual vs requested camera format/resolution to terminal.

Edit the variables below to test different configs.

Usage:
    python stereo_detection_experiment.py
"""

import cv2
import numpy as np
import time

# ====================================================================
#  CAMERA SETTINGS — EDIT THESE TO TEST
# ====================================================================
LEFT_CAM_ID     = 1
RIGHT_CAM_ID    = 2
FRAME_WIDTH     = 640
FRAME_HEIGHT    = 480
FPS             = 100
# ====================================================================

# === DETECTION CONFIG ===
ROI_LEFT        = None   # None = full frame, or (x, y, w, h)
ROI_RIGHT       = None
BALL_MIN_AREA   = 80
BALL_MAX_AREA   = 2000
MIN_CIRCULARITY = 0.35
SEARCH_RADIUS   = 150

# === Scoring weights ===
W_CIRCULARITY = 0.4
W_PROXIMITY   = 0.3
W_COLOR       = 0.3
W_NO_HISTORY  = 0.15


def open_camera_with_info(device_id, width, height, name):
    """Open camera, force MJPG, print actual vs requested settings."""
    cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Readback from driver
    rb_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rb_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rb_fps = cap.get(cv2.CAP_PROP_FPS)
    rb_fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    rb_fourcc_str = "".join([chr((rb_fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

    # Actual frame from camera
    ret, test_frame = cap.read()
    if ret:
        actual_h, actual_w = test_frame.shape[:2]
    else:
        actual_w, actual_h = -1, -1

    print(f"\n{'='*60}")
    print(f"  {name} (device {device_id})")
    print(f"{'='*60}")
    print(f"  {'':20s} {'REQUESTED':>12s}  {'READBACK':>12s}  {'ACTUAL FRAME':>12s}")
    print(f"  {'Width':20s} {width:>12d}  {rb_w:>12d}  {actual_w:>12d}")
    print(f"  {'Height':20s} {height:>12d}  {rb_h:>12d}  {actual_h:>12d}")
    print(f"  {'FPS':20s} {FPS:>12d}  {rb_fps:>12.0f}  {'(measure)':>12s}")
    print(f"  {'FOURCC':20s} {'MJPG':>12s}  {rb_fourcc_str:>12s}  {'---':>12s}")

    res_match = (actual_w == width and actual_h == height)
    if res_match:
        print(f"  STATUS: OK")
    else:
        print(f"  STATUS: MISMATCH - actual {actual_w}x{actual_h} != requested {width}x{height}")
    print(f"{'='*60}")

    return cap, res_match


def get_roi(frame, roi):
    """Crop frame to ROI, or return full frame if ROI is None."""
    if roi is None:
        return frame, 0, 0
    x, y, w, h = roi
    return frame[y:y+h, x:x+w], x, y


def detect_ball(roi_frame, bg_sub, kernel_open, kernel_close, last_pos):
    """Run MOG2 + contour detection. Returns (best_candidate, all_candidates, rejected, fg_mask)."""
    fg_mask = bg_sub.apply(roi_frame, learningRate=0.002)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_candidate = None
    best_score     = -1
    all_candidates = []
    rejected       = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        if area < BALL_MIN_AREA:
            rejected.append({'center': (cx, cy), 'area': area, 'reason': 'SMALL'})
            continue
        if area > BALL_MAX_AREA:
            rejected.append({'center': (cx, cy), 'area': area, 'reason': 'BIG'})
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < MIN_CIRCULARITY:
            rejected.append({'center': (cx, cy), 'area': area, 'circularity': circularity, 'reason': 'SHAPE'})
            continue

        score = circularity * W_CIRCULARITY

        if last_pos is not None:
            dist = np.hypot(cx - last_pos[0], cy - last_pos[1])
            if dist < SEARCH_RADIUS:
                score += (1.0 - dist / SEARCH_RADIUS) * W_PROXIMITY
        else:
            score += W_NO_HISTORY

        mask_temp = np.zeros(roi_frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask_temp, [cnt], -1, 255, -1)
        mean_bgr = cv2.mean(roi_frame, mask=mask_temp)[:3]
        pixel = np.uint8([[list(mean_bgr)]])
        hsv_val = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
        hue, sat, val = int(hsv_val[0]), int(hsv_val[1]), int(hsv_val[2])

        is_orange = (5 <= hue <= 25 and sat > 80 and val > 80)
        if is_orange:
            score += W_COLOR

        candidate = {
            'center': (cx, cy),
            'area': area,
            'circularity': circularity,
            'score': score,
            'contour': cnt,
            'is_orange': is_orange
        }
        all_candidates.append(candidate)

        if score > best_score:
            best_score     = score
            best_candidate = candidate

    return best_candidate, all_candidates, rejected, fg_mask


def draw_detections(vis, best, candidates, rejected, last_pos, status_text=""):
    """Draw detection overlays on a frame."""
    for rej in rejected:
        rx, ry = int(rej['center'][0]), int(rej['center'][1])
        cv2.drawMarker(vis, (rx, ry), (0, 0, 200), cv2.MARKER_TILTED_CROSS, 8, 1)

    for cand in candidates:
        ccx, ccy = int(cand['center'][0]), int(cand['center'][1])
        r = max(6, int(np.sqrt(cand['area'] / np.pi)))
        cv2.circle(vis, (ccx, ccy), r, (0, 255, 255), 1)
        label = f"A:{cand['area']:.0f} C:{cand['circularity']:.2f} S:{cand['score']:.2f}"
        cv2.putText(vis, label, (ccx+10, ccy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    if best is not None:
        bcx, bcy = int(best['center'][0]), int(best['center'][1])
        r = max(10, int(np.sqrt(best['area'] / np.pi)))
        cv2.circle(vis, (bcx, bcy), r + 4, (0, 255, 0), 2)
        ball_label = f"A:{best['area']:.0f} C:{best['circularity']:.2f} S:{best['score']:.2f}"
        cv2.putText(vis, ball_label, (bcx+15, bcy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if last_pos is not None:
        cv2.circle(vis, (int(last_pos[0]), int(last_pos[1])), SEARCH_RADIUS, (255, 150, 0), 1)

    if status_text:
        cv2.putText(vis, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    return vis


# =============================================================================
#  MAIN
# =============================================================================
if __name__ == '__main__':

    print("\n" + "#"*60)
    print("  STEREO BALL DETECTION EXPERIMENT")
    print("#"*60)
    print(f"\n  Requested: {FRAME_WIDTH}x{FRAME_HEIGHT} MJPG @ {FPS}fps")
    print(f"  Left cam:  device {LEFT_CAM_ID}")
    print(f"  Right cam: device {RIGHT_CAM_ID}")

    cap_left, left_ok = open_camera_with_info(LEFT_CAM_ID, FRAME_WIDTH, FRAME_HEIGHT, "LEFT CAMERA")
    cap_right, right_ok = open_camera_with_info(RIGHT_CAM_ID, FRAME_WIDTH, FRAME_HEIGHT, "RIGHT CAMERA")

    if not left_ok or not right_ok:
        print("\nWARNING: Resolution mismatch detected! Continue? (y/n)")
        if input().strip().lower() != 'y':
            cap_left.release()
            cap_right.release()
            quit()

    bg_left  = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=False)
    bg_right = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=False)
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # Build background
    print("\nRemove ball from view, press SPACE...")
    while True:
        # grab()/retrieve() ensures both frames from same trigger pulse
        if not cap_left.grab() or not cap_right.grab():
            continue
        ret_l, frame_l = cap_left.retrieve()
        ret_r, frame_r = cap_right.retrieve()
        if not ret_l or not ret_r:
            continue
        preview = np.hstack([
            cv2.resize(frame_l, (480, 300)),
            cv2.resize(frame_r, (480, 300))
        ])
        cv2.putText(preview, "LEFT", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(preview, "RIGHT", (490, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(preview, "Remove ball, press SPACE", (300, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow('Setup', preview)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    cv2.destroyWindow('Setup')

    print("Learning background (3 seconds)...")
    t_start = time.time()
    while time.time() - t_start < 3.0:
        if not cap_left.grab() or not cap_right.grab():
            continue
        ret_l, frame_l = cap_left.retrieve()
        ret_r, frame_r = cap_right.retrieve()
        if not ret_l or not ret_r:
            continue
        roi_l, _, _ = get_roi(frame_l, ROI_LEFT)
        roi_r, _, _ = get_roi(frame_r, ROI_RIGHT)
        bg_left.apply(roi_l, learningRate=0.05)
        bg_right.apply(roi_r, learningRate=0.05)
    print("Ready! Toss the ball.\n")

    last_pos_left  = None
    last_pos_right = None
    fps_counter = 0
    fps_timer   = time.perf_counter()
    actual_fps  = 0

    while True:
        if not cap_left.grab() or not cap_right.grab():
            continue
        ret_l, frame_l = cap_left.retrieve()
        ret_r, frame_r = cap_right.retrieve()
        if not ret_l or not ret_r:
            continue

        roi_l, ox_l, oy_l = get_roi(frame_l, ROI_LEFT)
        roi_r, ox_r, oy_r = get_roi(frame_r, ROI_RIGHT)

        best_l, cands_l, rej_l, mask_l = detect_ball(roi_l, bg_left, kernel_open, kernel_close, last_pos_left)
        best_r, cands_r, rej_r, mask_r = detect_ball(roi_r, bg_right, kernel_open, kernel_close, last_pos_right)

        if best_l is not None:
            last_pos_left = best_l['center']
        if best_r is not None:
            last_pos_right = best_r['center']

        fps_counter += 1
        if fps_counter % 50 == 0:
            actual_fps = 50.0 / (time.perf_counter() - fps_timer)
            fps_timer  = time.perf_counter()

        vis_l = roi_l.copy()
        vis_r = roi_r.copy()

        status_l = f"LEFT | FPS:{actual_fps:.0f} | {'DETECTED' if best_l else 'SEARCHING'} | P:{len(cands_l)} R:{len(rej_l)}"
        status_r = f"RIGHT | {'DETECTED' if best_r else 'SEARCHING'} | P:{len(cands_r)} R:{len(rej_r)}"

        vis_l = draw_detections(vis_l, best_l, cands_l, rej_l, last_pos_left, status_l)
        vis_r = draw_detections(vis_r, best_r, cands_r, rej_r, last_pos_right, status_r)

        # Disparity when both detect
        if best_l is not None and best_r is not None:
            lx = best_l['center'][0]
            rx = best_r['center'][0]
            disparity = lx - rx
            cv2.putText(vis_l, f"Disparity: {disparity:.1f}px", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        # Resize for display
        disp_w, disp_h = 580, 380
        vis_l_small = cv2.resize(vis_l, (disp_w, disp_h))
        vis_r_small = cv2.resize(vis_r, (disp_w, disp_h))

        mask_l_color = cv2.cvtColor(mask_l, cv2.COLOR_GRAY2BGR)
        mask_r_color = cv2.cvtColor(mask_r, cv2.COLOR_GRAY2BGR)
        mask_l_small = cv2.resize(mask_l_color, (disp_w, disp_h))
        mask_r_small = cv2.resize(mask_r_color, (disp_w, disp_h))
        cv2.putText(mask_l_small, "LEFT MASK", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(mask_r_small, "RIGHT MASK", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        top_row    = np.hstack([vis_l_small, vis_r_small])
        bottom_row = np.hstack([mask_l_small, mask_r_small])
        combined   = np.vstack([top_row, bottom_row])

        cv2.imshow('Stereo Ball Detection', combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            bg_left  = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=False)
            bg_right = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=False)
            last_pos_left  = None
            last_pos_right = None
            print("Reset! Remove ball for 2 seconds...")
            t_start = time.time()
            while time.time() - t_start < 2.0:
                if not cap_left.grab() or not cap_right.grab():
                    continue
                ret_l, fl = cap_left.retrieve()
                ret_r, fr = cap_right.retrieve()
                if not ret_l or not ret_r:
                    continue
                rl, _, _ = get_roi(fl, ROI_LEFT)
                rr, _, _ = get_roi(fr, ROI_RIGHT)
                bg_left.apply(rl, learningRate=0.05)
                bg_right.apply(rr, learningRate=0.05)
            print("Ready!")

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()
    print("\nDone.")