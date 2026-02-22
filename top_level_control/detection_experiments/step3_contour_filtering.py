import cv2
import numpy as np
import time

# === CONFIG ===
CAM_ID          = 1
ROI             = (10, 28, 1166, 668)
BALL_MIN_AREA   = 150
BALL_MAX_AREA   = 1100
MIN_CIRCULARITY = 0.45
SEARCH_RADIUS   = 150

# === Scoring weights (should roughly sum to 1.0 at max) ===
W_CIRCULARITY = 0.4    # Shape score: perfect circle = 0.4
W_PROXIMITY   = 0.3    # Near last position = up to 0.3
W_COLOR       = 0.3    # Orange ball boost = 0.3
W_NO_HISTORY  = 0.15   # Score when no previous position exists

# === Setup ===
cap = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
cap.set(cv2.CAP_PROP_FPS, 100)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

bg_sub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=False)
kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# Build background
print("Remove ball, press SPACE...")
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    cv2.imshow('Setup', frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break
cv2.destroyWindow('Setup')

print("Learning background...")
t_start = time.time()
while time.time() - t_start < 3.0:
    ret, frame = cap.read()
    if not ret:
        continue
    roi_x, roi_y, roi_w, roi_h = ROI
    roi_frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    bg_sub.apply(roi_frame, learningRate=0.05)
print("Ready! Toss the ball.\n")

# === Tracking state ===
last_pos = None

fps_counter = 0
fps_timer   = time.perf_counter()
actual_fps  = 0

# # TEMP: Stats collection for tuning
# _ac_areas = []
# _ac_circs = []
# _ac_scores = []
# _was_detected = False

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Crop to ROI
    roi_x, roi_y, roi_w, roi_h = ROI
    roi_frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

    # --- Background subtraction ---
    fg_mask = bg_sub.apply(roi_frame, learningRate=0.002)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)

    # --- Contour detection + filtering ---
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_candidate = None
    best_score     = -1
    all_candidates = []
    rejected       = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5:
            continue

        # Centroid
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # --- Filter 1: Size ---
        if area < BALL_MIN_AREA:
            rejected.append({'center': (cx, cy), 'area': area, 'reason': 'SMALL'})
            continue
        if area > BALL_MAX_AREA:
            rejected.append({'center': (cx, cy), 'area': area, 'reason': 'BIG'})
            continue

        # --- Filter 2: Circularity ---
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < MIN_CIRCULARITY:
            rejected.append({'center': (cx, cy), 'area': area, 'circularity': circularity, 'reason': 'SHAPE'})
            continue

        # --- Scoring ---
        score = circularity * W_CIRCULARITY

        # Proximity to last known position
        if last_pos is not None:
            dist = np.hypot(cx - last_pos[0], cy - last_pos[1])
            if dist < SEARCH_RADIUS:
                score += (1.0 - dist / SEARCH_RADIUS) * W_PROXIMITY
        else:
            score += W_NO_HISTORY

        # --- Filter 3: Orange ball color boost ---
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

    # --- Update tracking state ---
    detected = best_candidate is not None
    if detected:
        last_pos = best_candidate['center']
        # a = best_candidate['area']
        # c = best_candidate['circularity']
        # s = best_candidate['score']
        # orange_tag = " [ORANGE]" if best_candidate['is_orange'] else ""
        # print(f"  A: {a:7.1f}   C: {c:.3f}   S: {s:.2f}{orange_tag}")
        # _ac_areas.append(a)
        # _ac_circs.append(c)
        # _ac_scores.append(s)
        # _was_detected = True
    # elif _was_detected:
        # avg_a = np.mean(_ac_areas)
        # avg_c = np.mean(_ac_circs)
        # avg_s = np.mean(_ac_scores)
        # print(f"\n--- LOST BALL ({len(_ac_areas)} frames) ---")
        # print(f"  Area  =>  avg: {avg_a:.0f}   min: {np.min(_ac_areas):.0f}   max: {np.max(_ac_areas):.0f}")
        # print(f"  Circ  =>  avg: {avg_c:.3f}   min: {np.min(_ac_circs):.3f}   max: {np.max(_ac_circs):.3f}")
        # print(f"  Score =>  avg: {avg_s:.2f}   min: {np.min(_ac_scores):.2f}   max: {np.max(_ac_scores):.2f}\n")
        # _ac_areas.clear()
        # _ac_circs.clear()
        # _ac_scores.clear()
        # _was_detected = False

    # --- Visualization ---
    vis = roi_frame.copy()

    # Rejected contours in red
    for rej in rejected:
        rx, ry = int(rej['center'][0]), int(rej['center'][1])
        cv2.drawMarker(vis, (rx, ry), (0, 0, 200), cv2.MARKER_TILTED_CROSS, 8, 1)
        if rej['reason'] == 'SHAPE':
            label = f"A:{rej['area']:.0f} C:{rej['circularity']:.2f} [SHAPE]"
        else:
            label = f"A:{rej['area']:.0f} [{rej['reason']}]"
        cv2.putText(vis, label, (rx+10, ry-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 200), 1)

    # All passing candidates in yellow
    for cand in all_candidates:
        ccx, ccy = int(cand['center'][0]), int(cand['center'][1])
        r = max(6, int(np.sqrt(cand['area'] / np.pi)))
        cv2.circle(vis, (ccx, ccy), r, (0, 255, 255), 1)
        label = f"A:{cand['area']:.0f} C:{cand['circularity']:.2f} S:{cand['score']:.2f}"
        cv2.putText(vis, label, (ccx+10, ccy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Best candidate in green
    if detected:
        bcx, bcy = int(best_candidate['center'][0]), int(best_candidate['center'][1])
        r = max(10, int(np.sqrt(best_candidate['area'] / np.pi)))
        cv2.circle(vis, (bcx, bcy), r + 4, (0, 255, 0), 2)
        ball_label = f"A:{best_candidate['area']:.0f}  C:{best_candidate['circularity']:.2f}  S:{best_candidate['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(ball_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        lx, ly = bcx + 15, bcy - 10
        cv2.rectangle(vis, (lx-2, ly-th-4), (lx+tw+4, ly+6), (0, 0, 0), -1)
        cv2.putText(vis, ball_label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, f"({bcx},{bcy})", (lx, ly+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)

    # Search radius
    if last_pos is not None:
        cv2.circle(vis, (int(last_pos[0]), int(last_pos[1])), SEARCH_RADIUS, (255, 150, 0), 1)

    # FPS
    fps_counter += 1
    if fps_counter % 50 == 0:
        actual_fps = 50.0 / (time.perf_counter() - fps_timer)
        fps_timer  = time.perf_counter()

    # Top info bar
    vis_h, vis_w = vis.shape[:2]
    status = "DETECTED" if detected else "SEARCHING..."
    s_color = (0, 255, 0) if detected else (0, 0, 255)
    cv2.putText(vis, f"FPS: {actual_fps:.0f} | {status} | Pass: {len(all_candidates)} | Reject: {len(rejected)}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, s_color, 2)

    # Bottom tuning panel
    panel_h = 50
    vis[vis_h-panel_h:vis_h, :] = (vis[vis_h-panel_h:vis_h, :] * 0.3).astype(np.uint8)
    py = vis_h - 32
    cv2.putText(vis, f"Area: {BALL_MIN_AREA}-{BALL_MAX_AREA}", (10, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(vis, f"Circ: >{MIN_CIRCULARITY:.2f}", (220, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(vis, f"Search R: {SEARCH_RADIUS}px", (400, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(vis, f"Weights: C={W_CIRCULARITY} P={W_PROXIMITY} Col={W_COLOR}", (600, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    # Legend
    py2 = vis_h - 12
    cv2.circle(vis, (15, py2-3), 5, (0, 255, 0), -1)
    cv2.putText(vis, "Ball", (25, py2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    cv2.circle(vis, (80, py2-3), 5, (0, 255, 255), -1)
    cv2.putText(vis, "Candidate", (90, py2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    cv2.drawMarker(vis, (185, py2-3), (0, 0, 200), cv2.MARKER_TILTED_CROSS, 8, 1)
    cv2.putText(vis, "Rejected", (195, py2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 200), 1)
    cv2.circle(vis, (275, py2-3), 5, (255, 150, 0), -1)
    cv2.putText(vis, "Search", (285, py2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 150, 0), 1)
    cv2.putText(vis, "[q]uit  [r]eset bg", (vis_w-200, py2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    # Side-by-side layout
    disp_w, disp_h = 640, 400
    vis_small  = cv2.resize(vis, (disp_w, disp_h))
    mask_color = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
    mask_disp  = cv2.resize(mask_color, (disp_w, disp_h))
    cv2.putText(mask_disp, "FOREGROUND MASK", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(mask_disp, f"Contours: {len(contours)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    combined = np.hstack([vis_small, mask_disp])
    cv2.imshow('Ball Detection', combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        bg_sub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=False)
        last_pos = None
        print("Reset! Remove ball for 2 seconds...")
        t_start = time.time()
        while time.time() - t_start < 2.0:
            ret, f = cap.read()
            if not ret:
                continue
            roi_f = f[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            bg_sub.apply(roi_f, learningRate=0.05)
        print("Ready!")

cap.release()
cv2.destroyAllWindows()