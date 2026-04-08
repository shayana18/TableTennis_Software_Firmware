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

# Scoring weights
W_CIRCULARITY = 0.4
W_PROXIMITY   = 0.3
W_COLOR       = 0.3
W_NO_HISTORY  = 0.15

# Kalman filter config
KF_PROCESS_NOISE     = 0.1
KF_MEASUREMENT_NOISE = 2.0
KF_MAX_COAST_FRAMES  = 5


# ============================================================
# KALMAN FILTER CLASS
# ============================================================
class BallKalman2D:
    def __init__(self, dt=0.01, process_noise=0.1, measurement_noise=2.0, max_coast=5):
        self.kf = cv2.KalmanFilter(4, 2)

        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]], dtype=np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]], dtype=np.float32)

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.processNoiseCov[2, 2] = process_noise * 5
        self.kf.processNoiseCov[3, 3] = process_noise * 5

        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 100

        self.initialized = False
        self.missed = 0
        self.max_coast = max_coast

    def update(self, detection):
        if detection is not None:
            meas = np.array([[np.float32(detection[0])],
                             [np.float32(detection[1])]])

            if not self.initialized:
                self.kf.statePost = np.array([
                    [meas[0, 0]], [meas[1, 0]], [0], [0]], dtype=np.float32)
                self.initialized = True
                self.missed = 0
                return detection, (0.0, 0.0), False

            self.kf.predict()
            state = self.kf.correct(meas)
            self.missed = 0
            return (float(state[0, 0]), float(state[1, 0])), \
                   (float(state[2, 0]), float(state[3, 0])), False

        else:
            if not self.initialized:
                return None, (0.0, 0.0), True

            self.missed += 1
            if self.missed > self.max_coast:
                self.initialized = False
                return None, (0.0, 0.0), True

            state = self.kf.predict()
            return (float(state[0, 0]), float(state[1, 0])), \
                   (float(state[2, 0]), float(state[3, 0])), True

    def reset(self):
        self.initialized = False
        self.missed = 0


# ============================================================
# SETUP
# ============================================================
cap = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
cap.set(cv2.CAP_PROP_FPS, 100)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

bg_sub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=False)
kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

kf = BallKalman2D(
    dt=1.0 / 100,
    process_noise=KF_PROCESS_NOISE,
    measurement_noise=KF_MEASUREMENT_NOISE,
    max_coast=KF_MAX_COAST_FRAMES
)

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

# Stats
_ac_areas  = []
_ac_circs  = []
_ac_scores = []
_was_detected = False

# Trail for filtered positions only
trail_filtered = []
TRAIL_LENGTH   = 60


while True:
    ret, frame = cap.read()
    if not ret:
        continue

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

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # Filter 1: Size
        if area < BALL_MIN_AREA:
            rejected.append({'center': (cx, cy), 'area': area, 'reason': 'SMALL'})
            continue
        if area > BALL_MAX_AREA:
            rejected.append({'center': (cx, cy), 'area': area, 'reason': 'BIG'})
            continue

        # Filter 2: Circularity
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < MIN_CIRCULARITY:
            rejected.append({'center': (cx, cy), 'area': area, 'circularity': circularity, 'reason': 'SHAPE'})
            continue

        # Scoring
        score = circularity * W_CIRCULARITY

        if last_pos is not None:
            dist = np.hypot(cx - last_pos[0], cy - last_pos[1])
            if dist < SEARCH_RADIUS:
                score += (1.0 - dist / SEARCH_RADIUS) * W_PROXIMITY
        else:
            score += W_NO_HISTORY

        # Filter 3: Orange color boost
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

    # --- Raw detection result ---
    detected = best_candidate is not None
    raw_detection = best_candidate['center'] if detected else None

    # --- Kalman filter update ---
    filtered_pos, velocity, is_coasting = kf.update(raw_detection)

    # Use filtered position for search radius
    if raw_detection is not None:
        last_pos = raw_detection
    if filtered_pos is not None:
        last_pos = filtered_pos
    elif not is_coasting:
        last_pos = None

    # Update trail
    if filtered_pos is not None:
        trail_filtered.append((int(filtered_pos[0]), int(filtered_pos[1])))
    if len(trail_filtered) > TRAIL_LENGTH:
        trail_filtered.pop(0)
    if filtered_pos is None and not is_coasting:
        trail_filtered.clear()

    # --- Stats printing ---
    if detected:
        a = best_candidate['area']
        c = best_candidate['circularity']
        s = best_candidate['score']
        orange_tag = " [ORANGE]" if best_candidate['is_orange'] else ""
        print(f"  A: {a:7.1f}   C: {c:.3f}   S: {s:.2f}   V: ({velocity[0]:+.1f},{velocity[1]:+.1f}){orange_tag}")
        _ac_areas.append(a)
        _ac_circs.append(c)
        _ac_scores.append(s)
        _was_detected = True
    elif is_coasting and filtered_pos is not None:
        print(f"  [COASTING frame {kf.missed}/{KF_MAX_COAST_FRAMES}]  "
              f"predicted: ({filtered_pos[0]:.0f},{filtered_pos[1]:.0f})  "
              f"V: ({velocity[0]:+.1f},{velocity[1]:+.1f})")
    elif _was_detected:
        avg_a = np.mean(_ac_areas)
        avg_c = np.mean(_ac_circs)
        avg_s = np.mean(_ac_scores)
        print(f"\n--- LOST BALL ({len(_ac_areas)} frames) ---")
        print(f"  Area  =>  avg: {avg_a:.0f}   min: {np.min(_ac_areas):.0f}   max: {np.max(_ac_areas):.0f}")
        print(f"  Circ  =>  avg: {avg_c:.3f}   min: {np.min(_ac_circs):.3f}   max: {np.max(_ac_circs):.3f}")
        print(f"  Score =>  avg: {avg_s:.2f}   min: {np.min(_ac_scores):.2f}   max: {np.max(_ac_scores):.2f}\n")
        _ac_areas.clear()
        _ac_circs.clear()
        _ac_scores.clear()
        _was_detected = False

    # --- Visualization ---
    vis = roi_frame.copy()

    # Filtered trail in cyan (fading)
    for i in range(1, len(trail_filtered)):
        alpha = i / len(trail_filtered)
        color = (int(200 * alpha), int(200 * alpha), 0)
        cv2.line(vis, trail_filtered[i-1], trail_filtered[i], color, 2)

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

    # Raw detection — small green dot
    if raw_detection is not None:
        rcx, rcy = int(raw_detection[0]), int(raw_detection[1])
        cv2.circle(vis, (rcx, rcy), 4, (0, 255, 0), -1)

    # Filtered position
    if filtered_pos is not None:
        fx, fy = int(filtered_pos[0]), int(filtered_pos[1])

        if is_coasting:
            cv2.circle(vis, (fx, fy), 16, (0, 165, 255), 2)
            cv2.putText(vis, "COAST", (fx+20, fy-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        else:
            cv2.circle(vis, (fx, fy), 14, (255, 255, 0), 2)

        # Velocity arrow
        vx_draw = int(velocity[0] * 0.4)
        vy_draw = int(velocity[1] * 0.4)
        if abs(vx_draw) > 2 or abs(vy_draw) > 2:
            cv2.arrowedLine(vis, (fx, fy), (fx + vx_draw, fy + vy_draw),
                            (255, 0, 255), 1, tipLength=0.3)

        # Velocity info
        spd = np.hypot(velocity[0], velocity[1])
        ball_label = f"V: ({velocity[0]:+.1f},{velocity[1]:+.1f}) spd:{spd:.1f} px/f"
        cv2.putText(vis, ball_label, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

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
    if is_coasting and filtered_pos is not None:
        status  = f"COASTING ({kf.missed}/{KF_MAX_COAST_FRAMES})"
        s_color = (0, 165, 255)
    elif detected:
        status  = "DETECTED"
        s_color = (0, 255, 0)
    else:
        status  = "SEARCHING..."
        s_color = (0, 0, 255)
    cv2.putText(vis, f"FPS: {actual_fps:.0f} | {status} | Pass: {len(all_candidates)} | Reject: {len(rejected)}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, s_color, 2)

    # Bottom panel
    panel_h = 50
    vis[vis_h-panel_h:vis_h, :] = (vis[vis_h-panel_h:vis_h, :] * 0.3).astype(np.uint8)
    py = vis_h - 32
    cv2.putText(vis, f"Area: {BALL_MIN_AREA}-{BALL_MAX_AREA}", (10, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(vis, f"Circ: >{MIN_CIRCULARITY:.2f}", (220, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(vis, f"KF noise: P={KF_PROCESS_NOISE} M={KF_MEASUREMENT_NOISE}", (400, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(vis, f"Coast: {KF_MAX_COAST_FRAMES}f", (720, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    # Legend
    py2 = vis_h - 12
    cv2.circle(vis, (15, py2-3), 4, (0, 255, 0), -1)
    cv2.putText(vis, "Raw", (25, py2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    cv2.circle(vis, (65, py2-3), 5, (255, 255, 0), -1)
    cv2.putText(vis, "Filtered", (75, py2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
    cv2.circle(vis, (145, py2-3), 5, (0, 165, 255), -1)
    cv2.putText(vis, "Coast", (155, py2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1)
    cv2.line(vis, (200, py2-3), (220, py2-3), (255, 0, 255), 2)
    cv2.putText(vis, "Velocity", (225, py2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)
    cv2.putText(vis, "[q]uit  [r]eset bg", (vis_w-200, py2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    # Side-by-side
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
        kf.reset()
        trail_filtered.clear()
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