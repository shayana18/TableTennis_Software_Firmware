import cv2
import numpy as np
import time

# === CONFIG — edit these ===
CAM_ID = 1          # Test with one camera first
ROI    = None       # Set to (x, y, w, h) from Step 1, or None for full frame
                    # e.g. ROI = (100, 50, 1000, 600)

# === Setup camera ===
cap = cv2.VideoCapture(CAM_ID)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
cap.set(cv2.CAP_PROP_FPS, 100)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# === Background Subtractor ===
# This is what replaces your HSV inRange call
bg_sub = cv2.createBackgroundSubtractorMOG2(
    history=300,          # How many frames to model background (higher = more stable)
    varThreshold=40,      # Sensitivity: LOWER = more sensitive (detects more), HIGHER = fewer false positives
    detectShadows=False   # We don't need shadow detection — saves CPU
)

# Morphological kernels for cleaning up the mask
kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# === Phase 1: Build background model (NO BALL IN VIEW) ===
print("=== PHASE 1: BUILDING BACKGROUND MODEL ===")
print(">>> Remove the ball from the table! Press SPACE when ready <<<")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    cv2.imshow('Setup - Remove Ball', frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

print("Learning background for 3 seconds...")
t_start = time.time()
while time.time() - t_start < 3.0:
    ret, frame = cap.read()
    if not ret:
        continue

    roi_frame = frame
    if ROI:
        x, y, w, h = ROI
        roi_frame = frame[y:y+h, x:x+w]

    # Feed frames to background model with high learning rate
    bg_sub.apply(roi_frame, learningRate=0.05)

print("Background model ready!")
print(">>> Now toss the ball. Press 'q' to quit <<<\n")

# === Phase 2: Detection loop ===
fps_counter = 0
fps_timer = time.perf_counter()
actual_fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Crop to ROI
    roi_frame = frame
    if ROI:
        x, y, w, h = ROI
        roi_frame = frame[y:y+h, x:x+w]

    # ========================================
    # THIS REPLACES YOUR HSV CODE
    # ========================================

    # 1. Background subtraction (replaces cv2.inRange with HSV)
    #    learningRate=0.002 means the model slowly adapts to lighting changes
    #    but won't absorb the fast-moving ball into the background
    fg_mask = bg_sub.apply(roi_frame, learningRate=0.002)

    # 2. Clean up the mask
    #    Open: removes small noise specks (erode then dilate)
    #    Close: fills small holes in the ball blob (dilate then erode)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)

    # ========================================
    # END OF REPLACEMENT
    # ========================================

    # FPS counter
    fps_counter += 1
    if fps_counter % 50 == 0:
        now = time.perf_counter()
        actual_fps = 50.0 / (now - fps_timer)
        fps_timer = now

    # Visualization: show original + mask side by side (resized to fit screen)
    disp_w, disp_h = 640, 400
    cam_small  = cv2.resize(roi_frame, (disp_w, disp_h))
    mask_color = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
    mask_small = cv2.resize(mask_color, (disp_w, disp_h))
    combined   = np.hstack([cam_small, mask_small])

    cv2.putText(combined, f"FPS: {actual_fps:.0f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(combined, "LEFT: Camera | RIGHT: Foreground Mask", (10, combined.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Background Subtraction Test', combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Reset background if it gets messed up
        print("Resetting background model... remove ball!")
        bg_sub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=False)
        t_start = time.time()
        while time.time() - t_start < 2.0:
            ret, f = cap.read()
            roi_f = f[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]] if ROI else f
            bg_sub.apply(roi_f, learningRate=0.05)
        print("Background reset done!")

cap.release()
cv2.destroyAllWindows()
