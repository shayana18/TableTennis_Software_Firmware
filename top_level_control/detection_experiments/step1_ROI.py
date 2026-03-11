import cv2

CAM_ID = 1

# CAP_DSHOW for Windows instead of CAP_V4L2
cap = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
cap.set(cv2.CAP_PROP_FPS, 100)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

points = []

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            points.append((x, y))
            print(f"  Point {len(points)}: ({x}, {y})")
        if len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            rx = min(x1, x2)
            ry = min(y1, y2)
            rw = abs(x2 - x1)
            rh = abs(y2 - y1)
            print(f"\n{'='*50}")
            print(f"  YOUR ROI -- paste this into your code:")
            print(f"  ROI = ({rx}, {ry}, {rw}, {rh})")
            print(f"{'='*50}")
            print(f"\n  Press 'q' to quit, 'r' to redo")

cv2.namedWindow('ROI Selector')
cv2.setMouseCallback('ROI Selector', mouse_callback)

print("=" * 50)
print("  ROI SELECTOR")
print("=" * 50)
print("  Click TOP-LEFT corner of play area")
print("  Then click BOTTOM-RIGHT corner")
print("")
print("  Include the TABLE + AIRSPACE above it")
print("  Exclude the floor, your body, etc.")
print("=" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    vis = frame.copy()
    h, w = vis.shape[:2]
    
    # Grid for reference
    for gx in range(0, w, 100):
        cv2.line(vis, (gx, 0), (gx, h), (50, 50, 50), 1)
        cv2.putText(vis, str(gx), (gx+2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100,100,100), 1)
    for gy in range(0, h, 100):
        cv2.line(vis, (0, gy), (w, gy), (50, 50, 50), 1)
        cv2.putText(vis, str(gy), (2, gy+12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100,100,100), 1)
    
    # Draw clicked points
    for i, pt in enumerate(points):
        cv2.circle(vis, pt, 6, (0, 0, 255), -1)
        cv2.putText(vis, f"P{i+1} ({pt[0]},{pt[1]})", (pt[0]+10, pt[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Draw ROI rectangle + preview if both points selected
    if len(points) == 2:
        cv2.rectangle(vis, points[0], points[1], (0, 255, 0), 2)
        
        x1, y1 = points[0]
        x2, y2 = points[1]
        rx, ry = min(x1,x2), min(y1,y2)
        rw, rh = abs(x2-x1), abs(y2-y1)
        
        if rw > 0 and rh > 0:
            roi_preview = frame[ry:ry+rh, rx:rx+rw]
            preview_h = 200
            scale = preview_h / rh
            preview_w = int(rw * scale)
            if preview_w > 0:
                preview = cv2.resize(roi_preview, (preview_w, preview_h))
                cv2.putText(preview, "ROI Preview", (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                vis[5:5+preview_h, w-preview_w-5:w-5] = preview
    
    instructions = "Click 2 corners | 'r' = redo | 'q' = quit"
    cv2.putText(vis, instructions, (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    cv2.imshow('ROI Selector', vis)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        points = []
        print("\n  Reset -- click two new corners")

cap.release()
cv2.destroyAllWindows()