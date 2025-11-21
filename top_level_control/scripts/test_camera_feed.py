"""
Test camera feeds from index 1 and 2.
Press 'q' to quit.
"""

import cv2
import numpy as np

def main():
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(2)

    if not cap1.isOpened():
        print("ERROR: Camera 1 not found")
        return
    if not cap2.isOpened():
        print("ERROR: Camera 2 not found")
        return

    print("Cameras opened. Press 'q' to quit.")

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            continue

        # Resize to same height if different
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        if h1 != h2:
            target_h = max(h1, h2)
            frame1 = cv2.resize(frame1, (int(w1 * target_h / h1), target_h))
            frame2 = cv2.resize(frame2, (int(w2 * target_h / h2), target_h))

        # Add labels
        cv2.putText(frame1, "CAM 1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame2, "CAM 2", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Combine side by side
        combined = np.hstack([frame1, frame2])

        # Resize if too wide
        h, w = combined.shape[:2]
        if w > 1920:
            scale = 1920 / w
            combined = cv2.resize(combined, None, fx=scale, fy=scale)

        cv2.imshow("Camera Feed Test", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
