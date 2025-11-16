import cv2
import os
import time
import threading
from queue import Queue, Empty, Full

# Capture parameters
CAMERA_ID = 1 # Camera ID 
CHESSBOARD_SIZE = (10, 7)  # Number of inner corners per chessboard row and column
OUTPUT_DIRECTORY = 'calibration_images_cam1_new'  # Directory to save calibrations images

FRAME_QUEUE_SIZE = 5  # Buffer a few frames to decouple capture and processing

# Auto-capture settings
AUTO_CAPTURE = True  # Enable auto-capture
AUTO_CAPTURE_INTERVAL = 10  # Seconds between auto-captures

def capture_calibration_images():
    """
    Capture images of a chessboard pattern for camera calibration.
    Run in an infinite loop until user presses 'q' or Escape to quit.
    Press 'c' to capture an image.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    
    # Open camera
    cap = cv2.VideoCapture(CAMERA_ID)

    # Set width and height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_RES[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_RES[1])
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) 

    
    if not cap.isOpened():
        print(f"Error: Could not open camera {CAMERA_ID}")
        return
    
    # Get camera resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {width}x{height}")
    
    # Shared state for threaded pipeline
    frame_queue = Queue(maxsize=FRAME_QUEUE_SIZE)
    stop_event = threading.Event()

    def frame_capture():
        """Continuously capture frames and push the freshest ones into the queue."""
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to capture frame")
                time.sleep(0.01)
                continue
            try:
                frame_queue.put_nowait(frame)
            except Full:
                # Drop the oldest frame to keep only the most recent data
                try:
                    frame_queue.get_nowait()
                except Empty:
                    pass
                try:
                    frame_queue.put_nowait(frame)
                except Full:
                    # If the queue is still full, skip this frame
                    continue

    capture_thread = threading.Thread(target=frame_capture, daemon=True)
    capture_thread.start()

    # Counter for captured images
    img_counter = 0
    latest_raw_frame = None
    last_capture_time = time.time()

    print("Press 'c' to capture an image manually")
    print("Press 'q' or Escape to quit")
    if AUTO_CAPTURE:
        print(f"Auto-capture enabled: capturing every {AUTO_CAPTURE_INTERVAL} seconds when chessboard detected")
    print(f"Images will be saved to {OUTPUT_DIRECTORY}")

    try:
        while True:
            try:
                frame = frame_queue.get(timeout=1.0)
                latest_raw_frame = frame
            except Empty:
                if stop_event.is_set():
                    break
                continue

            display_frame = latest_raw_frame.copy()
            gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)

            # Try the robust SB detector first
            ret_chess = False
            corners = None
            try:
                ret_chess, corners = cv2.findChessboardCornersSB(
                    gray, CHESSBOARD_SIZE,
                    flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
                )
            except AttributeError:
                # Fallback for older OpenCV: use the classic detector with good flags
                ret_chess, corners = cv2.findChessboardCorners(
                    gray, CHESSBOARD_SIZE,
                    flags=(cv2.CALIB_CB_ADAPTIVE_THRESH |
                        cv2.CALIB_CB_NORMALIZE_IMAGE |
                        cv2.CALIB_CB_FAST_CHECK)
                )
                if ret_chess:
                    # Refine corner locations
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Draw corners if found
            if ret_chess:
                cv2.drawChessboardCorners(display_frame, CHESSBOARD_SIZE, corners, ret_chess)
                cv2.putText(display_frame, "Chessboard detected!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "NOT detected", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Auto-capture countdown and capture
            if AUTO_CAPTURE:
                time_since_last = time.time() - last_capture_time
                countdown = int(AUTO_CAPTURE_INTERVAL - time_since_last)
                if countdown > 0:
                    cv2.putText(display_frame, f"Next capture in: {countdown}s", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                # Auto-capture if interval passed and chessboard detected
                if time_since_last >= AUTO_CAPTURE_INTERVAL and ret_chess:
                    img_name = os.path.join(OUTPUT_DIRECTORY, f"calibration_{img_counter:02d}.jpg")
                    cv2.imwrite(img_name, latest_raw_frame)
                    print(f"Auto-captured {img_name}")
                    img_counter += 1
                    last_capture_time = time.time()

            # Display the frame
            cv2.imshow('Camera Calibration', display_frame)

            # Wait for key press
            key = cv2.waitKey(1) & 0xFF

            # 'q' or Escape to quit
            if key == ord('q') or key == 27:  # 27 is the ASCII code for Escape
                print("Exiting...")
                break

            # 'c' to capture manually
            elif key == ord('c') and latest_raw_frame is not None:
                # Save the image
                img_name = os.path.join(OUTPUT_DIRECTORY, f"calibration_{img_counter:02d}.jpg")
                cv2.imwrite(img_name, latest_raw_frame)
                print(f"Manually captured {img_name}")
                img_counter += 1
                last_capture_time = time.time()  # Reset auto-capture timer
    finally:
        stop_event.set()
        capture_thread.join(timeout=2)
        # Release camera and close windows
        cap.release()
        cv2.destroyAllWindows()
    
    print(f"Captured {img_counter} images for calibration")

if __name__ == "__main__":
    capture_calibration_images()