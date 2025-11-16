import numpy as np
import cv2
import pickle

# Live camera measurement tool
# Press 'c' to capture, click 2 points, get distance

def main():
    # Get camera choice
    print("Which camera?")
    print("1. Camera 1")
    print("2. Camera 2")
    cam_choice = input("Enter 1 or 2: ").strip()

    if cam_choice == '1':
        calibration_file = 'output_cam1/calibration_data.pkl'
        camera_index = 0
    else:
        calibration_file = 'output_cam2_new/calibration_data.pkl'
        camera_index = 1

    # Get distance from camera to object
    distance_str = input("Distance from camera to object (cm): ").strip()
    try:
        camera_distance_cm = float(distance_str)
    except:
        print("Invalid distance. Using 100 cm.")
        camera_distance_cm = 100.0

    # Load calibration data
    with open(calibration_file, 'rb') as f:
        calibration_data = pickle.load(f)

    camera_matrix = calibration_data['camera_matrix']
    dist_coeffs = calibration_data['distortion_coefficients']
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    focal_length_avg = (fx + fy) / 2

    print(f"\nCamera ready. Distance: {camera_distance_cm} cm")
    print("Press 'c' to capture image")
    print("Press 'q' to quit\n")

    # Open camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_index}")
        return

    captured_frame = None
    points = []

    def mouse_callback(event, x, y, flags, param):
        """Handle mouse clicks on captured image"""
        if event == cv2.EVENT_LBUTTONDOWN and captured_frame is not None:
            points.append((x, y))

            # Draw point
            display_frame = captured_frame.copy()
            for i, pt in enumerate(points):
                cv2.circle(display_frame, pt, 7, (0, 255, 0), -1)
                cv2.putText(display_frame, f'P{i+1}', (pt[0] + 15, pt[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # If 2 points, calculate distance
            if len(points) == 2:
                cv2.line(display_frame, points[0], points[1], (0, 255, 255), 3)

                # Calculate distance
                point_a = np.array(points[0])
                point_b = np.array(points[1])
                pixel_distance = np.linalg.norm(point_b - point_a)
                real_distance_cm = (pixel_distance * camera_distance_cm) / focal_length_avg

                # Display result
                mid_x = (points[0][0] + points[1][0]) // 2
                mid_y = (points[0][1] + points[1][1]) // 2
                cv2.putText(display_frame, f'{real_distance_cm:.2f} cm',
                           (mid_x, mid_y - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                print(f"Distance: {real_distance_cm:.2f} cm")

            cv2.imshow('Measurement', display_frame)

    cv2.namedWindow('Measurement')
    cv2.setMouseCallback('Measurement', mouse_callback)

    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame")
            break

        # Undistort the frame
        frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)

        # Show live feed or captured image
        if captured_frame is None:
            # Show live feed
            display = frame_undistorted.copy()
            cv2.putText(display, "Press 'c' to capture", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Measurement', display)

        # Key handling
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c') and captured_frame is None:
            # Capture frame
            captured_frame = frame_undistorted.copy()
            points = []
            print("\nImage captured! Click 2 points to measure.")
            print("Press 'r' to reset, 'n' for new capture")
            cv2.imshow('Measurement', captured_frame)

        elif key == ord('r'):
            # Reset points
            points = []
            if captured_frame is not None:
                cv2.imshow('Measurement', captured_frame)
                print("Points reset. Click 2 new points.")

        elif key == ord('n'):
            # New capture
            captured_frame = None
            points = []
            print("\nReady for new capture. Press 'c'")

        elif key == ord('q'):
            # Quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
