import cv2
import numpy as np
import time
from threading import Thread
from queue import Queue

# Test camera performance under heavy computation
# Compares single-threaded vs multi-threaded approach

def heavy_processing(frame):
    """Simulate heavy computation like HSV + ball detection + physics"""
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Multiple color masks (simulating ball detection)
    mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(mask1, mask2)

    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Simulate physics calculations (matrix operations)
    for _ in range(10):
        dummy = np.linalg.inv(np.random.rand(50, 50) + np.eye(50))

    # Gaussian blur (heavy)
    blurred = cv2.GaussianBlur(frame, (15, 15), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    return frame, len(contours)


def test_single_threaded(camera_index, duration=10):
    """Test without multithreading - everything in one loop"""
    print(f"\n=== SINGLE-THREADED TEST (Camera {camera_index}) ===")
    print("Running for", duration, "seconds...")

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FPS, 100)

    frames_processed = 0
    start_time = time.time()
    fps_list = []

    while time.time() - start_time < duration:
        loop_start = time.time()

        # Capture
        ret, frame = cap.read()
        if not ret:
            continue

        # Heavy processing
        processed_frame, num_contours = heavy_processing(frame)

        # Display
        cv2.putText(processed_frame, f'FPS: {len(fps_list)}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Single-Threaded', processed_frame)

        frames_processed += 1
        loop_time = time.time() - loop_start
        fps_list.append(1.0 / loop_time if loop_time > 0 else 0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    print(f"Frames processed: {frames_processed}")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Min FPS: {min(fps_list):.2f}")
    print(f"Max FPS: {max(fps_list):.2f}")

    return avg_fps


def test_multi_threaded(camera_index, duration=10):
    """Test with multithreading - capture and process in separate threads"""
    print(f"\n=== MULTI-THREADED TEST (Camera {camera_index}) ===")
    print("Running for", duration, "seconds...")

    # Shared queues
    frame_queue = Queue(maxsize=2)  # Keep only latest frames
    result_queue = Queue(maxsize=2)

    running = [True]  # Shared flag

    def capture_thread():
        """Continuously capture frames"""
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FPS, 100)

        while running[0]:
            ret, frame = cap.read()
            if ret:
                # Drop old frame if queue is full
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except:
                        pass
                frame_queue.put(frame)

        cap.release()

    def process_thread():
        """Process frames from queue"""
        while running[0]:
            if not frame_queue.empty():
                frame = frame_queue.get()
                processed_frame, num_contours = heavy_processing(frame)

                # Drop old result if queue is full
                if result_queue.full():
                    try:
                        result_queue.get_nowait()
                    except:
                        pass
                result_queue.put(processed_frame)

    # Start threads
    cap_thread = Thread(target=capture_thread)
    proc_thread = Thread(target=process_thread)
    cap_thread.start()
    proc_thread.start()

    frames_displayed = 0
    start_time = time.time()
    fps_list = []
    last_time = time.time()

    while time.time() - start_time < duration:
        if not result_queue.empty():
            frame = result_queue.get()

            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - last_time) if (current_time - last_time) > 0 else 0
            last_time = current_time
            fps_list.append(fps)

            # Display
            cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Multi-Threaded', frame)

            frames_displayed += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Stop threads
    running[0] = False
    cap_thread.join()
    proc_thread.join()
    cv2.destroyAllWindows()

    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    print(f"Frames displayed: {frames_displayed}")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Min FPS: {min(fps_list):.2f}" if fps_list else "Min FPS: N/A")
    print(f"Max FPS: {max(fps_list):.2f}" if fps_list else "Max FPS: N/A")

    return avg_fps


def main():
    print("Camera Performance Test")
    print("Testing heavy computation (HSV, contours, physics simulation)")
    print("=" * 50)

    # Get camera choice
    cam_choice = input("Which camera? (1 or 2): ").strip()
    camera_index = 0 if cam_choice == '1' else 1

    test_duration = 10  # seconds

    # Test single-threaded
    single_fps = test_single_threaded(camera_index, test_duration)

    print("\nWaiting 2 seconds before next test...")
    time.sleep(2)

    # Test multi-threaded
    multi_fps = test_multi_threaded(camera_index, test_duration)

    # Results
    print("\n" + "=" * 50)
    print("RESULTS:")
    print(f"Single-threaded average FPS: {single_fps:.2f}")
    print(f"Multi-threaded average FPS: {multi_fps:.2f}")
    improvement = ((multi_fps - single_fps) / single_fps * 100) if single_fps > 0 else 0
    print(f"Improvement: {improvement:.1f}%")
    print("=" * 50)

    if multi_fps > 60:
        print("Performance is good for real-time tracking!")
    elif multi_fps > 30:
        print("Performance is acceptable, but may need optimization")
    else:
        print("Performance is low - consider lighter processing or better hardware")


if __name__ == "__main__":
    main()
