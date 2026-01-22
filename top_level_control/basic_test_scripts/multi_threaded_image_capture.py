import threading
import cv2
from queue import Queue, Empty, Full
import ball_tracker as bdc
from pypylon import pylon

# Camera Constants
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
REQUESTED_FPS = 100
AUTO_EXPOSURE = 0.25
PROP_EXPOSURE = -6 


processing_img_queue = Queue(maxsize=10)
frame_queue = Queue(maxsize=1)
stop = threading.Event()

def capture_frames(left_cam_id, right_cam_id, processing_img_queue):
    """
    Function for capturing frames from a camera. To be run in a separate thread.

    :param left_cam_id: Index of the left camera to capture from
    :param right_cam_id: Index of the right camera to capture from
    :param processing_img_queue: Queue to store captured frames
    """
    left_cap = cv2.VideoCapture(left_cam_id)
    right_cap = cv2.VideoCapture(right_cam_id)
    left_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    right_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    left_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) 
    left_cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_WIDTH)
    left_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_HEIGHT)
    left_cap.set(cv2.CAP_PROP_FPS, REQUESTED_FPS)
    left_cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, AUTO_EXPOSURE)   # manual
    left_cap.set(cv2.CAP_PROP_EXPOSURE, PROP_EXPOSURE)

    right_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) 
    right_cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_WIDTH)
    right_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_HEIGHT)
    right_cap.set(cv2.CAP_PROP_FPS, REQUESTED_FPS)
    right_cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, AUTO_EXPOSURE)   # manual
    right_cap.set(cv2.CAP_PROP_EXPOSURE, PROP_EXPOSURE)

    if not left_cap.isOpened() or not right_cap.isOpened():
        print("Error: Could not open camera.")
        stop.set()
        return
    while not stop.is_set():
        left_ret, left_frame = left_cap.read()
        right_ret, right_frame = right_cap.read()
        if not left_ret or not right_ret:
            print("Failed to capture frame")
            stop.set()
            break
        try:
            #seperate queue for processing so we do not starve processsing thread
            processing_img_queue.put_nowait((left_frame, right_frame))
        except Full:
            pass
    left_cap.release()
    right_cap.release()

def process_frames(processing_img_queue, left_tracker, right_tracker):
    """
    Function for image processing. 
    To be ran in a separate thread than capture and to process frames captured from a camera
    
    :param processing_img_queue: Description
    """
    # history is how many past frames to use for background model
    # varThreshold is the threshold on the squared Mahalanobis distance to decide

    while not stop.is_set():
        try:
            # get image from captured image queue
            left_frame, right_frame = processing_img_queue.get(timeout=0.1)
        except Empty:
            continue
        left_detector = left_tracker.detect((left_frame))
        right_detector = right_tracker.detect((right_frame))
        frame_mask_set = (
            left_detector["full_frame"],
            left_detector["motion_frame"],
            right_detector["motion_frame"],
            right_detector["full_frame"]
        )
        # put processed image into background subtracted image queue
        try:
            frame_queue.put_nowait(frame_mask_set)
        except Full:
            try:
                frame_queue.get_nowait()
            except Empty:
                pass
            # replacing old frame with new one
            frame_queue.put_nowait(frame_mask_set)

def start_cameras(self, width=1920, height=1200):
    """Open and configure both Basler cameras."""
    
    # Get all connected Basler cameras
    tlFactory = pylon.TlFactory.GetInstance()
    devices = tlFactory.EnumerateDevices()
    
    if len(devices) < 2:
        raise RuntimeError(f"Found {len(devices)} Basler cameras, need at least 2")
    
    print(f"\nFound {len(devices)} Basler camera(s):")
    for i, dev in enumerate(devices):
        serial = dev.GetSerialNumber()
        model = dev.GetModelName()
        print(f"  [{i}] {model} (Serial: {serial})")
    
    # Find cameras by serial number or use first two
    cam_left_dev = None
    cam_right_dev = None
    
    if self.serial_left and self.serial_right:
        # Match by serial number
        for dev in devices:
            serial = dev.GetSerialNumber()
            if serial == self.serial_left:
                cam_left_dev = dev
            elif serial == self.serial_right:
                cam_right_dev = dev
        
        if cam_left_dev is None:
            raise RuntimeError(f"Left camera with serial {self.serial_left} not found")
        if cam_right_dev is None:
            raise RuntimeError(f"Right camera with serial {self.serial_right} not found")
    else:
        # Use first two cameras found
        print("\n[WARNING] No serial numbers specified, using first two cameras found")
        print("          Set serial numbers in config/stereo_config.yaml for consistent assignment")
        cam_left_dev = devices[0]
        cam_right_dev = devices[1]
    
    # Create and open cameras
    self.cam_left = pylon.InstantCamera(tlFactory.CreateDevice(cam_left_dev))
    self.cam_right = pylon.InstantCamera(tlFactory.CreateDevice(cam_right_dev))
    
    self.cam_left.Open()
    self.cam_right.Open()

if __name__ == "__main__":

    left_tracker = bdc.BallTracker()
    right_tracker = bdc.BallTracker()
    capture_thread = threading.Thread(target=capture_frames, args=(40042702, 40042704, processing_img_queue))
    process_thread = threading.Thread(target=process_frames, args=(processing_img_queue, left_tracker, right_tracker))
    capture_thread.start()
    process_thread.start()
    print("Controls: q=quit, m=motion gate, c=color gate, s=shape gate")

    while not stop.is_set():
        try:
            left_full_frame, left_motion_frame, right_motion_frame, right_full_frame = frame_queue.get(timeout=0.3)
        except Empty:
            continue
        cv2.imshow("Left Full Detection", left_full_frame)
        # cv2.imshow("Left Motion Detection", left_motion_frame)
        # cv2.imshow("Right Motion Detection", right_motion_frame)
        cv2.imshow("Right Full Detection", right_full_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            stop.set()
            break
        elif key == ord("m"):
            left_tracker.toggle_gate("motion")
            right_tracker.toggle_gate("motion")
            left_gates = left_tracker.get_gates()
            right_gates = right_tracker.get_gates()
            print(f"Motion gate {'ENABLED' if (left_gates['motion'] and right_gates['motion']) else 'DISABLED'}")
        elif key == ord("c"):
            left_tracker.toggle_gate("color")
            right_tracker.toggle_gate("color")
            left_gates = left_tracker.get_gates()
            right_gates = right_tracker.get_gates()
            print(f"Color gate {'ENABLED' if (left_gates['color'] and right_gates['color']) else 'DISABLED'}")
        elif key == ord("s"):
            left_tracker.toggle_gate("shape")
            right_tracker.toggle_gate("shape")
            left_gates = left_tracker.get_gates()
            right_gates = right_tracker.get_gates()
            print(f"Shape gate {'ENABLED' if (left_gates['shape'] and right_gates['shape']) else 'DISABLED'}")
    
    capture_thread.join()
    process_thread.join()

    cv2.destroyAllWindows()
