import argparse
import platform
import queue
import sys
import threading
import time

import cv2

# Camera Constants
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
REQUESTED_FPS = 100
AUTO_EXPOSURE = 0.25
PROP_EXPOSURE = -6 

def _capture_frames(index, cap, frame_queue, stop_event):
    """ worker function to capture frames from a given camera index"""
    while not stop_event.is_set():
        ok, frame = cap.read()
        packet = {"data": frame, "timestamp": time.perf_counter()}
        while not stop_event.is_set():
            try:
                frame_queue.put_nowait(packet)
                break
            except queue.Full:
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass


def _drain_queue(frame_queue):
    packet = None
    while True:
        try:
            packet = frame_queue.get_nowait()
        except queue.Empty:
            break
    return packet

def open_camera(index=0, backend=None):
    """Open the camera at `index` using an optional backend name."""
    if backend is None:
        cap = cv2.VideoCapture(index)
    else:
        backend_attr = f"CAP_{backend.strip().upper()}"
        if not hasattr(cv2, backend_attr):
            raise ValueError(f"Unknown backend '{backend}'")
        backend_id = getattr(cv2, backend_attr)
        cap = cv2.VideoCapture(index, backend_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {index} (backend={backend or 'auto'})")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, REQUESTED_FPS)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, AUTO_EXPOSURE)   # manual
    cap.set(cv2.CAP_PROP_EXPOSURE, PROP_EXPOSURE)
    return cap


def show_live_feed(indices=None, backend=None, window_name="Live Feed"):
    """
    Display one window per camera index in `indices`.
    
    Press 'q' while any window is focused to close all feeds.
    """
    if hasattr(cv2, "setNumThreads"):
        cv2.setNumThreads(1)
    if indices is None:
        indices = [0]
    elif isinstance(indices, int):
        indices = [indices]
    else:
        indices = list(indices)

    stop_event = threading.Event()
    capture_info = {}
    for idx in indices:
        cap = open_camera(index=idx, backend=backend)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(3):
            cap.read()
        backend_name = getattr(cap, "getBackendName", lambda: backend or "auto")()
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        win_name = f"{window_name} [{idx}]"
        print(
            f"Opened camera {idx} using backend '{backend_name}' "
            f"at {int(actual_width)}x{int(actual_height)}"
        )
        frame_queue = queue.Queue(maxsize=2)
        thread = threading.Thread(
            target=_capture_frames,
            name=f"camera-capture-{idx}",
            args=(idx, cap, frame_queue, stop_event),
            daemon=True,
        )
        thread.start()
        capture_info[idx] = {
            "cap": cap,
            "queue": frame_queue,
            "thread": thread,
            "window": win_name,
            "last_time": None,
            "ema_fps": 0.0,
            "last_report": None,
        }

    try:
        while True:
            any_frame = False
            for idx, info in capture_info.items():
                cap = info["cap"]
                win_name = info["window"]
                frame = _drain_queue(info["queue"])
                if frame is None:
                    continue
                now = frame["timestamp"]
                last = info["last_time"]
                if last is not None:
                    dt = now - last
                    if dt > 0:
                        inst_fps = 1.0 / dt
                        ema = info["ema_fps"]
                        info["ema_fps"] = inst_fps if ema == 0.0 else (0.9 * ema + 0.1 * inst_fps)
                info["last_time"] = now

                if info["ema_fps"]:
                    last_report = info["last_report"]
                    if last_report is None or now - last_report >= 1.0:
                        print(f"Camera {idx}: {info['ema_fps']:.1f} FPS")
                        info["last_report"] = now

                any_frame = True
                cv2.imshow(win_name, frame["data"])

            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
                break
            if not any_frame:
                print("Warning: no frames available from any camera.")
    finally:
        stop_event.set()
        for info in capture_info.values():
            info["thread"].join(timeout=1.0)
        for info in capture_info.values():
            info["cap"].release()
        cv2.destroyAllWindows()


def suggest_backend():
    system = platform.system().lower()
    if system == "darwin":
        return "avfoundation"
    if system == "windows":
        return "dshow"
    if system == "linux":
        return "v4l2"
    return None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Open a live camera feed with OpenCV.")
    parser.add_argument(
        "-i", "--index", "--indices", dest="indices", nargs="+", type=int, default=[0],
        help="Camera indices to open (default: 0). Provide multiple values for multiple feeds."
    )
    parser.add_argument(
        "-b", "--backend", type=str, default=None,
        help="OpenCV capture backend name, e.g. avfoundation, dshow, v4l2."
    )
    parser.add_argument(
        "--suggest-backend", action="store_true",
        help="Print the likely backend for this OS and exit."
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.suggest_backend:
        backend_hint = suggest_backend()
        if backend_hint:
            print(f"Suggested backend for this OS: {backend_hint}")
        else:
            print("Could not determine a backend suggestion for this OS.")
        return 0

    show_live_feed(indices=args.indices, backend=args.backend)
    return 0


if __name__ == "__main__":
    sys.exit(main())
