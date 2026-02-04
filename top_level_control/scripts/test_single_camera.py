"""
Test Single Camera - Threshold Tuning (with Live FPS + Exposure Overlay)

Forces Basler pixel format to Bayer BG8 (raw Bayer) for maximum throughput / FPS.
Debayering is done by the pylon ImageFormatConverter to BGR8 for OpenCV.

CAMERA: Basler acA1920-150uc USB 3.0
        2.3MP, up to 150fps @ 1920x1200 (mode dependent)

REQUIREMENTS:
    pip install pypylon opencv-python numpy

CONTROLS:
    q - Quit
    t - Toggle threshold tuner (HSV + LAB sliders)
    d - Toggle debug view (show HSV, LAB, fused masks)
    s - Save thresholds to ball_thresholds.json
    p - Print current thresholds
    c - Cycle camera (by device index)
    --list - List all connected Basler cameras
"""

import cv2
import sys
import os
import json
import argparse
import numpy as np
import time
from collections import deque

try:
    from pypylon import pylon
except ImportError:
    print("ERROR: pypylon not installed. Run: pip install pypylon")
    sys.exit(1)

# Allow importing project modules (adjust if your repo layout differs)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracking.ball_tracker import EnhancedBallTracker


# ============================================================================
# BASLER CAMERA FUNCTIONS
# ============================================================================

def list_basler_cameras():
    """List all connected Basler cameras."""
    tlFactory = pylon.TlFactory.GetInstance()
    devices = tlFactory.EnumerateDevices()

    if len(devices) == 0:
        print("\nNo Basler cameras found")
        return []

    print(f"\nFound {len(devices)} Basler camera(s):")
    print("-" * 60)
    cameras = []
    for i, device in enumerate(devices):
        serial = device.GetSerialNumber()
        model = device.GetModelName()
        print(f"  [{i}] {model} - Serial: {serial}")
        cameras.append({'index': i, 'serial': serial, 'model': model})
    print("-" * 60)
    return cameras


def get_basler_camera(serial=None, device_index=0):
    """Get Basler camera by serial or index."""
    tlFactory = pylon.TlFactory.GetInstance()
    devices = tlFactory.EnumerateDevices()

    if len(devices) == 0:
        raise RuntimeError("No Basler cameras found")

    if serial and serial.strip():
        for device in devices:
            if device.GetSerialNumber() == serial:
                return pylon.InstantCamera(tlFactory.CreateDevice(device))
        raise RuntimeError(f"Camera with serial '{serial}' not found")

    if device_index >= len(devices):
        raise RuntimeError(f"Device index {device_index} out of range")

    return pylon.InstantCamera(tlFactory.CreateDevice(devices[device_index]))


def safe_get_float(feature, default=None):
    """Safely read a float/integer feature if it exists/accessible."""
    try:
        return float(feature.GetValue())
    except Exception:
        return default


def safe_get_str(feature, default=None):
    """Safely read a string/enum feature if it exists/accessible."""
    try:
        return str(feature.GetValue())
    except Exception:
        return default


def set_pixel_format_bayer_bg8(camera):
    """
    Force PixelFormat to Bayer BG8 if supported.

    Basler/pylon will expose supported pixel formats via camera.PixelFormat.Symbolics.
    We try a few common names to be robust across firmware/PFNC variants.
    """
    desired = ["BayerBG8", "BayerBG8Packed"]  # packed variant uncommon for 8-bit, but safe to try

    # Stop grabbing before changing format
    if camera.IsGrabbing():
        camera.StopGrabbing()

    # Try direct set
    last_err = None
    for fmt in desired:
        try:
            camera.PixelFormat.SetValue(fmt)
            actual = safe_get_str(camera.PixelFormat, default=None)
            if actual == fmt:
                return True, actual
            # Some cameras may accept but report a slightly different canonical string; accept if contains BG8
            if actual and ("Bayer" in actual and "BG" in actual and "8" in actual):
                return True, actual
        except Exception as e:
            last_err = e

    # Try by enumerating what the camera supports and choosing closest match
    try:
        supported = list(camera.PixelFormat.Symbolics)
        # Prefer exact BayerBG8 if present
        for s in supported:
            if s == "BayerBG8":
                camera.PixelFormat.SetValue(s)
                return True, safe_get_str(camera.PixelFormat, default=s)
        # Otherwise pick any Bayer BG 8-bit option
        for s in supported:
            if "Bayer" in s and ("BG" in s or "BGBG" in s) and s.endswith("8"):
                camera.PixelFormat.SetValue(s)
                return True, safe_get_str(camera.PixelFormat, default=s)
        return False, safe_get_str(camera.PixelFormat, default=None)
    except Exception as e:
        return False, f"Error setting PixelFormat: {last_err or e}"


def configure_basler_camera(camera, width=1920, height=1200, exposure_us=3000.0, gain_db=0.0):
    """
    Configure Basler camera settings.

    Notes:
    - ExposureTime is in microseconds.
    - Width/Height/Offsets may be constrained by camera increments; Basler will clamp if needed.
    - PixelFormat is forced to Bayer BG8 (raw Bayer) for max throughput.
    """
    camera.Open()

    # Ensure we are not grabbing while configuring
    if camera.IsGrabbing():
        camera.StopGrabbing()

    # Force Bayer BG8
    ok, actual_pf = set_pixel_format_bayer_bg8(camera)
    if not ok:
        print(f"WARNING: Could not force Bayer BG8. Current PixelFormat: {actual_pf}")
    else:
        print(f"PixelFormat set to: {actual_pf}")

    # Resolution (ROI)
    try:
        camera.Width.SetValue(width)
        camera.Height.SetValue(height)
    except Exception:
        pass

    # Center ROI (if supported)
    try:
        max_w = camera.WidthMax.GetValue()
        max_h = camera.HeightMax.GetValue()

        off_x = max(0, (max_w - camera.Width.GetValue()) // 2)
        off_y = max(0, (max_h - camera.Height.GetValue()) // 2)

        camera.OffsetX.SetValue(off_x)
        camera.OffsetY.SetValue(off_y)
    except Exception:
        pass

    # Exposure and gain
    try:
        camera.ExposureAuto.SetValue("Off")
    except Exception:
        pass
    try:
        camera.ExposureTime.SetValue(float(exposure_us))
    except Exception:
        pass

    try:
        camera.GainAuto.SetValue("Off")
    except Exception:
        pass
    try:
        camera.Gain.SetValue(float(gain_db))
    except Exception:
        pass

    # Read back actuals
    actual_width = safe_get_float(camera.Width, default=0)
    actual_height = safe_get_float(camera.Height, default=0)
    actual_exp = safe_get_float(camera.ExposureTime, default=None)
    actual_gain = safe_get_float(camera.Gain, default=None)
    actual_pf_readback = safe_get_str(camera.PixelFormat, default=None)

    actual_fps_setting = 0.0
    try:
        if camera.AcquisitionFrameRateEnable.GetValue():
            actual_fps_setting = float(camera.AcquisitionFrameRate.GetValue())
    except Exception:
        pass

    return {
        'actual_width': int(actual_width) if actual_width else 0,
        'actual_height': int(actual_height) if actual_height else 0,
        'actual_exposure_us': actual_exp,
        'actual_gain_db': actual_gain,
        'actual_pixel_format': actual_pf_readback,
        'actual_fps_setting': actual_fps_setting,
        'settings_match': (int(actual_width) == width and int(actual_height) == height)
    }


def create_image_converter():
    """Create BGR image converter (debayer happens here)."""
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    return converter


def grab_frame(camera, converter):
    """
    Grab single frame from Basler camera.

    Returns:
        (frame_bgr, cam_timestamp_ns_or_none)
    """
    if not camera.IsGrabbing():
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grab_result.GrabSucceeded():
        ts = None
        try:
            ts = int(grab_result.TimeStamp)  # often ns, but treat as monotonic ticks
        except Exception:
            pass

        image = converter.Convert(grab_result)
        frame = image.GetArray()
        grab_result.Release()
        return frame, ts

    grab_result.Release()
    return None, None


# ============================================================================
# MAIN TESTER CLASS
# ============================================================================

class SingleCameraTester:
    def __init__(self, serial=None, device_index=0):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.thresholds_path = os.path.join(self.script_dir, '..', 'config', 'ball_thresholds.json')

        self.camera_serial = serial
        self.camera_index = device_index
        self.camera = None
        self.converter = None
        self.tracker = EnhancedBallTracker()

        self.frame_width = 1920
        self.frame_height = 1200

        # Default camera settings (adjust here if you want)
        self.exposure_us = 3000.0  # microseconds
        self.gain_db = 0.0

        self.show_tuner = False
        self.show_debug = False

        # Live FPS trackers
        self._host_times = deque(maxlen=60)
        self._cam_ts = deque(maxlen=60)
        self.live_fps_host = 0.0
        self.live_fps_cam = 0.0

        self.load_thresholds()

    def load_thresholds(self):
        """Load thresholds from JSON file."""
        if os.path.exists(self.thresholds_path):
            try:
                with open(self.thresholds_path, 'r') as f:
                    data = json.load(f)

                if 'hsv_lower' in data:
                    self.tracker.set_hsv_thresholds(data['hsv_lower'], data['hsv_upper'])
                if 'lab_lower' in data:
                    self.tracker.set_lab_thresholds(data['lab_lower'], data['lab_upper'])

                print(f"Loaded thresholds from {self.thresholds_path}")
            except Exception as e:
                print(f"Warning: Could not load thresholds: {e}")

    def save_thresholds(self):
        """Save current thresholds to JSON file."""
        data = {
            'hsv_lower': self.tracker.hsv_lower.tolist(),
            'hsv_upper': self.tracker.hsv_upper.tolist(),
            'lab_lower': self.tracker.lab_lower.tolist(),
            'lab_upper': self.tracker.lab_upper.tolist()
        }

        os.makedirs(os.path.dirname(self.thresholds_path), exist_ok=True)

        with open(self.thresholds_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n[SAVED] Thresholds saved to {self.thresholds_path}")

    def print_thresholds(self):
        """Print current thresholds."""
        print("\n" + "=" * 50)
        print("CURRENT THRESHOLDS")
        print("=" * 50)
        print(f"HSV Lower: {self.tracker.hsv_lower.tolist()}")
        print(f"HSV Upper: {self.tracker.hsv_upper.tolist()}")
        print(f"LAB Lower: {self.tracker.lab_lower.tolist()}")
        print(f"LAB Upper: {self.tracker.lab_upper.tolist()}")
        print("=" * 50)

    def create_tuner(self):
        """Create trackbar window."""
        cv2.namedWindow('Threshold Tuner', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Threshold Tuner', 420, 470)

        cv2.createTrackbar('H Low', 'Threshold Tuner', int(self.tracker.hsv_lower[0]), 179, lambda x: None)
        cv2.createTrackbar('H High', 'Threshold Tuner', int(self.tracker.hsv_upper[0]), 179, lambda x: None)
        cv2.createTrackbar('S Low', 'Threshold Tuner', int(self.tracker.hsv_lower[1]), 255, lambda x: None)
        cv2.createTrackbar('S High', 'Threshold Tuner', int(self.tracker.hsv_upper[1]), 255, lambda x: None)
        cv2.createTrackbar('V Low', 'Threshold Tuner', int(self.tracker.hsv_lower[2]), 255, lambda x: None)
        cv2.createTrackbar('V High', 'Threshold Tuner', int(self.tracker.hsv_upper[2]), 255, lambda x: None)

        cv2.createTrackbar('L Low', 'Threshold Tuner', int(self.tracker.lab_lower[0]), 255, lambda x: None)
        cv2.createTrackbar('L High', 'Threshold Tuner', int(self.tracker.lab_upper[0]), 255, lambda x: None)
        cv2.createTrackbar('A Low', 'Threshold Tuner', int(self.tracker.lab_lower[1]), 255, lambda x: None)
        cv2.createTrackbar('A High', 'Threshold Tuner', int(self.tracker.lab_upper[1]), 255, lambda x: None)
        cv2.createTrackbar('B Low', 'Threshold Tuner', int(self.tracker.lab_lower[2]), 255, lambda x: None)
        cv2.createTrackbar('B High', 'Threshold Tuner', int(self.tracker.lab_upper[2]), 255, lambda x: None)

    def update_from_tuner(self):
        """Read trackbar values and update tracker."""
        if not self.show_tuner:
            return

        try:
            hsv_lower = [
                cv2.getTrackbarPos('H Low', 'Threshold Tuner'),
                cv2.getTrackbarPos('S Low', 'Threshold Tuner'),
                cv2.getTrackbarPos('V Low', 'Threshold Tuner')
            ]
            hsv_upper = [
                cv2.getTrackbarPos('H High', 'Threshold Tuner'),
                cv2.getTrackbarPos('S High', 'Threshold Tuner'),
                cv2.getTrackbarPos('V High', 'Threshold Tuner')
            ]
            lab_lower = [
                cv2.getTrackbarPos('L Low', 'Threshold Tuner'),
                cv2.getTrackbarPos('A Low', 'Threshold Tuner'),
                cv2.getTrackbarPos('B Low', 'Threshold Tuner')
            ]
            lab_upper = [
                cv2.getTrackbarPos('L High', 'Threshold Tuner'),
                cv2.getTrackbarPos('A High', 'Threshold Tuner'),
                cv2.getTrackbarPos('B High', 'Threshold Tuner')
            ]

            self.tracker.set_hsv_thresholds(hsv_lower, hsv_upper)
            self.tracker.set_lab_thresholds(lab_lower, lab_upper)
        except Exception:
            pass

    def update_fps(self, cam_timestamp=None):
        """Update live FPS estimates."""
        now = time.perf_counter()
        self._host_times.append(now)

        if len(self._host_times) >= 2:
            dt = self._host_times[-1] - self._host_times[0]
            if dt > 0:
                self.live_fps_host = (len(self._host_times) - 1) / dt

        if cam_timestamp is not None:
            self._cam_ts.append(cam_timestamp)
            if len(self._cam_ts) >= 2:
                dts = self._cam_ts[-1] - self._cam_ts[0]
                if dts > 0:
                    dt_seconds = dts * 1e-9  # assume ns
                    if dt_seconds > 0:
                        self.live_fps_cam = (len(self._cam_ts) - 1) / dt_seconds

    def start_camera(self):
        """Open camera with Basler configuration."""
        if self.camera:
            try:
                if self.camera.IsGrabbing():
                    self.camera.StopGrabbing()
                self.camera.Close()
            except Exception:
                pass

        try:
            self.camera = get_basler_camera(
                serial=self.camera_serial,
                device_index=self.camera_index
            )
            settings = configure_basler_camera(
                self.camera,
                self.frame_width,
                self.frame_height,
                exposure_us=self.exposure_us,
                gain_db=self.gain_db
            )
            self.converter = create_image_converter()

            info = self.camera.GetDeviceInfo()
            serial = info.GetSerialNumber()
            model = info.GetModelName()

            print(f"\nCamera opened: {model} (Serial: {serial})")
            print(f"  Resolution: {settings['actual_width']}x{settings['actual_height']}")
            if settings['actual_pixel_format'] is not None:
                print(f"  PixelFormat: {settings['actual_pixel_format']}")
            if settings['actual_exposure_us'] is not None:
                print(f"  Exposure: {settings['actual_exposure_us']:.0f} us ({settings['actual_exposure_us']/1000:.3f} ms)")
            if settings['actual_gain_db'] is not None:
                print(f"  Gain: {settings['actual_gain_db']:.2f} dB")

            if not settings['settings_match']:
                print("  WARNING: Resolution mismatch (camera may have clamped to valid increments).")

            return True
        except Exception as e:
            print(f"Failed to open camera: {e}")
            return False

    def run(self):
        """Main loop."""
        print("\n" + "=" * 60)
        print("SINGLE CAMERA TEST - Threshold Tuning (Basler acA1920-150uc)")
        print("=" * 60)
        print(f"\nResolution: {self.frame_width}x{self.frame_height}")
        print("\nCONTROLS:")
        print("  q - Quit")
        print("  t - Toggle threshold tuner")
        print("  d - Toggle debug view")
        print("  s - Save thresholds to JSON")
        print("  p - Print current thresholds")
        print("  c - Cycle camera index")
        print("=" * 60)

        if not self.start_camera():
            return

        try:
            while True:
                frame, cam_ts = grab_frame(self.camera, self.converter)
                if frame is None:
                    continue

                self.update_fps(cam_ts)
                self.update_from_tuner()

                result = self.tracker.detect(frame, return_debug=self.show_debug)

                vis = frame.copy()

                # Detection overlay
                if result.get('found', False):
                    center = result['center']
                    radius = int(result['radius'])
                    cv2.circle(vis, center, radius, (0, 255, 0), 2)
                    cv2.circle(vis, center, 3, (0, 0, 255), -1)
                    cv2.putText(
                        vis, f"Found: ({center[0]}, {center[1]})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )
                else:
                    cv2.putText(
                        vis, "No ball detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                    )

                # Live camera settings
                exp_us = safe_get_float(self.camera.ExposureTime, default=None)
                gain_db = safe_get_float(self.camera.Gain, default=None)
                pix_fmt = safe_get_str(self.camera.PixelFormat, default=None)

                y0 = 60
                if pix_fmt:
                    cv2.putText(
                        vis, f"PixelFormat: {pix_fmt}",
                        (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                    )
                    y0 += 24
                if exp_us is not None:
                    cv2.putText(
                        vis, f"Exposure: {exp_us:.0f} us ({exp_us/1000.0:.3f} ms)",
                        (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                    )
                    y0 += 24
                if gain_db is not None:
                    cv2.putText(
                        vis, f"Gain: {gain_db:.2f} dB",
                        (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                    )
                    y0 += 24

                cv2.putText(
                    vis, f"Live FPS (host): {self.live_fps_host:.1f}",
                    (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )
                y0 += 24

                if self.live_fps_cam > 0:
                    cv2.putText(
                        vis, f"Live FPS (cam ts): {self.live_fps_cam:.1f}",
                        (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                    )
                    y0 += 24

                # Bottom UI
                cv2.putText(
                    vis, f"Camera Index: {self.camera_index}",
                    (10, vis.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
                )
                cv2.putText(
                    vis, "q:quit t:tuner d:debug s:save p:print c:cycle",
                    (10, vis.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1
                )

                # Resize for display
                display_width = 960
                display_height = int(display_width * self.frame_height / self.frame_width)
                vis_disp = cv2.resize(vis, (display_width, display_height))

                cv2.imshow('Single Camera Test', vis_disp)

                # Debug masks
                if self.show_debug and result.get('debug'):
                    debug = result['debug']
                    h, w = 150, 200
                    masks = []
                    for mask, label in [
                        (debug.get('mask_hsv'), 'HSV'),
                        (debug.get('mask_lab'), 'LAB'),
                        (result.get('mask'), 'FUSED')
                    ]:
                        if mask is not None:
                            m = cv2.resize(mask, (w, h))
                            m = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
                            cv2.putText(m, label, (5, 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            masks.append(m)

                    if masks:
                        debug_view = cv2.hconcat(masks)
                        cv2.imshow('Debug Masks', debug_view)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('t'):
                    self.show_tuner = not self.show_tuner
                    if self.show_tuner:
                        self.create_tuner()
                        print("\n[TUNER] Enabled")
                    else:
                        try:
                            cv2.destroyWindow('Threshold Tuner')
                        except Exception:
                            pass
                        print("\n[TUNER] Disabled")
                elif key == ord('d'):
                    self.show_debug = not self.show_debug
                    if not self.show_debug:
                        try:
                            cv2.destroyWindow('Debug Masks')
                        except Exception:
                            pass
                    print(f"\n[DEBUG] {'Enabled' if self.show_debug else 'Disabled'}")
                elif key == ord('s'):
                    self.save_thresholds()
                elif key == ord('p'):
                    self.print_thresholds()
                elif key == ord('c'):
                    cameras = list_basler_cameras()
                    if cameras:
                        self.camera_index = (self.camera_index + 1) % len(cameras)
                        self.camera_serial = None

                        # Reset FPS trackers
                        self._host_times.clear()
                        self._cam_ts.clear()
                        self.live_fps_host = 0.0
                        self.live_fps_cam = 0.0

                        self.start_camera()

        except KeyboardInterrupt:
            pass
        finally:
            if self.camera:
                try:
                    if self.camera.IsGrabbing():
                        self.camera.StopGrabbing()
                    self.camera.Close()
                except Exception:
                    pass
            cv2.destroyAllWindows()

        print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description='Test single Basler camera')
    parser.add_argument('--serial', type=str, help='Camera serial number')
    parser.add_argument('--index', type=int, default=0, help='Camera device index')
    parser.add_argument('--list', action='store_true', help='List connected cameras')
    args = parser.parse_args()

    if args.list:
        list_basler_cameras()
        return

    tester = SingleCameraTester(serial=args.serial, device_index=args.index)
    tester.run()


if __name__ == '__main__':
    main()
