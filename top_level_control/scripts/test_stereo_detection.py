"""
Test Stereo Detection - Ball Detection on Both Cameras

Test ball tracking on both cameras with INDEPENDENT thresholds.
Each camera can have different HSV and LAB thresholds to handle
different lighting conditions.

CAMERA: Basler acA1920-150uc USB 3.0
        2.3MP, 150fps @ 1920x1200

REQUIREMENTS:
    pip install pypylon

CONTROLS:
    q - Quit
    l - Cycle lighting mode
    d - Toggle debug view (HSV + LAB + Fused for both cameras)
    t - Toggle threshold tuner (separate for LEFT and RIGHT)
    1 - Select LEFT camera for tuning
    2 - Select RIGHT camera for tuning
    c - Copy LEFT thresholds to RIGHT (sync cameras)
    p - Print current thresholds
    s - Save thresholds to ball_thresholds_stereo.json
    w - Same as 's' (write/save)

    Camera Controls:
    e - Increase exposure (+500us)
    E - Decrease exposure (-500us)
    g - Increase gain (+1)
    G - Decrease gain (-1)
    a - Auto exposure (once)
    b - Auto white balance (once)

THRESHOLD FILE:
    Saves to config/ball_thresholds_stereo.json with format:
    {
        "left": { "hsv_lower": [...], "hsv_upper": [...], "lab_lower": [...], "lab_upper": [...] },
        "right": { "hsv_lower": [...], "hsv_upper": [...], "lab_lower": [...], "lab_upper": [...] }
    }
"""

import cv2
import sys
import os
import json
import numpy as np
import yaml
from concurrent.futures import ThreadPoolExecutor

try:
    from pypylon import pylon
except ImportError:
    print("ERROR: pypylon not installed. Run: pip install pypylon")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracking.ball_tracker import EnhancedBallTracker


# ============================================================================
# BASLER CAMERA FUNCTIONS
# ============================================================================

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


def configure_basler_camera(camera, width=1920, height=1200, exposure_us=2000):
    """
    Configure Basler camera settings optimized for high-speed ball tracking.

    Basler acA1920-150uc features used:
    - Resolution/ROI: Configurable for frame rate vs. field of view tradeoff
    - Exposure: Short exposure (2000μs default) to minimize motion blur
    - Bandwidth limiting: Prevents USB conflicts when using 2 cameras
    - Light source preset: Optimizes color processing for consistent detection
    - Chunk timestamps: Enables accurate frame timing for velocity calculation
    """
    camera.Open()

    # === RESOLUTION & ROI ===
    camera.Width.SetValue(width)
    camera.Height.SetValue(height)

    max_w = camera.WidthMax.GetValue()
    max_h = camera.HeightMax.GetValue()
    camera.OffsetX.SetValue((max_w - width) // 2)
    camera.OffsetY.SetValue((max_h - height) // 2)

    # === EXPOSURE SETTINGS ===
    # Short exposure reduces motion blur on fast-moving ball
    # 2000μs = 2ms, allows up to 500fps theoretical (limited by sensor to 150fps)
    camera.ExposureAuto.SetValue("Off")
    camera.ExposureTime.SetValue(exposure_us)

    # === GAIN SETTINGS ===
    camera.GainAuto.SetValue("Off")
    camera.Gain.SetValue(0)  # Start with no gain (less noise)

    # === BANDWIDTH LIMITING ===
    # Critical for dual USB cameras - prevents bandwidth conflicts
    # USB 3.0 = 5Gbps = ~400MB/s, split between 2 cameras = 200MB/s each
    try:
        camera.DeviceLinkThroughputLimitMode.SetValue("On")
        camera.DeviceLinkThroughputLimit.SetValue(200000000)  # 200 MB/s per camera
    except Exception:
        pass  # Not all cameras support this

    # === LIGHT SOURCE PRESET ===
    # Optimizes white balance for consistent color detection
    # Use Daylight5000K for typical indoor lighting, Daylight6500K for bright white LEDs
    try:
        camera.BslLightSourcePreset.SetValue("Daylight5000K")
    except Exception:
        try:
            camera.LightSourcePreset.SetValue("Daylight5000K")
        except Exception:
            pass  # Older cameras may not support this

    # === CHUNK DATA (TIMESTAMPS) ===
    # Enable timestamp chunks for accurate frame timing
    # Critical for velocity calculation in trajectory prediction
    try:
        camera.ChunkModeActive.SetValue(True)
        camera.ChunkSelector.SetValue("Timestamp")
        camera.ChunkEnable.SetValue(True)
    except Exception:
        pass  # Chunk mode not available on all cameras

    # === AUTO FUNCTION PROFILE ===
    # When auto exposure is triggered, prioritize short exposure over low gain
    # This reduces motion blur at the cost of slightly more noise
    try:
        camera.AutoFunctionProfile.SetValue("MinimizeExposureTime")
    except Exception:
        try:
            camera.BslAutoFunctionProfile.SetValue("MinimizeExposureTime")
        except Exception:
            pass

    return {
        'actual_width': camera.Width.GetValue(),
        'actual_height': camera.Height.GetValue(),
        'exposure_us': camera.ExposureTime.GetValue(),
        'settings_match': (camera.Width.GetValue() == width and camera.Height.GetValue() == height)
    }


def create_image_converter():
    """Create BGR image converter."""
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    return converter


def grab_frame(camera, converter):
    """
    Grab single frame from Basler camera.

    Returns:
        tuple: (frame, timestamp_ns) or (None, None) if failed
               timestamp_ns is camera hardware timestamp in nanoseconds (if available)
    """
    if not camera.IsGrabbing():
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    grab_result = camera.RetrieveResult(500, pylon.TimeoutHandling_ThrowException)

    if grab_result.GrabSucceeded():
        image = converter.Convert(grab_result)
        frame = image.GetArray()

        # Extract hardware timestamp from chunk data if available
        # This provides more accurate timing than software timestamps
        timestamp_ns = None
        try:
            if grab_result.ChunkTimestamp.IsReadable():
                timestamp_ns = grab_result.ChunkTimestamp.Value
        except Exception:
            pass

        grab_result.Release()
        return frame, timestamp_ns

    grab_result.Release()
    return None, None


# ============================================================================
# BASLER STEREO CAPTURE CLASS
# ============================================================================

class BaslerStereoCapture:
    """Stereo camera capture for Basler cameras."""

    def __init__(self, left_serial=None, right_serial=None, left_index=0, right_index=1):
        self.left_serial = left_serial
        self.right_serial = right_serial
        self.left_index = left_index
        self.right_index = right_index

        self.cam_left = None
        self.cam_right = None
        self.converter_left = None
        self.converter_right = None
        self.executor = ThreadPoolExecutor(max_workers=2)

    def start_cameras(self, width=1920, height=1200):
        """Open and configure cameras."""
        print("\nOpening Basler cameras...")

        self.cam_left = get_basler_camera(serial=self.left_serial, device_index=self.left_index)
        self.cam_right = get_basler_camera(serial=self.right_serial, device_index=self.right_index)

        settings_left = configure_basler_camera(self.cam_left, width, height)
        settings_right = configure_basler_camera(self.cam_right, width, height)

        self.converter_left = create_image_converter()
        self.converter_right = create_image_converter()

        print(f"\n  LEFT Camera:")
        print(f"    Serial: {self.cam_left.GetDeviceInfo().GetSerialNumber()}")
        print(f"    Resolution: {settings_left['actual_width']}x{settings_left['actual_height']}")
        print(f"    Status: {'OK' if settings_left['settings_match'] else 'WARNING - mismatch'}")

        print(f"\n  RIGHT Camera:")
        print(f"    Serial: {self.cam_right.GetDeviceInfo().GetSerialNumber()}")
        print(f"    Resolution: {settings_right['actual_width']}x{settings_right['actual_height']}")
        print(f"    Status: {'OK' if settings_right['settings_match'] else 'WARNING - mismatch'}")

        print("\nCameras started")

    def read(self):
        """
        Read frames from both cameras in parallel.

        Returns:
            tuple: (ret_left, frame_left, ret_right, frame_right, timestamps)
                   timestamps is dict with 'left' and 'right' hardware timestamps (ns)
        """
        future_left = self.executor.submit(grab_frame, self.cam_left, self.converter_left)
        future_right = self.executor.submit(grab_frame, self.cam_right, self.converter_right)

        frame_left, ts_left = None, None
        frame_right, ts_right = None, None

        try:
            frame_left, ts_left = future_left.result(timeout=0.5)
        except Exception:
            pass

        try:
            frame_right, ts_right = future_right.result(timeout=0.5)
        except Exception:
            pass

        ret_left = frame_left is not None
        ret_right = frame_right is not None

        timestamps = {'left': ts_left, 'right': ts_right}

        return ret_left, frame_left, ret_right, frame_right, timestamps

    def stop_cameras(self):
        """Release cameras."""
        self.executor.shutdown(wait=False)
        if self.cam_left:
            self.cam_left.Close()
        if self.cam_right:
            self.cam_right.Close()
        print("Cameras stopped")


# ============================================================================
# HELPER CLASSES (Same as Arducam version)
# ============================================================================

class StereoDetectorIndependent:
    """Stereo detector with independent thresholds per camera."""
    
    def __init__(self, config):
        self.config = config
        self.capture = None
        self.tracker_left = EnhancedBallTracker()
        self.tracker_right = EnhancedBallTracker()
    
    def start_cameras(self, width, height):
        """Start cameras."""
        left_serial = self.config.get('camera_left', {}).get('serial', '')
        right_serial = self.config.get('camera_right', {}).get('serial', '')
        
        self.capture = BaslerStereoCapture(
            left_serial=left_serial if left_serial else None,
            right_serial=right_serial if right_serial else None,
            left_index=0,
            right_index=1
        )
        self.capture.start_cameras(width, height)
    
    def read_frames(self):
        """Read frames from cameras."""
        return self.capture.read()
    
    def stop_cameras(self):
        """Stop cameras."""
        if self.capture:
            self.capture.stop_cameras()


class LightingNormalizer:
    """Lighting normalization techniques."""
    
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.clahe_strong = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        
        self.mode = 0
        self.mode_names = ["OFF", "CLAHE", "CLAHE Strong", "White Balance", "Gamma Auto", "Full Pipeline"]
    
    def cycle_mode(self):
        self.mode = (self.mode + 1) % len(self.mode_names)
        return self.mode_names[self.mode]
    
    def get_mode_name(self):
        return self.mode_names[self.mode]
    
    def normalize(self, frame):
        if frame is None:
            return None
        if self.mode == 0:
            return frame
        elif self.mode == 1:
            return self._apply_clahe(frame)
        elif self.mode == 2:
            return self._apply_clahe(frame, strong=True)
        elif self.mode == 3:
            return self._apply_white_balance(frame)
        elif self.mode == 4:
            return self._apply_gamma(frame)
        elif self.mode == 5:
            result = self._apply_white_balance(frame)
            result = self._apply_clahe(result)
            return self._apply_gamma(result)
        return frame
    
    def _apply_clahe(self, frame, strong=False):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe_strong.apply(l) if strong else self.clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    
    def _apply_white_balance(self, frame):
        result = frame.astype(np.float32)
        for i in range(3):
            avg = np.mean(result[:, :, i])
            if avg > 0:
                result[:, :, i] *= 128.0 / avg
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _apply_gamma(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        if brightness < 80:
            gamma = 0.6
        elif brightness > 180:
            gamma = 1.5
        else:
            return frame
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(frame, table)


class StereoThresholdTuner:
    """Manages threshold tuning for both cameras independently."""
    
    def __init__(self, detector):
        self.detector = detector
        self.active_camera = 'left'
        self.tuner_open = False
        
        self.thresholds = {
            'left': {
                'hsv_lower': [0, 78, 69],
                'hsv_upper': [50, 255, 255],
                'lab_lower': [16, 125, 160],
                'lab_upper': [255, 248, 255]
            },
            'right': {
                'hsv_lower': [0, 78, 69],
                'hsv_upper': [50, 255, 255],
                'lab_lower': [16, 125, 160],
                'lab_upper': [255, 248, 255]
            }
        }
    
    def load_thresholds(self, filepath):
        """Load thresholds from JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if 'left' in data and 'right' in data:
                for cam in ['left', 'right']:
                    for key in ['hsv_lower', 'hsv_upper', 'lab_lower', 'lab_upper']:
                        if key in data[cam]:
                            self.thresholds[cam][key] = data[cam][key]
                print(f"Loaded STEREO thresholds from {filepath}")
            else:
                for key in ['hsv_lower', 'hsv_upper', 'lab_lower', 'lab_upper']:
                    if key in data:
                        self.thresholds['left'][key] = data[key]
                        self.thresholds['right'][key] = data[key].copy()
                print(f"Loaded single thresholds from {filepath}")
            
            self._apply_thresholds()
            
        except Exception as e:
            print(f"Warning: Could not load thresholds: {e}")
    
    def save_thresholds(self, filepath):
        """Save thresholds to JSON file in stereo format."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.thresholds, f, indent=2)
        
        print(f"\n[SAVED] Stereo thresholds saved to {filepath}")
    
    def _apply_thresholds(self):
        """Apply current thresholds to trackers."""
        self.detector.tracker_left.set_hsv_thresholds(
            self.thresholds['left']['hsv_lower'],
            self.thresholds['left']['hsv_upper']
        )
        self.detector.tracker_left.set_lab_thresholds(
            self.thresholds['left']['lab_lower'],
            self.thresholds['left']['lab_upper']
        )
        
        self.detector.tracker_right.set_hsv_thresholds(
            self.thresholds['right']['hsv_lower'],
            self.thresholds['right']['hsv_upper']
        )
        self.detector.tracker_right.set_lab_thresholds(
            self.thresholds['right']['lab_lower'],
            self.thresholds['right']['lab_upper']
        )
    
    def set_active_camera(self, camera):
        """Set which camera is being tuned."""
        if camera in ['left', 'right']:
            self.active_camera = camera
            print(f"\n[TUNER] Now tuning {camera.upper()} camera")
            if self.tuner_open:
                self._update_trackbars_from_thresholds()
    
    def copy_left_to_right(self):
        """Copy left camera thresholds to right."""
        for key in self.thresholds['left']:
            self.thresholds['right'][key] = self.thresholds['left'][key].copy()
        self._apply_thresholds()
        print("\n[COPIED] Left thresholds copied to Right")
    
    def open_tuner(self):
        """Open threshold tuner window."""
        cv2.namedWindow('Threshold Tuner', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Threshold Tuner', 400, 500)
        
        t = self.thresholds[self.active_camera]
        
        cv2.createTrackbar('H Low', 'Threshold Tuner', t['hsv_lower'][0], 179, lambda x: None)
        cv2.createTrackbar('H High', 'Threshold Tuner', t['hsv_upper'][0], 179, lambda x: None)
        cv2.createTrackbar('S Low', 'Threshold Tuner', t['hsv_lower'][1], 255, lambda x: None)
        cv2.createTrackbar('S High', 'Threshold Tuner', t['hsv_upper'][1], 255, lambda x: None)
        cv2.createTrackbar('V Low', 'Threshold Tuner', t['hsv_lower'][2], 255, lambda x: None)
        cv2.createTrackbar('V High', 'Threshold Tuner', t['hsv_upper'][2], 255, lambda x: None)
        
        cv2.createTrackbar('L Low', 'Threshold Tuner', t['lab_lower'][0], 255, lambda x: None)
        cv2.createTrackbar('L High', 'Threshold Tuner', t['lab_upper'][0], 255, lambda x: None)
        cv2.createTrackbar('A Low', 'Threshold Tuner', t['lab_lower'][1], 255, lambda x: None)
        cv2.createTrackbar('A High', 'Threshold Tuner', t['lab_upper'][1], 255, lambda x: None)
        cv2.createTrackbar('B Low', 'Threshold Tuner', t['lab_lower'][2], 255, lambda x: None)
        cv2.createTrackbar('B High', 'Threshold Tuner', t['lab_upper'][2], 255, lambda x: None)
        
        self.tuner_open = True
        print(f"\n[TUNER] Opened - tuning {self.active_camera.upper()} camera")
    
    def close_tuner(self):
        """Close threshold tuner window."""
        cv2.destroyWindow('Threshold Tuner')
        self.tuner_open = False
        print("\n[TUNER] Closed")
    
    def _update_trackbars_from_thresholds(self):
        """Update trackbar positions from current thresholds."""
        if not self.tuner_open:
            return
        
        t = self.thresholds[self.active_camera]
        
        try:
            cv2.setTrackbarPos('H Low', 'Threshold Tuner', t['hsv_lower'][0])
            cv2.setTrackbarPos('H High', 'Threshold Tuner', t['hsv_upper'][0])
            cv2.setTrackbarPos('S Low', 'Threshold Tuner', t['hsv_lower'][1])
            cv2.setTrackbarPos('S High', 'Threshold Tuner', t['hsv_upper'][1])
            cv2.setTrackbarPos('V Low', 'Threshold Tuner', t['hsv_lower'][2])
            cv2.setTrackbarPos('V High', 'Threshold Tuner', t['hsv_upper'][2])
            
            cv2.setTrackbarPos('L Low', 'Threshold Tuner', t['lab_lower'][0])
            cv2.setTrackbarPos('L High', 'Threshold Tuner', t['lab_upper'][0])
            cv2.setTrackbarPos('A Low', 'Threshold Tuner', t['lab_lower'][1])
            cv2.setTrackbarPos('A High', 'Threshold Tuner', t['lab_upper'][1])
            cv2.setTrackbarPos('B Low', 'Threshold Tuner', t['lab_lower'][2])
            cv2.setTrackbarPos('B High', 'Threshold Tuner', t['lab_upper'][2])
        except:
            pass
    
    def update_from_trackbars(self):
        """Read trackbar values and update thresholds."""
        if not self.tuner_open:
            return
        
        try:
            self.thresholds[self.active_camera]['hsv_lower'] = [
                cv2.getTrackbarPos('H Low', 'Threshold Tuner'),
                cv2.getTrackbarPos('S Low', 'Threshold Tuner'),
                cv2.getTrackbarPos('V Low', 'Threshold Tuner')
            ]
            self.thresholds[self.active_camera]['hsv_upper'] = [
                cv2.getTrackbarPos('H High', 'Threshold Tuner'),
                cv2.getTrackbarPos('S High', 'Threshold Tuner'),
                cv2.getTrackbarPos('V High', 'Threshold Tuner')
            ]
            self.thresholds[self.active_camera]['lab_lower'] = [
                cv2.getTrackbarPos('L Low', 'Threshold Tuner'),
                cv2.getTrackbarPos('A Low', 'Threshold Tuner'),
                cv2.getTrackbarPos('B Low', 'Threshold Tuner')
            ]
            self.thresholds[self.active_camera]['lab_upper'] = [
                cv2.getTrackbarPos('L High', 'Threshold Tuner'),
                cv2.getTrackbarPos('A High', 'Threshold Tuner'),
                cv2.getTrackbarPos('B High', 'Threshold Tuner')
            ]
            
            self._apply_thresholds()
        except:
            pass
    
    def print_thresholds(self):
        """Print current thresholds for both cameras."""
        print("\n" + "=" * 60)
        print("CURRENT THRESHOLDS")
        print("=" * 60)
        for cam in ['left', 'right']:
            print(f"\n{cam.upper()} Camera:")
            print(f"  HSV Lower: {self.thresholds[cam]['hsv_lower']}")
            print(f"  HSV Upper: {self.thresholds[cam]['hsv_upper']}")
            print(f"  LAB Lower: {self.thresholds[cam]['lab_lower']}")
            print(f"  LAB Upper: {self.thresholds[cam]['lab_upper']}")
        print("=" * 60)


def create_debug_view(frame_left, frame_right, tracker_left, tracker_right):
    """Create debug visualization showing masks for both cameras."""
    h, w = 120, 160
    
    result_left = tracker_left.detect(frame_left, return_debug=True)
    result_right = tracker_right.detect(frame_right, return_debug=True)
    
    def make_mask_row(result, label_prefix):
        masks = []
        if 'debug' in result and result['debug']:
            debug = result['debug']
            for mask, label in [
                (debug.get('mask_hsv'), f'{label_prefix} HSV'),
                (debug.get('mask_lab'), f'{label_prefix} LAB'),
                (result['mask'], f'{label_prefix} FUSED')
            ]:
                if mask is not None:
                    m = cv2.resize(mask, (w, h))
                    m = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
                    cv2.putText(m, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    masks.append(m)
        
        if masks:
            return cv2.hconcat(masks)
        return None
    
    row_left = make_mask_row(result_left, "L")
    row_right = make_mask_row(result_right, "R")
    
    if row_left is not None and row_right is not None:
        return cv2.vconcat([row_left, row_right])
    elif row_left is not None:
        return row_left
    elif row_right is not None:
        return row_right
    
    return None


# ============================================================================
# MAIN
# ============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '..', 'config', 'stereo_config.yaml')
    thresholds_path = os.path.join(script_dir, '..', 'config', 'ball_thresholds_stereo.json')
    thresholds_single_path = os.path.join(script_dir, '..', 'config', 'ball_thresholds.json')
    
    # Default config for Basler
    config = {
        'camera_left': {'serial': ''},
        'camera_right': {'serial': ''},
        'frame_width': 1920,
        'frame_height': 1200
    }
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
            if loaded_config:
                config.update(loaded_config)
    
    FRAME_WIDTH = config.get('frame_width', 1920)
    FRAME_HEIGHT = config.get('frame_height', 1200)
    
    print("\n" + "=" * 60)
    print("STEREO DETECTION TEST (Basler acA1920-150uc)")
    print("=" * 60)
    print(f"\nCamera: Basler acA1920-150uc USB 3.0")
    print(f"  Left Serial:  {config.get('camera_left', {}).get('serial', '') or '(auto)'}")
    print(f"  Right Serial: {config.get('camera_right', {}).get('serial', '') or '(auto)'}")
    print(f"\nResolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    
    print("\nCONTROLS:")
    print("  q - Quit")
    print("  l - Cycle lighting mode")
    print("  d - Toggle debug view (show masks)")
    print("  t - Toggle threshold tuner")
    print("  1 - Select LEFT camera for tuning")
    print("  2 - Select RIGHT camera for tuning")
    print("  c - Copy LEFT thresholds to RIGHT")
    print("  p - Print current thresholds")
    print("  s/w - Save thresholds to JSON")
    print("\n  Camera Controls:")
    print("  e/E - Increase/Decrease exposure")
    print("  g/G - Increase/Decrease gain")
    print("  a   - Auto exposure (once)")
    print("  b   - Auto white balance (once)")
    print("=" * 60)
    
    detector = StereoDetectorIndependent(config)
    tuner = StereoThresholdTuner(detector)
    lighting = LightingNormalizer()
    
    if os.path.exists(thresholds_path):
        tuner.load_thresholds(thresholds_path)
    elif os.path.exists(thresholds_single_path):
        tuner.load_thresholds(thresholds_single_path)
    else:
        print("\nUsing default thresholds")
    
    try:
        detector.start_cameras(FRAME_WIDTH, FRAME_HEIGHT)
    except Exception as e:
        print(f"\nERROR: {e}")
        return
    
    show_debug = False
    
    try:
        while True:
            ret_left, frame_left_raw, ret_right, frame_right_raw, timestamps = detector.read_frames()

            if not ret_left or not ret_right:
                continue

            frame_left = lighting.normalize(frame_left_raw)
            frame_right = lighting.normalize(frame_right_raw)
            
            if tuner.tuner_open:
                tuner.update_from_trackbars()
            
            result_left = detector.tracker_left.detect(frame_left)
            result_right = detector.tracker_right.detect(frame_right)
            
            left_vis = frame_left.copy()
            right_vis = frame_right.copy()
            
            if result_left['found']:
                center = result_left['center']
                radius = int(result_left['radius'])
                cv2.circle(left_vis, center, radius, (0, 255, 0), 2)
                cv2.circle(left_vis, center, 3, (0, 0, 255), -1)
                cv2.putText(left_vis, f"L: ({center[0]}, {center[1]})", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(left_vis, "L: No ball", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if result_right['found']:
                center = result_right['center']
                radius = int(result_right['radius'])
                cv2.circle(right_vis, center, radius, (0, 255, 0), 2)
                cv2.circle(right_vis, center, 3, (0, 0, 255), -1)
                cv2.putText(right_vis, f"R: ({center[0]}, {center[1]})", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(right_vis, "R: No ball", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            both_found = result_left['found'] and result_right['found']
            status = "BOTH DETECTED" if both_found else "Need both cameras"
            color = (0, 255, 0) if both_found else (0, 0, 255)
            cv2.putText(left_vis, status, (10, left_vis.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if tuner.tuner_open:
                active = tuner.active_camera.upper()
                cv2.putText(left_vis, f"Tuning: {active} (1=L, 2=R)", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            cv2.putText(left_vis, f"Light: {lighting.get_mode_name()}", (10, left_vis.shape[0] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            cv2.putText(right_vis, "q:quit t:tuner d:debug s:save", (10, right_vis.shape[0] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Resize for display (1920x1200 is large)
            display_width = 640
            display_height = int(display_width * FRAME_HEIGHT / FRAME_WIDTH)
            left_small = cv2.resize(left_vis, (display_width, display_height))
            right_small = cv2.resize(right_vis, (display_width, display_height))
            combined = cv2.hconcat([left_small, right_small])
            cv2.imshow('Stereo Detection', combined)
            
            if show_debug:
                debug_view = create_debug_view(
                    frame_left, frame_right,
                    detector.tracker_left, detector.tracker_right
                )
                if debug_view is not None:
                    cv2.imshow('Debug Masks', debug_view)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('l'):
                mode = lighting.cycle_mode()
                print(f"[LIGHTING] {mode}")
            elif key == ord('d'):
                show_debug = not show_debug
                if not show_debug:
                    cv2.destroyWindow('Debug Masks')
                print(f"[DEBUG] {'ON' if show_debug else 'OFF'}")
            elif key == ord('t'):
                if tuner.tuner_open:
                    tuner.close_tuner()
                else:
                    tuner.open_tuner()
            elif key == ord('1'):
                tuner.set_active_camera('left')
            elif key == ord('2'):
                tuner.set_active_camera('right')
            elif key == ord('c'):
                tuner.copy_left_to_right()
            elif key == ord('p'):
                tuner.print_thresholds()
            elif key == ord('s') or key == ord('w'):
                tuner.save_thresholds(thresholds_path)
            elif key == ord('e'):  # Increase exposure
                for cam in [detector.capture.cam_left, detector.capture.cam_right]:
                    current = cam.ExposureTime.GetValue()
                    cam.ExposureTime.SetValue(min(current + 500, 100000))
                print(f"[EXPOSURE] {detector.capture.cam_left.ExposureTime.GetValue():.0f}us")
            elif key == ord('E'):  # Decrease exposure
                for cam in [detector.capture.cam_left, detector.capture.cam_right]:
                    current = cam.ExposureTime.GetValue()
                    cam.ExposureTime.SetValue(max(current - 500, 100))
                print(f"[EXPOSURE] {detector.capture.cam_left.ExposureTime.GetValue():.0f}us")
            elif key == ord('g'):  # Increase gain
                for cam in [detector.capture.cam_left, detector.capture.cam_right]:
                    current = cam.Gain.GetValue()
                    cam.Gain.SetValue(min(current + 1, 20))
                print(f"[GAIN] {detector.capture.cam_left.Gain.GetValue():.1f}")
            elif key == ord('G'):  # Decrease gain
                for cam in [detector.capture.cam_left, detector.capture.cam_right]:
                    current = cam.Gain.GetValue()
                    cam.Gain.SetValue(max(current - 1, 0))
                print(f"[GAIN] {detector.capture.cam_left.Gain.GetValue():.1f}")
            elif key == ord('a'):  # Auto exposure once
                for cam in [detector.capture.cam_left, detector.capture.cam_right]:
                    cam.ExposureAuto.SetValue("Once")
                print("[AUTO EXPOSURE] Running...")
            elif key == ord('b'):  # Auto white balance once
                for cam in [detector.capture.cam_left, detector.capture.cam_right]:
                    cam.BalanceWhiteAuto.SetValue("Once")
                print("[AUTO WHITE BALANCE] Running...")

    except KeyboardInterrupt:
        print("\n\nInterrupted")
    
    finally:
        detector.stop_cameras()
        cv2.destroyAllWindows()
    
    print("\nDone!")


if __name__ == '__main__':
    main()