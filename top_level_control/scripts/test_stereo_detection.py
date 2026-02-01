"""
Test Stereo Detection - Ball Detection on Both Cameras

Test ball tracking on both cameras with INDEPENDENT thresholds.
Each camera can have different HSV and LAB thresholds to handle
different lighting conditions.

CAMERA: Arducam OV9782 Global Shutter USB Camera
        1MP, 100fps @ 1280x800 MJPG

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracking.ball_tracker import EnhancedBallTracker


def configure_camera_for_arducam(cap, width=1280, height=720):
    """
    Configure camera for Arducam OV9782 global shutter cameras.
    Forces MJPG codec and specified resolution, then verifies actual settings.
    
    Args:
        cap: cv2.VideoCapture object
        width: Desired width (default 1280)
        height: Desired height (default 800)
    
    Returns:
        dict with actual accepted values
    """
    # Set FOURCC to MJPG first - critical for getting full framerate on Arducam
    fourcc_mjpg = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc_mjpg)
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Try to set higher framerate (Arducam OV9782 supports 100fps at 1280x800 MJPG)
    cap.set(cv2.CAP_PROP_FPS, 120)
    
    # Disable auto-exposure for consistent detection
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    
    # Read back actual values
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    
    # Decode FOURCC integer to string
    actual_fourcc_str = "".join([chr((actual_fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
    
    settings = {
        'requested_width': width,
        'requested_height': height,
        'requested_fourcc': 'MJPG',
        'actual_width': actual_width,
        'actual_height': actual_height,
        'actual_fps': actual_fps,
        'actual_fourcc': actual_fourcc_str,
        'settings_match': (actual_width == width and actual_height == height)
    }
    
    return settings


def print_camera_settings(camera_name, settings):
    """Pretty print camera settings for verification."""
    print(f"\n  {camera_name}:")
    print(f"    Resolution: {settings['actual_width']}x{settings['actual_height']} "
          f"(requested {settings['requested_width']}x{settings['requested_height']})")
    print(f"    FOURCC: {settings['actual_fourcc']} (requested {settings['requested_fourcc']})")
    print(f"    FPS: {settings['actual_fps']}")
    
    if settings['settings_match']:
        print(f"    Status: ✓ OK")
        return True
    else:
        print(f"    Status: ⚠ WARNING - Settings mismatch!")
        return False


class ArducamStereoCapture:
    """
    Stereo camera capture optimized for Arducam OV9782 cameras.
    Uses MJPG codec for full framerate at 1280x800.
    """
    
    def __init__(self, cam_left_id=0, cam_right_id=1):
        self.cam_left_id = cam_left_id
        self.cam_right_id = cam_right_id
        self.cap_left = None
        self.cap_right = None
    
    def start_cameras(self, width=1280, height=720):
        """Open and configure cameras with MJPG codec."""
        self.cap_left = cv2.VideoCapture(self.cam_left_id)
        self.cap_right = cv2.VideoCapture(self.cam_right_id)
        
        if not self.cap_left.isOpened():
            raise RuntimeError(f"Failed to open left camera (ID: {self.cam_left_id})")
        if not self.cap_right.isOpened():
            raise RuntimeError(f"Failed to open right camera (ID: {self.cam_right_id})")
        
        # Configure both cameras for Arducam OV9782
        print("\nConfiguring cameras (Arducam OV9782 MJPG mode):")
        settings_left = configure_camera_for_arducam(self.cap_left, width, height)
        settings_right = configure_camera_for_arducam(self.cap_right, width, height)
        
        ok_left = print_camera_settings("LEFT Camera", settings_left)
        ok_right = print_camera_settings("RIGHT Camera", settings_right)
        
        if not (ok_left and ok_right):
            print("\n  ⚠ Some camera settings don't match requested values.")
            print("    Detection may still work, but verify image quality.")
        
        print(f"\nCameras started: Left=ID{self.cam_left_id}, Right=ID{self.cam_right_id}")
    
    def read(self):
        """Read frames from both cameras."""
        ret_left, frame_left = self.cap_left.read()
        ret_right, frame_right = self.cap_right.read()
        return ret_left, frame_left, ret_right, frame_right
    
    def stop_cameras(self):
        """Release cameras."""
        if self.cap_left:
            self.cap_left.release()
        if self.cap_right:
            self.cap_right.release()
        print("Cameras stopped")


class StereoDetectorIndependent:
    """Stereo detector with independent thresholds per camera."""
    
    def __init__(self, config):
        """
        Initialize detector.
        
        Args:
            config: Dictionary with camera configuration
        """
        self.config = config
        self.capture = None
        
        # Independent trackers with their own thresholds
        self.tracker_left = EnhancedBallTracker()
        self.tracker_right = EnhancedBallTracker()
    
    def start_cameras(self, width, height):
        """Start cameras."""
        cam_left_id = self.config.get('camera_left', {}).get('id', 0)
        cam_right_id = self.config.get('camera_right', {}).get('id', 1)
        
        self.capture = ArducamStereoCapture(cam_left_id, cam_right_id)
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
        self.active_camera = 'left'  # 'left' or 'right'
        self.tuner_open = False
        
        # Store thresholds for each camera
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
                # Stereo format
                for cam in ['left', 'right']:
                    for key in ['hsv_lower', 'hsv_upper', 'lab_lower', 'lab_upper']:
                        if key in data[cam]:
                            self.thresholds[cam][key] = data[cam][key]
                print(f"Loaded STEREO thresholds from {filepath}")
            else:
                # Single format - apply to both
                for key in ['hsv_lower', 'hsv_upper', 'lab_lower', 'lab_upper']:
                    if key in data:
                        self.thresholds['left'][key] = data[key]
                        self.thresholds['right'][key] = data[key].copy()
                print(f"Loaded single thresholds from {filepath}")
            
            # Apply to trackers
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
        # Left
        self.detector.tracker_left.set_hsv_thresholds(
            self.thresholds['left']['hsv_lower'],
            self.thresholds['left']['hsv_upper']
        )
        self.detector.tracker_left.set_lab_thresholds(
            self.thresholds['left']['lab_lower'],
            self.thresholds['left']['lab_upper']
        )
        
        # Right
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
        
        # Create trackbars
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
    
    # Get masks from both trackers
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


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '..', 'config', 'stereo_config.yaml')
    thresholds_path = os.path.join(script_dir, '..', 'config', 'ball_thresholds_stereo.json')
    thresholds_single_path = os.path.join(script_dir, '..', 'config', 'ball_thresholds.json')
    
    # Default config for Arducam OV9782
    config = {
        'camera_left': {'id': 1},
        'camera_right': {'id': 2},
        'frame_width': 1280,
        'frame_height': 720
    }
    
    # Load config from file if exists
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
            if loaded_config:
                config.update(loaded_config)
    
    FRAME_WIDTH = config.get('frame_width', 1280)
    FRAME_HEIGHT = config.get('frame_height', 720)
    
    print("\n" + "=" * 60)
    print("STEREO DETECTION TEST (Arducam OV9782)")
    print("=" * 60)
    print(f"\nCamera: Arducam OV9782 Global Shutter")
    print(f"  Left ID:  {config.get('camera_left', {}).get('id', 1)}")
    print(f"  Right ID: {config.get('camera_right', {}).get('id', 2)}")
    print(f"\nResolution: {FRAME_WIDTH}x{FRAME_HEIGHT} (MJPG)")
    
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
    print("=" * 60)
    
    # Initialize detector
    detector = StereoDetectorIndependent(config)
    tuner = StereoThresholdTuner(detector)
    lighting = LightingNormalizer()
    
    # Load thresholds (try stereo first, then single)
    if os.path.exists(thresholds_path):
        tuner.load_thresholds(thresholds_path)
    elif os.path.exists(thresholds_single_path):
        tuner.load_thresholds(thresholds_single_path)
    else:
        print("\nUsing default thresholds")
    
    # Start cameras
    try:
        detector.start_cameras(FRAME_WIDTH, FRAME_HEIGHT)
    except Exception as e:
        print(f"\nERROR: {e}")
        return
    
    show_debug = False
    
    try:
        while True:
            # Capture frames
            ret_left, frame_left_raw, ret_right, frame_right_raw = detector.read_frames()
            
            if not ret_left or not ret_right:
                continue
            
            # Apply lighting normalization
            frame_left = lighting.normalize(frame_left_raw)
            frame_right = lighting.normalize(frame_right_raw)
            
            # Update thresholds from tuner
            if tuner.tuner_open:
                tuner.update_from_trackbars()
            
            # Detect ball
            result_left = detector.tracker_left.detect(frame_left)
            result_right = detector.tracker_right.detect(frame_right)
            
            # Draw on frames
            left_vis = frame_left.copy()
            right_vis = frame_right.copy()
            
            # Left detection
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
            
            # Right detection
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
            
            # Status
            both_found = result_left['found'] and result_right['found']
            status = "BOTH DETECTED" if both_found else "Need both cameras"
            color = (0, 255, 0) if both_found else (0, 0, 255)
            cv2.putText(left_vis, status, (10, left_vis.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Show tuner status and active camera
            if tuner.tuner_open:
                active = tuner.active_camera.upper()
                cv2.putText(left_vis, f"Tuning: {active} (1=L, 2=R)", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Lighting mode
            cv2.putText(left_vis, f"Light: {lighting.get_mode_name()}", (10, left_vis.shape[0] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Controls hint
            cv2.putText(right_vis, "q:quit t:tuner d:debug s:save", (10, right_vis.shape[0] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Combine and show (resize for display)
            display_width = 640
            display_height = int(display_width * FRAME_HEIGHT / FRAME_WIDTH)
            left_small = cv2.resize(left_vis, (display_width, display_height))
            right_small = cv2.resize(right_vis, (display_width, display_height))
            combined = cv2.hconcat([left_small, right_small])
            cv2.imshow('Stereo Detection', combined)
            
            # Debug view
            if show_debug:
                debug_view = create_debug_view(
                    frame_left, frame_right,
                    detector.tracker_left, detector.tracker_right
                )
                if debug_view is not None:
                    cv2.imshow('Debug Masks', debug_view)
            
            # Handle keys
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
    
    except KeyboardInterrupt:
        print("\n\nInterrupted")
    
    finally:
        detector.stop_cameras()
        cv2.destroyAllWindows()
    
    print("\nDone!")


if __name__ == '__main__':
    main()