"""
Test Single Camera - Threshold Tuning

Test ball detection on a single camera with HSV + LAB threshold tuning.
Use this to calibrate detection thresholds for your lighting conditions.

CAMERA: Basler acA1920-150uc USB 3.0
        2.3MP, 150fps @ 1920x1200

REQUIREMENTS:
    pip install pypylon

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


def configure_basler_camera(camera, width=1920, height=1200):
    """Configure Basler camera settings."""
    camera.Open()
    
    # Resolution
    camera.Width.SetValue(width)
    camera.Height.SetValue(height)
    
    # Center ROI
    max_w = camera.WidthMax.GetValue()
    max_h = camera.HeightMax.GetValue()
    camera.OffsetX.SetValue((max_w - width) // 2)
    camera.OffsetY.SetValue((max_h - height) // 2)
    
    # Exposure and gain
    camera.ExposureAuto.SetValue("Off")
    camera.ExposureTime.SetValue(3000)  # 3ms
    camera.GainAuto.SetValue("Off")
    camera.Gain.SetValue(0)
    
    actual_width = camera.Width.GetValue()
    actual_height = camera.Height.GetValue()
    actual_fps = camera.AcquisitionFrameRate.GetValue() if camera.AcquisitionFrameRateEnable.GetValue() else 0
    
    return {
        'actual_width': actual_width,
        'actual_height': actual_height,
        'actual_fps': actual_fps,
        'settings_match': (actual_width == width and actual_height == height)
    }


def create_image_converter():
    """Create BGR image converter."""
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    return converter


def grab_frame(camera, converter):
    """Grab single frame from Basler camera."""
    if not camera.IsGrabbing():
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    
    grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    
    if grab_result.GrabSucceeded():
        image = converter.Convert(grab_result)
        frame = image.GetArray()
        grab_result.Release()
        return frame
    
    grab_result.Release()
    return None


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
        
        self.show_tuner = False
        self.show_debug = False
        
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
        cv2.resizeWindow('Threshold Tuner', 400, 450)
        
        cv2.createTrackbar('H Low', 'Threshold Tuner', self.tracker.hsv_lower[0], 179, lambda x: None)
        cv2.createTrackbar('H High', 'Threshold Tuner', self.tracker.hsv_upper[0], 179, lambda x: None)
        cv2.createTrackbar('S Low', 'Threshold Tuner', self.tracker.hsv_lower[1], 255, lambda x: None)
        cv2.createTrackbar('S High', 'Threshold Tuner', self.tracker.hsv_upper[1], 255, lambda x: None)
        cv2.createTrackbar('V Low', 'Threshold Tuner', self.tracker.hsv_lower[2], 255, lambda x: None)
        cv2.createTrackbar('V High', 'Threshold Tuner', self.tracker.hsv_upper[2], 255, lambda x: None)
        
        cv2.createTrackbar('L Low', 'Threshold Tuner', self.tracker.lab_lower[0], 255, lambda x: None)
        cv2.createTrackbar('L High', 'Threshold Tuner', self.tracker.lab_upper[0], 255, lambda x: None)
        cv2.createTrackbar('A Low', 'Threshold Tuner', self.tracker.lab_lower[1], 255, lambda x: None)
        cv2.createTrackbar('A High', 'Threshold Tuner', self.tracker.lab_upper[1], 255, lambda x: None)
        cv2.createTrackbar('B Low', 'Threshold Tuner', self.tracker.lab_lower[2], 255, lambda x: None)
        cv2.createTrackbar('B High', 'Threshold Tuner', self.tracker.lab_upper[2], 255, lambda x: None)
    
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
        except:
            pass
    
    def start_camera(self):
        """Open camera with Basler configuration."""
        if self.camera:
            self.camera.Close()
        
        try:
            self.camera = get_basler_camera(
                serial=self.camera_serial,
                device_index=self.camera_index
            )
            settings = configure_basler_camera(self.camera, self.frame_width, self.frame_height)
            self.converter = create_image_converter()
            
            serial = self.camera.GetDeviceInfo().GetSerialNumber()
            print(f"\nCamera opened (Serial: {serial}):")
            print(f"  Resolution: {settings['actual_width']}x{settings['actual_height']}")
            
            if not settings['settings_match']:
                print("  WARNING: Resolution mismatch!")
            
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
                frame = grab_frame(self.camera, self.converter)
                if frame is None:
                    continue
                
                self.update_from_tuner()
                
                result = self.tracker.detect(frame, return_debug=self.show_debug)
                
                vis = frame.copy()
                
                if result['found']:
                    center = result['center']
                    radius = int(result['radius'])
                    cv2.circle(vis, center, radius, (0, 255, 0), 2)
                    cv2.circle(vis, center, 3, (0, 0, 255), -1)
                    cv2.putText(vis, f"Found: ({center[0]}, {center[1]})", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(vis, "No ball detected", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.putText(vis, f"Camera Index: {self.camera_index}", (10, vis.shape[0] - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(vis, "q:quit t:tuner d:debug s:save", (10, vis.shape[0] - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                # Resize for display (1920x1200 is large)
                display_width = 960
                display_height = int(display_width * self.frame_height / self.frame_width)
                vis = cv2.resize(vis, (display_width, display_height))
                
                cv2.imshow('Single Camera Test', vis)
                
                if self.show_debug and 'debug' in result and result['debug']:
                    debug = result['debug']
                    h, w = 150, 200
                    
                    masks = []
                    for mask, label in [
                        (debug.get('mask_hsv'), 'HSV'),
                        (debug.get('mask_lab'), 'LAB'),
                        (result['mask'], 'FUSED')
                    ]:
                        if mask is not None:
                            m = cv2.resize(mask, (w, h))
                            m = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
                            cv2.putText(m, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
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
                        cv2.destroyWindow('Threshold Tuner')
                        print("\n[TUNER] Disabled")
                
                elif key == ord('d'):
                    self.show_debug = not self.show_debug
                    if not self.show_debug:
                        cv2.destroyWindow('Debug Masks')
                    print(f"\n[DEBUG] {'Enabled' if self.show_debug else 'Disabled'}")
                
                elif key == ord('s'):
                    self.save_thresholds()
                
                elif key == ord('p'):
                    self.print_thresholds()
                
                elif key == ord('c'):
                    # Cycle camera index
                    cameras = list_basler_cameras()
                    if cameras:
                        self.camera_index = (self.camera_index + 1) % len(cameras)
                        self.camera_serial = None  # Use index instead
                        self.start_camera()
        
        except KeyboardInterrupt:
            pass
        
        finally:
            if self.camera:
                self.camera.Close()
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