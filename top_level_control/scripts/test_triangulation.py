"""
Test Triangulation - Stereo 3D Ball Tracking

Tests stereo calibration by detecting the ball in both cameras
and triangulating to get 3D position.

CAMERA: Basler acA1920-150uc USB 3.0
        2.3MP, 150fps @ 1920x1200

REQUIREMENTS:
    pip install pypylon

FEATURES:
- Loads HSV and LAB thresholds from config/ball_thresholds.json
- Debug mode shows HSV, LAB, and fused masks for both cameras
- Threshold tuner with sliders for HSV and LAB
- Auto-saves thresholds to JSON
- Measurement logging with statistics

CONTROLS:
    q - Quit
    s - Save current 3D measurement
    r - Reset/clear measurements
    t - Toggle threshold tuner (HSV + LAB sliders)
    d - Toggle debug view (show masks)
    p - Print current thresholds
    w - Save thresholds to JSON file

UNITS:
    All measurements are in the SAME UNITS as checkerboard_box_size_scale
    - X: Horizontal (+ right, - left) from camera center
    - Y: Vertical (+ down, - up) from camera center  
    - Z: Depth (distance from cameras)
"""

import cv2
import sys
import os
import json
import yaml
import numpy as np

try:
    from pypylon import pylon
except ImportError:
    print("ERROR: pypylon not installed. Run: pip install pypylon")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracking.stereo_triangulator import StereoTriangulator


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


def configure_basler_camera(camera, width=1920, height=1200):
    """Configure Basler camera settings."""
    camera.Open()
    
    camera.Width.SetValue(width)
    camera.Height.SetValue(height)
    
    max_w = camera.WidthMax.GetValue()
    max_h = camera.HeightMax.GetValue()
    camera.OffsetX.SetValue((max_w - width) // 2)
    camera.OffsetY.SetValue((max_h - height) // 2)
    
    camera.ExposureAuto.SetValue("Off")
    camera.ExposureTime.SetValue(3000)
    camera.GainAuto.SetValue("Off")
    camera.Gain.SetValue(0)
    
    return {
        'actual_width': camera.Width.GetValue(),
        'actual_height': camera.Height.GetValue(),
        'settings_match': (camera.Width.GetValue() == width and camera.Height.GetValue() == height)
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
# TRIANGULATION TESTER CLASS
# ============================================================================

class TriangulationTester:
    """Full-featured triangulation tester with debug and tuning modes."""
    
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.calibration_dir = os.path.join(self.script_dir, '..', 'camera_calibration', 'camera_parameters')
        self.config_path = os.path.join(self.script_dir, '..', 'config', 'stereo_config.yaml')
        self.thresholds_path = os.path.join(self.script_dir, '..', 'config', 'ball_thresholds.json')
        
        self.frame_width = 1920
        self.frame_height = 1200
        
        self.left_serial = None
        self.right_serial = None
        
        self.triangulator = None
        self.cam_left = None
        self.cam_right = None
        self.converter = None
        
        self.measurements = []
        self.show_debug = False
        self.show_tuner = False
        
        self.hsv_lower = [0, 78, 69]
        self.hsv_upper = [50, 255, 255]
        self.lab_lower = [16, 125, 160]
        self.lab_upper = [255, 248, 255]
        
        self.hsv_lower_left = self.hsv_lower.copy()
        self.hsv_upper_left = self.hsv_upper.copy()
        self.lab_lower_left = self.lab_lower.copy()
        self.lab_upper_left = self.lab_upper.copy()
        
        self.hsv_lower_right = self.hsv_lower.copy()
        self.hsv_upper_right = self.hsv_upper.copy()
        self.lab_lower_right = self.lab_lower.copy()
        self.lab_upper_right = self.lab_upper.copy()
        
        self.stereo_thresholds = False
        
    def load_config(self):
        """Load camera serials and resolution from config."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.left_serial = config.get('camera_left', {}).get('serial', '')
            self.right_serial = config.get('camera_right', {}).get('serial', '')
            self.frame_width = config.get('frame_width', 1920)
            self.frame_height = config.get('frame_height', 1200)
            print(f"Loaded config:")
            print(f"  Left Serial: {self.left_serial or '(auto)'}")
            print(f"  Right Serial: {self.right_serial or '(auto)'}")
            print(f"  Resolution: {self.frame_width}x{self.frame_height}")
        else:
            print(f"No config found, using defaults")
    
    def load_thresholds(self):
        """Load HSV and LAB thresholds from JSON."""
        stereo_path = os.path.join(self.script_dir, '..', 'config', 'ball_thresholds_stereo.json')
        single_path = self.thresholds_path
        
        if os.path.exists(stereo_path):
            try:
                with open(stereo_path, 'r') as f:
                    data = json.load(f)
                
                if 'left' in data and 'right' in data:
                    self.hsv_lower_left = data['left'].get('hsv_lower', self.hsv_lower)
                    self.hsv_upper_left = data['left'].get('hsv_upper', self.hsv_upper)
                    self.lab_lower_left = data['left'].get('lab_lower', self.lab_lower)
                    self.lab_upper_left = data['left'].get('lab_upper', self.lab_upper)
                    
                    self.hsv_lower_right = data['right'].get('hsv_lower', self.hsv_lower)
                    self.hsv_upper_right = data['right'].get('hsv_upper', self.hsv_upper)
                    self.lab_lower_right = data['right'].get('lab_lower', self.lab_lower)
                    self.lab_upper_right = data['right'].get('lab_upper', self.lab_upper)
                    
                    self.stereo_thresholds = True
                    print(f"\nLoaded STEREO thresholds from {stereo_path}")
                    return
            except Exception as e:
                print(f"Warning: Could not load stereo thresholds: {e}")
        
        if os.path.exists(single_path):
            try:
                with open(single_path, 'r') as f:
                    data = json.load(f)
                
                self.hsv_lower = data.get('hsv_lower', self.hsv_lower)
                self.hsv_upper = data.get('hsv_upper', self.hsv_upper)
                self.lab_lower = data.get('lab_lower', self.lab_lower)
                self.lab_upper = data.get('lab_upper', self.lab_upper)
                
                self.hsv_lower_left = self.hsv_lower_right = self.hsv_lower
                self.hsv_upper_left = self.hsv_upper_right = self.hsv_upper
                self.lab_lower_left = self.lab_lower_right = self.lab_lower
                self.lab_upper_left = self.lab_upper_right = self.lab_upper
                
                self.stereo_thresholds = False
                print(f"\nLoaded single thresholds from {single_path}")
            except Exception as e:
                print(f"Warning: Could not load thresholds: {e}")
        else:
            self.hsv_lower_left = self.hsv_lower_right = self.hsv_lower
            self.hsv_upper_left = self.hsv_upper_right = self.hsv_upper
            self.lab_lower_left = self.lab_lower_right = self.lab_lower
            self.lab_upper_left = self.lab_upper_right = self.lab_upper
            self.stereo_thresholds = False
            print(f"\nNo thresholds file found, using defaults")
    
    def save_thresholds(self):
        """Save current thresholds to JSON file."""
        data = {
            'hsv_lower': self.hsv_lower,
            'hsv_upper': self.hsv_upper,
            'lab_lower': self.lab_lower,
            'lab_upper': self.lab_upper
        }
        
        os.makedirs(os.path.dirname(self.thresholds_path), exist_ok=True)
        
        with open(self.thresholds_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n[SAVED] Thresholds saved to {self.thresholds_path}")
    
    def apply_thresholds(self):
        """Apply current thresholds to triangulator."""
        if self.triangulator:
            self.triangulator.tracker_left.set_hsv_thresholds(
                self.hsv_lower_left, self.hsv_upper_left)
            self.triangulator.tracker_left.set_lab_thresholds(
                self.lab_lower_left, self.lab_upper_left)
            
            self.triangulator.tracker_right.set_hsv_thresholds(
                self.hsv_lower_right, self.hsv_upper_right)
            self.triangulator.tracker_right.set_lab_thresholds(
                self.lab_lower_right, self.lab_upper_right)
    
    def check_calibration(self):
        """Check if calibration files exist."""
        required_files = [
            'camera0_intrinsics.dat',
            'camera1_intrinsics.dat',
            'camera0_rot_trans.dat',
            'camera1_rot_trans.dat'
        ]
        
        missing = []
        for f in required_files:
            path = os.path.join(self.calibration_dir, f)
            if not os.path.exists(path):
                missing.append(f)
        
        if missing:
            print("\n" + "=" * 60)
            print("ERROR: Missing calibration files!")
            print("=" * 60)
            for f in missing:
                print(f"  - {f}")
            print("\nRun calibration first:")
            print("  cd camera_calibration")
            print("  python calib.py calibration_settings.yaml")
            print("=" * 60)
            return False
        
        return True
    
    def create_tuner_window(self):
        """Create HSV + LAB threshold tuner window."""
        cv2.namedWindow('Threshold Tuner', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Threshold Tuner', 400, 450)
        
        cv2.createTrackbar('H Low', 'Threshold Tuner', self.hsv_lower[0], 179, lambda x: None)
        cv2.createTrackbar('H High', 'Threshold Tuner', self.hsv_upper[0], 179, lambda x: None)
        cv2.createTrackbar('S Low', 'Threshold Tuner', self.hsv_lower[1], 255, lambda x: None)
        cv2.createTrackbar('S High', 'Threshold Tuner', self.hsv_upper[1], 255, lambda x: None)
        cv2.createTrackbar('V Low', 'Threshold Tuner', self.hsv_lower[2], 255, lambda x: None)
        cv2.createTrackbar('V High', 'Threshold Tuner', self.hsv_upper[2], 255, lambda x: None)
        
        cv2.createTrackbar('L Low', 'Threshold Tuner', self.lab_lower[0], 255, lambda x: None)
        cv2.createTrackbar('L High', 'Threshold Tuner', self.lab_upper[0], 255, lambda x: None)
        cv2.createTrackbar('A Low', 'Threshold Tuner', self.lab_lower[1], 255, lambda x: None)
        cv2.createTrackbar('A High', 'Threshold Tuner', self.lab_upper[1], 255, lambda x: None)
        cv2.createTrackbar('B Low', 'Threshold Tuner', self.lab_lower[2], 255, lambda x: None)
        cv2.createTrackbar('B High', 'Threshold Tuner', self.lab_upper[2], 255, lambda x: None)
    
    def update_from_tuner(self):
        """Read values from threshold tuner and apply to trackers."""
        if not self.show_tuner:
            return
        
        try:
            self.hsv_lower = [
                cv2.getTrackbarPos('H Low', 'Threshold Tuner'),
                cv2.getTrackbarPos('S Low', 'Threshold Tuner'),
                cv2.getTrackbarPos('V Low', 'Threshold Tuner')
            ]
            self.hsv_upper = [
                cv2.getTrackbarPos('H High', 'Threshold Tuner'),
                cv2.getTrackbarPos('S High', 'Threshold Tuner'),
                cv2.getTrackbarPos('V High', 'Threshold Tuner')
            ]
            self.lab_lower = [
                cv2.getTrackbarPos('L Low', 'Threshold Tuner'),
                cv2.getTrackbarPos('A Low', 'Threshold Tuner'),
                cv2.getTrackbarPos('B Low', 'Threshold Tuner')
            ]
            self.lab_upper = [
                cv2.getTrackbarPos('L High', 'Threshold Tuner'),
                cv2.getTrackbarPos('A High', 'Threshold Tuner'),
                cv2.getTrackbarPos('B High', 'Threshold Tuner')
            ]
            
            self.triangulator.set_hsv_thresholds(self.hsv_lower, self.hsv_upper)
            self.triangulator.set_lab_thresholds(self.lab_lower, self.lab_upper)
        except:
            pass
    
    def create_debug_view(self, frame_left, frame_right):
        """Create debug visualization with masks for both cameras."""
        result_left = self.triangulator.tracker_left.detect(frame_left, return_debug=True)
        result_right = self.triangulator.tracker_right.detect(frame_right, return_debug=True)
        
        h, w = 120, 160
        
        def make_mask_vis(mask, label):
            if mask is None:
                return np.zeros((h, w, 3), dtype=np.uint8)
            m = cv2.resize(mask, (w, h))
            m = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
            cv2.putText(m, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            return m
        
        left_hsv = make_mask_vis(result_left.get('debug', {}).get('mask_hsv'), "L-HSV")
        left_lab = make_mask_vis(result_left.get('debug', {}).get('mask_lab'), "L-LAB")
        left_fused = make_mask_vis(result_left.get('mask'), "L-FUSED")
        
        right_hsv = make_mask_vis(result_right.get('debug', {}).get('mask_hsv'), "R-HSV")
        right_lab = make_mask_vis(result_right.get('debug', {}).get('mask_lab'), "R-LAB")
        right_fused = make_mask_vis(result_right.get('mask'), "R-FUSED")
        
        top_row = cv2.hconcat([left_hsv, left_lab, left_fused])
        bottom_row = cv2.hconcat([right_hsv, right_lab, right_fused])
        debug_view = cv2.vconcat([top_row, bottom_row])
        
        return debug_view
    
    def print_thresholds(self):
        """Print current threshold values."""
        print("\n" + "=" * 50)
        print("CURRENT THRESHOLDS")
        print("=" * 50)
        print(f"HSV Lower: {self.hsv_lower}")
        print(f"HSV Upper: {self.hsv_upper}")
        print(f"LAB Lower: {self.lab_lower}")
        print(f"LAB Upper: {self.lab_upper}")
        print("=" * 50)
    
    def print_controls(self):
        """Print control instructions."""
        print("\n" + "=" * 60)
        print("STEREO TRIANGULATION TEST (Basler acA1920-150uc)")
        print("=" * 60)
        print("\nVerify calibration by holding ball at known distances.")
        print(f"Resolution: {self.frame_width}x{self.frame_height}")
        print("\nCONTROLS:")
        print("  q - Quit")
        print("  s - Save current 3D measurement")
        print("  r - Reset/clear measurements")
        print("  t - Toggle threshold tuner (HSV + LAB sliders)")
        print("  d - Toggle debug view (show masks)")
        print("  p - Print current thresholds")
        print("  w - Save thresholds to JSON file")
        print("=" * 60)
    
    def start_cameras_with_basler(self):
        """Start cameras with Basler configuration."""
        print("\nOpening Basler cameras...")
        
        self.cam_left = get_basler_camera(
            serial=self.left_serial if self.left_serial else None,
            device_index=0
        )
        self.cam_right = get_basler_camera(
            serial=self.right_serial if self.right_serial else None,
            device_index=1
        )
        
        settings_left = configure_basler_camera(self.cam_left, self.frame_width, self.frame_height)
        settings_right = configure_basler_camera(self.cam_right, self.frame_width, self.frame_height)
        
        self.converter = create_image_converter()
        
        print(f"  LEFT:  {settings_left['actual_width']}x{settings_left['actual_height']} "
              f"(Serial: {self.cam_left.GetDeviceInfo().GetSerialNumber()})")
        print(f"  RIGHT: {settings_right['actual_width']}x{settings_right['actual_height']} "
              f"(Serial: {self.cam_right.GetDeviceInfo().GetSerialNumber()})")
    
    def grab_stereo_frames(self):
        """Grab frames from both cameras."""
        frame_left = grab_frame(self.cam_left, self.converter)
        frame_right = grab_frame(self.cam_right, self.converter)
        return frame_left, frame_right
    
    def run(self):
        """Main run loop."""
        self.print_controls()
        
        self.load_config()
        self.load_thresholds()
        
        if not self.check_calibration():
            return
        
        # Initialize triangulator (for calibration data and trackers only)
        try:
            self.triangulator = StereoTriangulator(
                calibration_dir=self.calibration_dir,
                cam_left_id=0,  # Not used - we handle cameras directly
                cam_right_id=1
            )
        except Exception as e:
            print(f"\nERROR initializing triangulator: {e}")
            return
        
        self.apply_thresholds()
        
        # Start Basler cameras directly
        try:
            self.start_cameras_with_basler()
            print("\nCameras started successfully!")
        except RuntimeError as e:
            print(f"\nERROR: {e}")
            return
        
        print("\n--- LIVE TRACKING ---")
        print("(Move ball in front of cameras)")
        
        try:
            while True:
                # Grab frames from Basler cameras
                frame_left, frame_right = self.grab_stereo_frames()
                
                if frame_left is None or frame_right is None:
                    continue
                
                self.update_from_tuner()
                
                # Detect ball in both frames
                result_left = self.triangulator.tracker_left.detect(frame_left)
                result_right = self.triangulator.tracker_right.detect(frame_right)
                
                # Triangulate if both found
                found_3d = False
                position_3d = None
                disparity = None
                
                if result_left['found'] and result_right['found']:
                    point_left = result_left['center']
                    point_right = result_right['center']
                    
                    disparity = point_left[0] - point_right[0]
                    
                    if disparity > 0:
                        X, Y, Z = self.triangulator.triangulate(point_left, point_right)
                        if Z > 0:
                            found_3d = True
                            position_3d = (X, Y, Z)
                
                # Draw results
                left_vis = frame_left.copy()
                right_vis = frame_right.copy()
                
                if result_left['found']:
                    center = result_left['center']
                    radius = int(result_left['radius'])
                    cv2.circle(left_vis, center, radius, (0, 255, 0), 2)
                    cv2.circle(left_vis, center, 3, (0, 0, 255), -1)
                
                if result_right['found']:
                    center = result_right['center']
                    radius = int(result_right['radius'])
                    cv2.circle(right_vis, center, radius, (0, 255, 0), 2)
                    cv2.circle(right_vis, center, 3, (0, 0, 255), -1)
                
                if found_3d:
                    X, Y, Z = position_3d
                    cv2.putText(left_vis, f"3D: X={X:.1f} Y={Y:.1f} Z={Z:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(left_vis, f"Disparity: {disparity:.1f}px", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Resize for display
                display_width = 640
                display_height = int(display_width * self.frame_height / self.frame_width)
                left_vis = cv2.resize(left_vis, (display_width, display_height))
                right_vis = cv2.resize(right_vis, (display_width, display_height))
                
                status = "TRACKING" if found_3d else "SEARCHING..."
                color = (0, 255, 0) if found_3d else (0, 0, 255)
                cv2.putText(left_vis, status, (10, display_height - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                l_det = "L: OK" if result_left['found'] else "L: --"
                r_det = "R: OK" if result_right['found'] else "R: --"
                cv2.putText(left_vis, l_det, (10, display_height - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(right_vis, r_det, (10, display_height - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                cv2.putText(left_vis, "q:quit t:tuner d:debug s:save w:write",
                           (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                combined = cv2.hconcat([left_vis, right_vis])
                
                if self.show_debug:
                    debug_view = self.create_debug_view(frame_left, frame_right)
                    cv2.imshow('Debug Masks', debug_view)
                
                cv2.imshow('Stereo Triangulation', combined)
                
                if found_3d:
                    X, Y, Z = position_3d
                    print(f"\r3D: X={X:7.1f}  Y={Y:7.1f}  Z={Z:7.1f}  (disp={disparity:5.1f}px)", end='')
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                
                elif key == ord('s'):
                    if found_3d:
                        X, Y, Z = position_3d
                        self.measurements.append((X, Y, Z))
                        print(f"\n\n[SAVED] #{len(self.measurements)}: X={X:.1f}, Y={Y:.1f}, Z={Z:.1f}")
                    else:
                        print("\n\n[ERROR] No ball detected - cannot save")
                
                elif key == ord('r'):
                    self.measurements = []
                    print("\n\n[RESET] Cleared all measurements")
                
                elif key == ord('t'):
                    self.show_tuner = not self.show_tuner
                    if self.show_tuner:
                        self.create_tuner_window()
                        print("\n[TUNER] Threshold tuner ENABLED")
                    else:
                        cv2.destroyWindow('Threshold Tuner')
                        print("\n[TUNER] Threshold tuner DISABLED")
                
                elif key == ord('d'):
                    self.show_debug = not self.show_debug
                    if not self.show_debug:
                        cv2.destroyWindow('Debug Masks')
                    print(f"\n[DEBUG] Debug view {'ENABLED' if self.show_debug else 'DISABLED'}")
                
                elif key == ord('p'):
                    self.print_thresholds()
                
                elif key == ord('w'):
                    self.save_thresholds()
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            if self.cam_left:
                self.cam_left.Close()
            if self.cam_right:
                self.cam_right.Close()
            cv2.destroyAllWindows()
        
        if self.measurements:
            print("\n" + "=" * 60)
            print("MEASUREMENT SUMMARY")
            print("=" * 60)
            for i, (X, Y, Z) in enumerate(self.measurements, 1):
                print(f"  #{i}: X={X:7.1f}  Y={Y:7.1f}  Z={Z:7.1f}")
            
            Z_values = [m[2] for m in self.measurements]
            print(f"\nDepth (Z) Statistics:")
            print(f"  Min:  {min(Z_values):7.1f}")
            print(f"  Max:  {max(Z_values):7.1f}")
            print(f"  Mean: {sum(Z_values)/len(Z_values):7.1f}")
            print(f"  Range: {max(Z_values)-min(Z_values):7.1f}")
            print("=" * 60)
        
        print("\nDone!")


def main():
    tester = TriangulationTester()
    tester.run()


if __name__ == '__main__':
    main()