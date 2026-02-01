"""
Test Triangulation - Stereo 3D Ball Tracking

Tests stereo calibration by detecting the ball in both cameras
and triangulating to get 3D position.

CAMERA: Arducam OV9782 Global Shutter USB Camera
        1MP, 100fps @ 1280x800 MJPG

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
q
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracking.stereo_triangulator import StereoTriangulator


def configure_camera_for_arducam(cap, width=1280, height=720):
    """
    Configure camera for Arducam OV9782 global shutter cameras.
    Forces MJPG codec and specified resolution.
    """
    fourcc_mjpg = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc_mjpg)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 100)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    actual_fourcc_str = "".join([chr((actual_fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
    
    return {
        'actual_width': actual_width,
        'actual_height': actual_height,
        'actual_fps': actual_fps,
        'actual_fourcc': actual_fourcc_str,
        'settings_match': (actual_width == width and actual_height == height)
    }


class TriangulationTester:
    """Full-featured triangulation tester with debug and tuning modes."""
    
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.calibration_dir = os.path.join(self.script_dir, '..', 'camera_calibration', 'camera_parameters')
        self.config_path = os.path.join(self.script_dir, '..', 'config', 'stereo_config.yaml')
        self.thresholds_path = os.path.join(self.script_dir, '..', 'config', 'ball_thresholds.json')
        
        self.frame_width = 1280
        self.frame_height = 720
        
        self.triangulator = None
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
        """Load camera IDs and resolution from config."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.cam_left_id = config['camera_left']['id']
            self.cam_right_id = config['camera_right']['id']
            self.frame_width = config.get('frame_width', 1280)
            self.frame_height = config.get('frame_height', 720)
            print(f"Loaded config: Left=ID{self.cam_left_id}, Right=ID{self.cam_right_id}")
            print(f"  Resolution: {self.frame_width}x{self.frame_height}")
        else:
            self.cam_left_id = 1
            self.cam_right_id = 2
            print(f"No config found, using defaults: Left=ID{self.cam_left_id}, Right=ID{self.cam_right_id}")
    
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
            print("  python calibrate.py calibration_settings.yaml")
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
        print("STEREO TRIANGULATION TEST (Arducam OV9782)")
        print("=" * 60)
        print("\nVerify calibration by holding ball at known distances.")
        print(f"Resolution: {self.frame_width}x{self.frame_height} (MJPG)")
        print("\nCONTROLS:")
        print("  q - Quit")
        print("  s - Save current 3D measurement")
        print("  r - Reset/clear measurements")
        print("  t - Toggle threshold tuner (HSV + LAB sliders)")
        print("  d - Toggle debug view (show masks)")
        print("  p - Print current thresholds")
        print("  w - Save thresholds to JSON file")
        print("=" * 60)
    
    def start_cameras_with_arducam_config(self):
        """Start cameras with Arducam OV9782 configuration."""
        self.triangulator.cap_left = cv2.VideoCapture(self.cam_left_id)
        self.triangulator.cap_right = cv2.VideoCapture(self.cam_right_id)
        
        if not self.triangulator.cap_left.isOpened():
            raise RuntimeError(f"Failed to open left camera (ID: {self.cam_left_id})")
        if not self.triangulator.cap_right.isOpened():
            raise RuntimeError(f"Failed to open right camera (ID: {self.cam_right_id})")
        
        print("\nConfiguring cameras (Arducam OV9782 MJPG mode):")
        settings_left = configure_camera_for_arducam(
            self.triangulator.cap_left, self.frame_width, self.frame_height)
        settings_right = configure_camera_for_arducam(
            self.triangulator.cap_right, self.frame_width, self.frame_height)
        
        print(f"  LEFT:  {settings_left['actual_width']}x{settings_left['actual_height']} "
              f"@ {settings_left['actual_fps']:.0f}fps ({settings_left['actual_fourcc']})")
        print(f"  RIGHT: {settings_right['actual_width']}x{settings_right['actual_height']} "
              f"@ {settings_right['actual_fps']:.0f}fps ({settings_right['actual_fourcc']})")
    
    def run(self):
        """Main run loop."""
        self.print_controls()
        
        self.load_config()
        self.load_thresholds()
        
        if not self.check_calibration():
            return
        
        try:
            self.triangulator = StereoTriangulator(
                calibration_dir=self.calibration_dir,
                cam_left_id=self.cam_left_id,
                cam_right_id=self.cam_right_id
            )
        except Exception as e:
            print(f"\nERROR initializing triangulator: {e}")
            return
        
        self.apply_thresholds()
        
        try:
            self.start_cameras_with_arducam_config()
            print("\nCameras started successfully!")
        except RuntimeError as e:
            print(f"\nERROR: {e}")
            return
        
        print("\n--- LIVE TRACKING ---")
        print("(Move ball in front of cameras)")
        
        try:
            while True:
                result = self.triangulator.update()
                
                if result['left_frame'] is None or result['right_frame'] is None:
                    continue
                
                self.update_from_tuner()
                
                left_vis, right_vis = self.triangulator.draw_results(result)
                
                display_width = 640
                display_height = int(display_width * self.frame_height / self.frame_width)
                left_vis = cv2.resize(left_vis, (display_width, display_height))
                right_vis = cv2.resize(right_vis, (display_width, display_height))
                
                status = "TRACKING" if result['found_3d'] else "SEARCHING..."
                color = (0, 255, 0) if result['found_3d'] else (0, 0, 255)
                cv2.putText(left_vis, status, (10, display_height - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                l_det = "L: OK" if result['left_detection']['found'] else "L: --"
                r_det = "R: OK" if result['right_detection']['found'] else "R: --"
                cv2.putText(left_vis, l_det, (10, display_height - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(right_vis, r_det, (10, display_height - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                cv2.putText(left_vis, "q:quit t:tuner d:debug s:save w:write",
                           (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                combined = cv2.hconcat([left_vis, right_vis])
                
                if self.show_debug:
                    debug_view = self.create_debug_view(
                        result['left_frame'], result['right_frame'])
                    cv2.imshow('Debug Masks', debug_view)
                
                cv2.imshow('Stereo Triangulation', combined)
                
                if result['found_3d']:
                    X, Y, Z = result['position_3d']
                    disp = result['disparity']
                    print(f"\r3D: X={X:7.1f}  Y={Y:7.1f}  Z={Z:7.1f}  (disp={disp:5.1f}px)", end='')
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                
                elif key == ord('s'):
                    if result['found_3d']:
                        X, Y, Z = result['position_3d']
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
            self.triangulator.stop_cameras()
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