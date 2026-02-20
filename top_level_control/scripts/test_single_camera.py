"""
Test Single Camera - Threshold Tuning

Test ball detection on a single camera with HSV + LAB threshold tuning.
Use this to calibrate detection thresholds for your lighting conditions.

CAMERA: Arducam OV9782 Global Shutter USB Camera
        1MP, 100fps @ 1280x800 MJPG

CONTROLS:
    q - Quit
    t - Toggle threshold tuner (HSV + LAB sliders)
    d - Toggle debug view (show HSV, LAB, fused masks)
    s - Save thresholds to ball_thresholds.json
    p - Print current thresholds
    c - Cycle camera (0, 1, 2...)
"""

import cv2
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracking.ball_tracker import EnhancedBallTracker
from config.camera_config import load_camera_settings, configure_camera


class SingleCameraTester:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.thresholds_path = os.path.join(self.script_dir, '..', 'config', 'ball_thresholds.json')

        cam_settings = load_camera_settings()
        self.camera_id = cam_settings['camera0']
        self.cap = None
        self.tracker = EnhancedBallTracker()

        self.frame_width = cam_settings['frame_width']
        self.frame_height = cam_settings['frame_height']

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
        """Open camera with Arducam configuration."""
        if self.cap:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"Failed to open camera {self.camera_id}")
            return False
        
        settings = configure_camera(self.cap, self.frame_width, self.frame_height)

        print(f"\nCamera {self.camera_id} opened:")
        print(f"  Resolution: {settings['width']}x{settings['height']}")
        print(f"  FPS: {settings['fps']:.0f}")
        print(f"  FOURCC: {settings['fourcc']}")
        
        if not settings['settings_match']:
            print("  WARNING: Resolution mismatch!")
        
        return True
    
    def run(self):
        """Main loop."""
        print("\n" + "=" * 60)
        print("SINGLE CAMERA TEST - Threshold Tuning (Arducam OV9782)")
        print("=" * 60)
        print(f"\nResolution: {self.frame_width}x{self.frame_height} (MJPG)")
        print("\nCONTROLS:")
        print("  q - Quit")
        print("  t - Toggle threshold tuner")
        print("  d - Toggle debug view")
        print("  s - Save thresholds to JSON")
        print("  p - Print current thresholds")
        print("  c - Cycle camera ID")
        print("=" * 60)
        
        if not self.start_camera():
            return
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
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
                
                cv2.putText(vis, f"Camera: {self.camera_id}", (10, vis.shape[0] - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(vis, "q:quit t:tuner d:debug s:save", (10, vis.shape[0] - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                display_width = 720
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
                    self.camera_id = (self.camera_id + 1) % 5
                    self.start_camera()
        
        except KeyboardInterrupt:
            pass
        
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
        
        print("\nDone!")


def main():
    tester = SingleCameraTester()
    tester.run()


if __name__ == '__main__':
    main()