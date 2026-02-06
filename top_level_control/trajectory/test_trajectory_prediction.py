"""
Test Trajectory Prediction - Live Integration Test

Tests the full trajectory prediction pipeline with live cameras:
  Cameras → Detection → Triangulation → Prediction → Visualization

CAMERA: Arducam OV9782 Global Shutter USB Camera
        1MP, 120fps @ 1280x720

USAGE:
    cd top_level_control
    python -m trajectory.test_trajectory_prediction

    OR place this file in tests/ folder and run:
    python test_trajectory_prediction.py

CONTROLS:
    q - Quit
    r - Reset predictor (clear history)
    v - Toggle velocity display
    t - Toggle trajectory visualization
    p - Print current stats
    z/x - Increase/decrease robot Z plane

VISUALIZATION:
    - Green circle: Detected ball
    - Blue line: Predicted trajectory
    - Red X: Predicted interception point
"""

import cv2
import sys
import os
import time
import numpy as np
import yaml
import json

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tracking.stereo_triangulator import StereoTriangulator
from trajectory.trajectory_predictor import TrajectoryPredictor


def configure_camera_for_arducam(cap, width=1280, height=720):
    """Configure camera for Arducam OV9782 with MJPG codec."""
    fourcc_mjpg = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc_mjpg)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 120)
    
    return {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS)
    }


class TrajectoryTester:
    """Test trajectory prediction with live stereo cameras."""
    
    def __init__(self):
        # Determine base directory (parent of trajectory/ or tests/)
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = os.path.dirname(self.script_dir)
        
        # File paths
        self.calibration_dir = os.path.join(self.base_dir, 'camera_calibration', 'camera_parameters')
        self.config_path = os.path.join(self.base_dir, 'config', 'stereo_config.yaml')
        self.thresholds_stereo = os.path.join(self.base_dir, 'config', 'ball_thresholds_stereo.json')
        self.thresholds_single = os.path.join(self.base_dir, 'config', 'ball_thresholds.json')
        
        # Camera settings
        self.frame_width = 1280
        self.frame_height = 720
        self.cam_left_id = 1
        self.cam_right_id = 2
        
        # Robot interception plane (Z distance where robot can reach)
        self.robot_z = 50.0  # cm - ADJUST THIS FOR YOUR SETUP
        
        # Components
        self.triangulator = None
        self.predictor = None
        
        # Display options
        self.show_velocity = True
        self.show_trajectory = True
        
        self.load_config()
    
    def load_config(self):
        """Load camera configuration."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.cam_left_id = config.get('camera_left', {}).get('id', 1)
            self.cam_right_id = config.get('camera_right', {}).get('id', 2)
            self.frame_width = config.get('frame_width', 1280)
            self.frame_height = config.get('frame_height', 720)
            print(f"[Config] Loaded: Left={self.cam_left_id}, Right={self.cam_right_id}, {self.frame_width}x{self.frame_height}")
    
    def load_thresholds(self):
        """Load ball detection thresholds (stereo or single)."""
        if os.path.exists(self.thresholds_stereo):
            self.triangulator.load_thresholds(self.thresholds_stereo)
            print(f"[Thresholds] Loaded stereo thresholds")
        elif os.path.exists(self.thresholds_single):
            self.triangulator.load_thresholds(self.thresholds_single)
            print(f"[Thresholds] Loaded single thresholds")
        else:
            print("[Thresholds] No threshold file found, using defaults")
    
    def start_cameras(self):
        """Start cameras with DirectShow backend."""
        print("\n[Cameras] Opening with DirectShow backend...")
        
        self.triangulator.cap_left = cv2.VideoCapture(self.cam_left_id, cv2.CAP_DSHOW)
        self.triangulator.cap_right = cv2.VideoCapture(self.cam_right_id, cv2.CAP_DSHOW)
        
        # Fallback to default if DirectShow fails
        if not self.triangulator.cap_left.isOpened():
            self.triangulator.cap_left = cv2.VideoCapture(self.cam_left_id)
        if not self.triangulator.cap_right.isOpened():
            self.triangulator.cap_right = cv2.VideoCapture(self.cam_right_id)
        
        if not self.triangulator.cap_left.isOpened() or not self.triangulator.cap_right.isOpened():
            raise RuntimeError("Failed to open cameras")
        
        # Configure for Arducam
        s_left = configure_camera_for_arducam(self.triangulator.cap_left, self.frame_width, self.frame_height)
        s_right = configure_camera_for_arducam(self.triangulator.cap_right, self.frame_width, self.frame_height)
        
        print(f"  LEFT:  {s_left['width']}x{s_left['height']} @ {s_left['fps']:.0f}fps")
        print(f"  RIGHT: {s_right['width']}x{s_right['height']} @ {s_right['fps']:.0f}fps")
        
        self.frame_width = s_left['width']
        self.frame_height = s_left['height']
    
    def draw_trajectory(self, frame, trajectory, color=(255, 100, 0)):
        """Draw predicted trajectory on frame."""
        if len(trajectory) < 2:
            return
        
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        scale = 3.0  # Pixels per cm
        
        points = []
        for x, y, z, t in trajectory:
            px = int(cx + x * scale)
            py = int(cy + y * scale)
            px = max(0, min(w - 1, px))
            py = max(0, min(h - 1, py))
            points.append((px, py))
        
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], color, 2)
        
        for pt in points[::10]:
            cv2.circle(frame, pt, 3, color, -1)
    
    def draw_intercept(self, frame, x, y, t_ms, color=(0, 0, 255)):
        """Draw interception point."""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        scale = 3.0
        
        px = int(cx + x * scale)
        py = int(cy + y * scale)
        px = max(20, min(w - 20, px))
        py = max(20, min(h - 20, py))
        
        # X marker
        cv2.line(frame, (px - 15, py - 15), (px + 15, py + 15), color, 3)
        cv2.line(frame, (px - 15, py + 15), (px + 15, py - 15), color, 3)
        cv2.circle(frame, (px, py), 20, color, 2)
        
        cv2.putText(frame, f"({x:.0f},{y:.0f}) {t_ms:.0f}ms", (px + 25, py),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def run(self):
        """Main loop."""
        print("\n" + "=" * 60)
        print("    TRAJECTORY PREDICTION TEST (Arducam OV9782)")
        print("=" * 60)
        print(f"\nRobot Z plane: {self.robot_z} cm")
        print("\nControls: q=quit r=reset v=velocity t=trajectory p=stats z/x=adjust Z")
        print("=" * 60)
        
        # Initialize triangulator
        try:
            self.triangulator = StereoTriangulator(
                calibration_dir=self.calibration_dir,
                cam_left_id=self.cam_left_id,
                cam_right_id=self.cam_right_id
            )
        except Exception as e:
            print(f"\nERROR: {e}")
            print(f"  Calibration path: {self.calibration_dir}")
            return
        
        self.load_thresholds()
        
        try:
            self.start_cameras()
        except Exception as e:
            print(f"\nERROR: {e}")
            return
        
        # Initialize predictor
        self.predictor = TrajectoryPredictor(
            buffer_size=10,
            min_points=3,
            velocity_method='regression',
            gravity=981.0,
            y_down=True
        )
        
        print("\n--- LIVE TRACKING ---\n")
        
        fps_time = time.time()
        fps = 0
        frame_count = 0
        
        try:
            while True:
                result = self.triangulator.update()
                if result['left_frame'] is None:
                    continue
                
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = 30.0 / (time.time() - fps_time)
                    fps_time = time.time()
                
                # Add position to predictor
                if result['found_3d']:
                    self.predictor.add_position(*result['position_3d'])
                
                # Get prediction
                pred = self.predictor.predict(target_z=self.robot_z)
                
                # Draw
                left_vis, right_vis = self.triangulator.draw_results(result)
                
                if self.show_trajectory and self.predictor.is_ready():
                    traj = self.predictor.predict_trajectory(duration=0.5, dt=0.01)
                    self.draw_trajectory(left_vis, traj)
                    if pred['valid']:
                        self.draw_intercept(left_vis, pred['intercept_x'], pred['intercept_y'],
                                          pred['time_to_intercept'] * 1000)
                
                if self.show_velocity:
                    vel = self.predictor.get_velocity()
                    stats = self.predictor.get_stats()
                    y = 90
                    cv2.putText(left_vis, f"Buffer: {stats['buffer_size']}/10  FPS: {fps:.0f}",
                               (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    if vel['valid']:
                        cv2.putText(left_vis, f"Velocity: Vz={vel['vz']:.0f} Speed={vel['speed']:.0f} cm/s",
                                   (10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                if pred['valid']:
                    strat = pred.get('strategy', 'z_plane')
                    if strat == 'apex':
                        osd_text = f"[APEX] X={pred['intercept_x']:.0f} Y={pred['intercept_y']:.0f} Z={pred['intercept_z']:.0f} in {pred['time_to_intercept']*1000:.0f}ms"
                    else:
                        osd_text = f"[Z-PLANE] Z={self.robot_z:.0f}: X={pred['intercept_x']:.0f} Y={pred['intercept_y']:.0f} in {pred['time_to_intercept']*1000:.0f}ms"
                    cv2.putText(left_vis, osd_text,
                               (10, left_vis.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Display
                dw = 640
                dh = int(dw * self.frame_height / self.frame_width)
                left_vis = cv2.resize(left_vis, (dw, dh))
                right_vis = cv2.resize(right_vis, (dw, dh))
                cv2.imshow('Trajectory Prediction', cv2.hconcat([left_vis, right_vis]))
                
                # Console output
                if result['found_3d'] and self.predictor.get_velocity()['valid']:
                    x, y, z = result['position_3d']
                    v = self.predictor.get_velocity()
                    print(f"\rPos:({x:5.0f},{y:5.0f},{z:5.0f}) Vel:({v['vx']:5.0f},{v['vy']:5.0f},{v['vz']:5.0f}) ", end='')
                    if pred['valid']:
                        strat_label = pred.get('strategy', 'z_plane').upper().replace('_', '-')
                        print(f"[{strat_label}] Intercept:({pred['intercept_x']:5.0f},{pred['intercept_y']:5.0f},{pred['intercept_z']:5.0f}) in {pred['time_to_intercept']*1000:4.0f}ms", end='')
                    print("   ", end='')
                
                # Keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.predictor.reset()
                    print("\n[RESET]\n")
                elif key == ord('v'):
                    self.show_velocity = not self.show_velocity
                elif key == ord('t'):
                    self.show_trajectory = not self.show_trajectory
                elif key == ord('p'):
                    print(f"\n[STATS] {self.predictor.get_stats()}\n")
                elif key == ord('z'):
                    self.robot_z += 10
                    print(f"\n[Robot Z = {self.robot_z}]\n")
                elif key == ord('x'):
                    self.robot_z = max(10, self.robot_z - 10)
                    print(f"\n[Robot Z = {self.robot_z}]\n")
        
        except KeyboardInterrupt:
            pass
        finally:
            self.triangulator.stop_cameras()
            cv2.destroyAllWindows()
        
        print("\nDone!")


if __name__ == '__main__':
    TrajectoryTester().run()