"""
Test Trajectory Prediction - Basler Cameras

Live test of trajectory prediction with stereo triangulation.

CAMERA: Basler acA1920-150uc USB 3.0
        2.3MP, 150fps @ 1920x1200

REQUIREMENTS:
    pip install pypylon

USAGE:
    cd basler_pipeline
    python -m trajectory.test_trajectory_prediction

CONTROLS:
    q - Quit
    r - Reset predictor (clear history)
    v - Toggle velocity display
    t - Toggle trajectory visualization
    p - Print current stats
    z - Increase robot Z plane (+10)
    x - Decrease robot Z plane (-10)
"""

import cv2
import sys
import os
import time
import numpy as np
import yaml

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tracking.stereo_triangulator import StereoTriangulator
from trajectory.trajectory_predictor import TrajectoryPredictor


class TrajectoryTester:
    """Test trajectory prediction with Basler stereo cameras."""
    
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = os.path.dirname(self.script_dir)
        
        # Paths
        self.calibration_dir = os.path.join(self.base_dir, 'camera_calibration', 'camera_parameters')
        self.config_path = os.path.join(self.base_dir, 'config', 'stereo_config.yaml')
        self.thresholds_stereo = os.path.join(self.base_dir, 'config', 'ball_thresholds_stereo.json')
        self.thresholds_single = os.path.join(self.base_dir, 'config', 'ball_thresholds.json')
        
        # Camera settings (Basler defaults)
        self.frame_width = 1920
        self.frame_height = 1200
        self.left_serial = None
        self.right_serial = None
        
        # Robot interception plane
        self.robot_z = 50.0  # cm
        
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
            
            self.left_serial = config.get('camera_left', {}).get('serial', '') or None
            self.right_serial = config.get('camera_right', {}).get('serial', '') or None
            self.frame_width = config.get('frame_width', 1920)
            self.frame_height = config.get('frame_height', 1200)
            
            print(f"[Config] Resolution: {self.frame_width}x{self.frame_height}")
    
    def load_thresholds(self):
        """Load ball detection thresholds."""
        if os.path.exists(self.thresholds_stereo):
            self.triangulator.load_thresholds(self.thresholds_stereo)
        elif os.path.exists(self.thresholds_single):
            self.triangulator.load_thresholds(self.thresholds_single)
    
    def draw_trajectory(self, frame, trajectory, color=(255, 100, 0)):
        """Draw predicted trajectory on frame."""
        if len(trajectory) < 2:
            return
        
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        scale = 5.0  # Adjust based on your setup
        
        points = []
        for x, y, z, t in trajectory:
            px = int(cx + x * scale)
            py = int(cy + y * scale)
            px = max(0, min(w - 1, px))
            py = max(0, min(h - 1, py))
            points.append((px, py))
        
        # Draw trajectory line
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], color, 2)
        
        # Draw points
        for pt in points[::10]:
            cv2.circle(frame, pt, 3, color, -1)
    
    def draw_intercept(self, frame, x, y, t_ms, color=(0, 0, 255)):
        """Draw interception point."""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        scale = 5.0
        
        px = int(cx + x * scale)
        py = int(cy + y * scale)
        px = max(20, min(w - 20, px))
        py = max(20, min(h - 20, py))
        
        # X marker
        cv2.line(frame, (px - 15, py - 15), (px + 15, py + 15), color, 3)
        cv2.line(frame, (px - 15, py + 15), (px + 15, py - 15), color, 3)
        cv2.circle(frame, (px, py), 20, color, 2)
        
        # Label
        cv2.putText(frame, f"({x:.0f},{y:.0f}) {t_ms:.0f}ms", (px + 25, py),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def run(self):
        """Main loop."""
        print("\n" + "=" * 60)
        print("    TRAJECTORY PREDICTION TEST (Basler acA1920-150uc)")
        print("=" * 60)
        print(f"\nRobot Z plane: {self.robot_z} cm")
        print("\nControls: q=quit r=reset v=velocity t=trajectory z/x=adjust Z")
        print("=" * 60)
        
        # Initialize triangulator
        try:
            self.triangulator = StereoTriangulator(
                calibration_dir=self.calibration_dir,
                left_serial=self.left_serial,
                right_serial=self.right_serial
            )
        except Exception as e:
            print(f"\nERROR: {e}")
            return
        
        self.load_thresholds()
        
        try:
            self.triangulator.start_cameras(self.frame_width, self.frame_height)
        except Exception as e:
            print(f"\nERROR starting cameras: {e}")
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
                    elapsed = time.time() - fps_time
                    fps = 30.0 / elapsed if elapsed > 0 else 0
                    fps_time = time.time()
                
                # Add position to predictor
                if result['found_3d']:
                    self.predictor.add_position(*result['position_3d'])
                
                # Get prediction
                pred = self.predictor.predict(target_z=self.robot_z)
                
                # Draw results
                left_vis, right_vis = self.triangulator.draw_results(result)
                
                # Draw trajectory
                if self.show_trajectory and self.predictor.is_ready():
                    traj = self.predictor.predict_trajectory(duration=0.5, dt=0.01)
                    self.draw_trajectory(left_vis, traj)
                    if pred['valid']:
                        self.draw_intercept(left_vis, pred['intercept_x'], 
                                          pred['intercept_y'],
                                          pred['time_to_intercept'] * 1000)
                
                # Draw velocity info
                if self.show_velocity:
                    vel = self.predictor.get_velocity()
                    stats = self.predictor.get_stats()
                    y = 90
                    cv2.putText(left_vis, f"Buffer: {stats['buffer_size']}/10  FPS: {fps:.0f}",
                               (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    if vel['valid']:
                        cv2.putText(left_vis, f"Vz={vel['vz']:.0f} Speed={vel['speed']:.0f} cm/s",
                                   (10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Draw prediction
                if pred['valid']:
                    text = f"INTERCEPT Z={self.robot_z:.0f}: ({pred['intercept_x']:.0f},{pred['intercept_y']:.0f}) in {pred['time_to_intercept']*1000:.0f}ms"
                    cv2.putText(left_vis, text, (10, left_vis.shape[0] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Resize for display (1920x1200 is large)
                dw = 800
                dh = int(dw * self.frame_height / self.frame_width)
                left_small = cv2.resize(left_vis, (dw, dh))
                right_small = cv2.resize(right_vis, (dw, dh))
                
                combined = cv2.hconcat([left_small, right_small])
                cv2.imshow('Trajectory Prediction', combined)
                
                # Console output
                if result['found_3d'] and self.predictor.get_velocity()['valid']:
                    x, y, z = result['position_3d']
                    v = self.predictor.get_velocity()
                    msg = f"\rPos:({x:5.0f},{y:5.0f},{z:5.0f}) Vel:({v['vx']:5.0f},{v['vy']:5.0f},{v['vz']:5.0f})"
                    if pred['valid']:
                        msg += f" -> Int:({pred['intercept_x']:5.0f},{pred['intercept_y']:5.0f}) {pred['time_to_intercept']*1000:4.0f}ms"
                    print(msg + "   ", end='')
                
                # Handle keys
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