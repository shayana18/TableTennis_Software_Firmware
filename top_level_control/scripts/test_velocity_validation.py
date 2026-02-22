"""
Velocity & Prediction Validation Test
=======================================
Validates the trajectory pipeline by comparing predicted vs actual ball positions.

WORKFLOW (arm-and-capture):
    1. Script starts → Recording #1 ARMED (green overlay)
    2. Toss ball → detection auto-starts recording
    3. Ball lost for 10+ consecutive frames → recording stops
    4. Press 'n' → arm next recording (#2, #3, ...)
    5. Press 'a' → analyze last recording (show plots)
    6. Press 'q' → quit

LIVE OVERLAY:
    - Green circle: Actual detected position
    - Magenta diamond: Where the model PREDICTED the ball would be
    - White line: Prediction error vector
    - The closer they are, the better the model

CONTROLS:
    q       - Quit
    n       - Arm next recording
    SPACE   - Manual start/stop override
    a       - Analyze last recording (show plots)
    r       - Reset / clear everything
    p       - Print current velocity stats

CAMERA: Arducam OV9782 Global Shutter USB Camera
        1MP, 100fps @ 1280x800 MJPG
"""

import cv2
import sys
import os
import time
import numpy as np
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracking.stereo_triangulator import StereoTriangulator
from trajectory.trajectory_predictor import TrajectoryPredictor
from trajectory.physics_model import PhysicsModel
from config.camera_config import load_camera_settings, configure_camera


class VelocityValidator:
    """Validates velocity estimation and trajectory prediction accuracy."""

    # How many frames ahead to predict for comparison
    LOOKAHEAD_FRAMES = 5

    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.calibration_dir = os.path.join(
            self.script_dir, '..', 'camera_calibration', 'camera_parameters')
        self.thresholds_stereo = os.path.join(
            self.script_dir, '..', 'config', 'ball_thresholds_stereo.json')
        self.thresholds_single = os.path.join(
            self.script_dir, '..', 'config', 'ball_thresholds.json')

        cam_settings = load_camera_settings()
        self.frame_width = cam_settings['frame_width']
        self.frame_height = cam_settings['frame_height']
        self.cam_left_id = cam_settings['camera0']
        self.cam_right_id = cam_settings['camera1']
        self.display_width = 640

        self.triangulator = None
        self.predictor = None

        # Arm-and-capture state
        self.recording = False
        self.armed = True               # Start armed for first recording
        self.consecutive_losses = 0
        self.GRACE_FRAMES = 10          # ~100ms at 100fps — tolerates brief flickers
        self.recording_number = 1
        self.all_recordings = []        # Past recordings for multi-toss review

        # Motion gate — skip detection when nothing is moving (saves CPU + prevents false triggers)
        self._prev_gray = None
        self._motion_detected = False
        self.MOTION_THRESH = 25         # Pixel intensity change to count as motion
        self.MOTION_MIN_FRAC = 0.003    # 0.3% of pixels must move

        # Recorded data for current recording
        self.recorded_positions = []   # (x, y, z, t)
        self.recorded_velocities = []  # (vx, vy, vz, speed, t)
        self.recorded_predictions = [] # (pred_x, pred_y, pred_z, actual_x, actual_y, actual_z, dt, t)

        # Live prediction for overlay
        self.live_predicted_pos = None  # Where we predicted the ball would be NOW
        self.live_prediction_error = None

        self.load_config()

    def load_config(self):
        pass  # Settings already loaded in __init__ via load_camera_settings()

    def load_thresholds(self):
        if os.path.exists(self.thresholds_stereo):
            self.triangulator.load_thresholds(self.thresholds_stereo)
        elif os.path.exists(self.thresholds_single):
            self.triangulator.load_thresholds(self.thresholds_single)

    def check_calibration(self):
        required = ['camera0_intrinsics.dat', 'camera1_intrinsics.dat',
                     'camera0_rot_trans.dat', 'camera1_rot_trans.dat']
        missing = [f for f in required
                   if not os.path.exists(os.path.join(self.calibration_dir, f))]
        if missing:
            print("ERROR: Missing calibration files:")
            for f in missing:
                print(f"  - {f}")
            return False
        return True

    def start_cameras(self):
        self.triangulator.cap_left = cv2.VideoCapture(self.cam_left_id)
        self.triangulator.cap_right = cv2.VideoCapture(self.cam_right_id)

        if not self.triangulator.cap_left.isOpened():
            raise RuntimeError(f"Failed to open left camera (ID: {self.cam_left_id})")
        if not self.triangulator.cap_right.isOpened():
            raise RuntimeError(f"Failed to open right camera (ID: {self.cam_right_id})")

        s_l = configure_camera(self.triangulator.cap_left, self.frame_width, self.frame_height)
        s_r = configure_camera(self.triangulator.cap_right, self.frame_width, self.frame_height)
        print(f"  LEFT:  {s_l['width']}x{s_l['height']} @ {s_l['fps']:.0f}fps")
        print(f"  RIGHT: {s_r['width']}x{s_r['height']} @ {s_r['fps']:.0f}fps")

    def has_motion(self, frame):
        """Lightweight motion check — compare current frame to previous."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (320, 200))

        if self._prev_gray is None:
            self._prev_gray = small
            return False

        diff = cv2.absdiff(self._prev_gray, small)
        self._prev_gray = small

        motion_pixels = np.count_nonzero(diff > self.MOTION_THRESH)
        total_pixels = small.shape[0] * small.shape[1]
        self._motion_detected = motion_pixels > (total_pixels * self.MOTION_MIN_FRAC)
        return self._motion_detected

    def capture_and_detect(self):
        """Read frames, apply motion gate, detect only when needed."""
        ret_l, frame_l = self.triangulator.cap_left.read()
        ret_r, frame_r = self.triangulator.cap_right.read()

        if not ret_l or not ret_r:
            return {'left_frame': None, 'right_frame': None,
                    'left_detection': {'found': False},
                    'right_detection': {'found': False},
                    'found_3d': False, 'position_3d': None, 'disparity': None}

        result = {
            'left_frame': frame_l, 'right_frame': frame_r,
            'left_detection': {'found': False},
            'right_detection': {'found': False},
            'found_3d': False, 'position_3d': None, 'disparity': None
        }

        # Motion gate: skip expensive detection when nothing is moving
        run_detection = self.recording or self.has_motion(frame_l)
        if not run_detection:
            return result

        # Full detection
        det_l = self.triangulator.tracker_left.detect(frame_l)
        det_r = self.triangulator.tracker_right.detect(frame_r)
        result['left_detection'] = det_l
        result['right_detection'] = det_r

        if det_l['found'] and det_r['found']:
            disparity = det_l['center'][0] - det_r['center'][0]
            result['disparity'] = disparity
            if disparity > 0:
                pos = self.triangulator.triangulate(det_l['center'], det_r['center'])
                if pos[2] > 0:
                    result['found_3d'] = True
                    result['position_3d'] = tuple(pos)

        return result

    def reset_recording(self):
        self.recorded_positions = []
        self.recorded_velocities = []
        self.recorded_predictions = []
        self.live_predicted_pos = None
        self.live_prediction_error = None
        self.consecutive_losses = 0
        self._prev_gray = None
        self.predictor.reset()
        self.recording = False
        self.armed = True
        self.recording_number = 1
        self.all_recordings = []
        print(f"\n[RESET] All cleared. Recording #{self.recording_number} armed.")

    def process_frame(self, result):
        """Process one frame: record data + make predictions."""
        t_now = time.perf_counter()

        if not result['found_3d']:
            if self.recording:
                self.consecutive_losses += 1
                if self.consecutive_losses >= self.GRACE_FRAMES:
                    self.recording = False
                    n = len(self.recorded_positions)
                    print(f"\n[REC #{self.recording_number} STOPPED] {n} frames captured.")
                    print("  Press 'n' for next recording, 'a' to analyze.")
            self.live_predicted_pos = None
            self.live_prediction_error = None
            return

        x, y, z = result['position_3d']

        # Armed and first detection → start recording
        if self.armed and not self.recording:
            self.recording = True
            self.armed = False
            self.consecutive_losses = 0
            self.recorded_positions = []
            self.recorded_velocities = []
            self.recorded_predictions = []
            self.predictor.reset()
            print(f"\n[REC #{self.recording_number} STARTED] Ball detected, recording...")

        if not self.recording:
            return

        # Ball re-found during grace period — reset loss counter
        self.consecutive_losses = 0

        # Add position to predictor
        self.predictor.add_position(x, y, z, t_now)

        # Record position
        self.recorded_positions.append((x, y, z, t_now))

        # Record velocity
        vel = self.predictor.get_velocity()
        if vel['valid']:
            self.recorded_velocities.append((vel['vx'], vel['vy'], vel['vz'], vel['speed'], t_now))

        # --- Live prediction comparison ---
        # Check if a previous prediction matches current time
        if self.live_predicted_pos is not None:
            pred = self.live_predicted_pos
            self.live_prediction_error = np.sqrt(
                (pred[0] - x)**2 + (pred[1] - y)**2 + (pred[2] - z)**2)

            # Record for post-analysis
            self.recorded_predictions.append((
                pred[0], pred[1], pred[2],  # predicted
                x, y, z,                     # actual
                pred[3],                      # prediction horizon (dt)
                t_now
            ))

        # Make prediction for NEXT frame (lookahead)
        if self.predictor.is_ready():
            # Predict position LOOKAHEAD_FRAMES frames ahead
            # Estimate dt from buffer
            avg_dt = self.predictor.position_buffer.get_average_dt()
            if avg_dt > 0:
                lookahead_dt = avg_dt * self.LOOKAHEAD_FRAMES
            else:
                lookahead_dt = 0.05  # 50ms default

            current_pos = self.predictor.get_current_position()
            current_vel = self.predictor.get_velocity()

            if current_pos and current_vel['valid']:
                # Apply the same Vy correction as the main predictor
                positions_arr, timestamps_arr = self.predictor.position_buffer.get_as_arrays()
                t_latest = timestamps_arr[-1]
                t_mean = timestamps_arr.mean()
                vy_corrected = current_vel['vy'] + (
                    self.predictor.physics_model.gravity_sign *
                    self.predictor.physics_model.gravity *
                    (t_latest - t_mean)
                )
                corrected_vel = (current_vel['vx'], vy_corrected, current_vel['vz'])

                pred_pos = self.predictor.physics_model.predict_position(
                    current_pos, corrected_vel, lookahead_dt)
                self.live_predicted_pos = (pred_pos[0], pred_pos[1], pred_pos[2], lookahead_dt)
            else:
                self.live_predicted_pos = None
        else:
            self.live_predicted_pos = None

    def draw_overlay(self, left_small, right_small, result):
        """Draw live prediction overlay on frames."""
        dw = self.display_width
        dh = int(dw * self.frame_height / self.frame_width)
        scale_x = dw / self.frame_width
        scale_y = dh / self.frame_height

        # State indicator
        if self.recording:
            n = len(self.recorded_positions)
            cv2.circle(left_small, (dw - 20, 20), 8, (0, 0, 255), -1)
            cv2.putText(left_small, f"REC #{self.recording_number} [{n}]", (dw - 130, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        elif self.armed:
            cv2.putText(left_small, f"ARMED #{self.recording_number} - toss ball",
                        (dw - 250, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(left_small, "Press 'n' for next", (dw - 180, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Actual detection (green circle)
        if result['found_3d']:
            for det, img in [(result['left_detection'], left_small),
                             (result['right_detection'], right_small)]:
                if det and det['found']:
                    cx = int(det['center'][0] * scale_x)
                    cy = int(det['center'][1] * scale_y)
                    r = int(det['radius'] * scale_x)
                    cv2.circle(img, (cx, cy), r, (0, 255, 0), 2)
                    cv2.circle(img, (cx, cy), 3, (0, 255, 0), -1)

        # Predicted position (magenta diamond) on left frame
        # Project 3D predicted position back to left camera pixel coords
        if self.live_predicted_pos is not None and self.triangulator.P0 is not None:
            px, py, pz, dt = self.live_predicted_pos
            # Project 3D → 2D using left camera projection matrix
            pt_3d = np.array([px, py, pz, 1.0])
            pt_2d = self.triangulator.P0 @ pt_3d
            if abs(pt_2d[2]) > 1e-6:
                px_img = int((pt_2d[0] / pt_2d[2]) * scale_x)
                py_img = int((pt_2d[1] / pt_2d[2]) * scale_y)

                if 0 <= px_img < dw and 0 <= py_img < dh:
                    # Diamond marker
                    pts = np.array([
                        [px_img, py_img - 10],
                        [px_img + 10, py_img],
                        [px_img, py_img + 10],
                        [px_img - 10, py_img]
                    ], np.int32)
                    cv2.polylines(left_small, [pts], True, (255, 0, 255), 2)

                    # Draw error line from predicted to actual
                    if result['left_detection'] and result['left_detection']['found']:
                        act_cx = int(result['left_detection']['center'][0] * scale_x)
                        act_cy = int(result['left_detection']['center'][1] * scale_y)
                        cv2.line(left_small, (px_img, py_img), (act_cx, act_cy),
                                 (255, 255, 255), 1)

        # Prediction error display
        if self.live_prediction_error is not None:
            err = self.live_prediction_error
            color = (0, 255, 0) if err < 2.0 else (0, 255, 255) if err < 5.0 else (0, 0, 255)
            cv2.putText(left_small, f"Pred err: {err:.1f}cm", (10, dh - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Velocity display
        vel = self.predictor.get_velocity()
        if vel['valid']:
            cv2.putText(left_small,
                        f"V: ({vel['vx']:.0f}, {vel['vy']:.0f}, {vel['vz']:.0f}) cm/s",
                        (10, dh - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            cv2.putText(left_small, f"Speed: {vel['speed']:.0f} cm/s",
                        (10, dh - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        # Motion indicator (small dot — green = motion, gray = idle)
        motion_color = (0, 255, 0) if self._motion_detected else (100, 100, 100)
        cv2.circle(left_small, (15, 40), 5, motion_color, -1)

        # Controls hint
        cv2.putText(right_small, "n:next a:analyze r:reset q:quit",
                     (10, dh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    def analyze(self):
        """Post-recording analysis with plots."""
        if len(self.recorded_positions) < 10:
            print(f"\nNot enough data ({len(self.recorded_positions)} frames). Need at least 10.")
            return

        try:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("\nmatplotlib not available. Install: pip install matplotlib")
            self._print_text_analysis()
            return

        positions = np.array(self.recorded_positions)  # (N, 4): x, y, z, t
        pos_xyz = positions[:, :3]
        pos_t = positions[:, 3] - positions[0, 3]  # Normalize to start at 0

        # --- Compute actual velocity (finite differences) ---
        actual_vx, actual_vy, actual_vz = [], [], []
        actual_t = []
        for i in range(1, len(positions)):
            dt = positions[i, 3] - positions[i - 1, 3]
            if dt > 0:
                actual_vx.append((positions[i, 0] - positions[i - 1, 0]) / dt)
                actual_vy.append((positions[i, 1] - positions[i - 1, 1]) / dt)
                actual_vz.append((positions[i, 2] - positions[i - 1, 2]) / dt)
                actual_t.append(pos_t[i])
        actual_vx = np.array(actual_vx)
        actual_vy = np.array(actual_vy)
        actual_vz = np.array(actual_vz)
        actual_t = np.array(actual_t)

        # Estimated velocities
        if self.recorded_velocities:
            est_vel = np.array(self.recorded_velocities)  # vx, vy, vz, speed, t
            est_t = est_vel[:, 4] - positions[0, 3]
        else:
            est_vel = None

        # Prediction errors
        if self.recorded_predictions:
            preds = np.array(self.recorded_predictions)
            pred_err = np.sqrt(
                (preds[:, 0] - preds[:, 3])**2 +
                (preds[:, 1] - preds[:, 4])**2 +
                (preds[:, 2] - preds[:, 5])**2
            )
            pred_t = preds[:, 7] - positions[0, 3]
        else:
            preds = None

        # --- Create plots ---
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle('Velocity & Prediction Validation', fontsize=14, fontweight='bold')

        # 1. Position trajectories (X, Y, Z over time)
        ax = axes[0, 0]
        ax.plot(pos_t, pos_xyz[:, 0], 'r-', label='X', linewidth=1.5)
        ax.plot(pos_t, pos_xyz[:, 1], 'g-', label='Y', linewidth=1.5)
        ax.plot(pos_t, pos_xyz[:, 2], 'b-', label='Z', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (cm)')
        ax.set_title('3D Position Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Velocity: Estimated vs Actual
        ax = axes[0, 1]
        if len(actual_t) > 3:
            # Smooth actual velocity with rolling average for readability
            window = min(5, len(actual_vx) // 2)
            if window >= 2:
                kernel = np.ones(window) / window
                smooth_vx = np.convolve(actual_vx, kernel, mode='valid')
                smooth_vy = np.convolve(actual_vy, kernel, mode='valid')
                smooth_vz = np.convolve(actual_vz, kernel, mode='valid')
                smooth_t = actual_t[window - 1:]
            else:
                smooth_vx, smooth_vy, smooth_vz = actual_vx, actual_vy, actual_vz
                smooth_t = actual_t

            ax.plot(smooth_t, smooth_vx, 'r-', alpha=0.5, linewidth=1, label='Actual Vx')
            ax.plot(smooth_t, smooth_vy, 'g-', alpha=0.5, linewidth=1, label='Actual Vy')
            ax.plot(smooth_t, smooth_vz, 'b-', alpha=0.5, linewidth=1, label='Actual Vz')

        if est_vel is not None and len(est_vel) > 0:
            ax.plot(est_t, est_vel[:, 0], 'r--', linewidth=2, label='Est Vx')
            ax.plot(est_t, est_vel[:, 1], 'g--', linewidth=2, label='Est Vy')
            ax.plot(est_t, est_vel[:, 2], 'b--', linewidth=2, label='Est Vz')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (cm/s)')
        ax.set_title('Velocity: Estimated (dashed) vs Actual (solid)')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        # 3. Speed over time
        ax = axes[1, 0]
        if len(actual_t) > 0:
            actual_speed = np.sqrt(actual_vx**2 + actual_vy**2 + actual_vz**2)
            ax.plot(actual_t, actual_speed, 'k-', alpha=0.4, linewidth=1, label='Actual speed')
        if est_vel is not None and len(est_vel) > 0:
            ax.plot(est_t, est_vel[:, 3], 'b-', linewidth=2, label='Estimated speed')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (cm/s)')
        ax.set_title('Speed Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Prediction error over time
        ax = axes[1, 1]
        if preds is not None and len(preds) > 0:
            ax.plot(pred_t, pred_err, 'r-', linewidth=1.5)
            ax.axhline(y=np.mean(pred_err), color='blue', linestyle='--',
                        label=f'Mean: {np.mean(pred_err):.2f} cm')
            ax.fill_between(pred_t, 0, pred_err, alpha=0.2, color='red')
            ax.set_ylabel('3D Error (cm)')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No prediction data', transform=ax.transAxes,
                    ha='center', fontsize=12)
        ax.set_xlabel('Time (s)')
        ax.set_title(f'Prediction Error ({self.LOOKAHEAD_FRAMES}-frame lookahead)')
        ax.grid(True, alpha=0.3)

        # 5. Predicted vs Actual scatter (per-axis)
        ax = axes[2, 0]
        if preds is not None and len(preds) > 0:
            ax.scatter(preds[:, 3], preds[:, 0], c='red', s=10, alpha=0.6, label='X')
            ax.scatter(preds[:, 4], preds[:, 1], c='green', s=10, alpha=0.6, label='Y')
            ax.scatter(preds[:, 5], preds[:, 2], c='blue', s=10, alpha=0.6, label='Z')
            # Perfect prediction line
            all_vals = np.concatenate([preds[:, 3:6].ravel(), preds[:, 0:3].ravel()])
            lims = [np.min(all_vals), np.max(all_vals)]
            ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.5, label='Perfect')
            ax.legend(fontsize=8)
        ax.set_xlabel('Actual (cm)')
        ax.set_ylabel('Predicted (cm)')
        ax.set_title('Predicted vs Actual Position')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # 6. Summary stats
        ax = axes[2, 1]
        ax.axis('off')

        summary_lines = [
            f"Frames recorded: {len(self.recorded_positions)}",
            f"Duration: {pos_t[-1]:.2f} s",
            f"Avg sample rate: {len(self.recorded_positions) / pos_t[-1]:.1f} fps" if pos_t[-1] > 0 else "",
            "",
        ]

        if est_vel is not None and len(est_vel) > 0:
            summary_lines += [
                f"Avg estimated speed: {np.mean(est_vel[:, 3]):.1f} cm/s",
                f"Max estimated speed: {np.max(est_vel[:, 3]):.1f} cm/s",
                "",
            ]

        if preds is not None and len(preds) > 0:
            summary_lines += [
                f"Prediction error ({self.LOOKAHEAD_FRAMES}-frame):",
                f"  Mean:   {np.mean(pred_err):.2f} cm",
                f"  Median: {np.median(pred_err):.2f} cm",
                f"  Max:    {np.max(pred_err):.2f} cm",
                f"  Std:    {np.std(pred_err):.2f} cm",
                "",
            ]

            # Per-axis errors
            err_x = np.abs(preds[:, 0] - preds[:, 3])
            err_y = np.abs(preds[:, 1] - preds[:, 4])
            err_z = np.abs(preds[:, 2] - preds[:, 5])
            summary_lines += [
                f"Per-axis mean error:",
                f"  X: {np.mean(err_x):.2f} cm",
                f"  Y: {np.mean(err_y):.2f} cm",
                f"  Z: {np.mean(err_z):.2f} cm",
            ]

        summary = "\n".join(summary_lines)
        ax.text(0.05, 0.95, summary, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        # Verdict
        if preds is not None and len(preds) > 0:
            mean_err = np.mean(pred_err)
            if mean_err < 2.0:
                verdict = "EXCELLENT"
                vcolor = '#4CAF50'
            elif mean_err < 5.0:
                verdict = "GOOD"
                vcolor = '#8BC34A'
            elif mean_err < 10.0:
                verdict = "FAIR"
                vcolor = '#FF9800'
            else:
                verdict = "NEEDS WORK"
                vcolor = '#F44336'

            ax.text(0.5, 0.02, verdict, transform=ax.transAxes,
                    fontsize=20, fontweight='bold', color=vcolor, ha='center')

        plt.tight_layout()
        plt.show()

    def _print_text_analysis(self):
        """Fallback text-only analysis."""
        positions = np.array(self.recorded_positions)
        t_span = positions[-1, 3] - positions[0, 3]

        print(f"\n{'=' * 60}")
        print("VALIDATION RESULTS (text mode)")
        print(f"{'=' * 60}")
        print(f"Frames: {len(self.recorded_positions)}")
        print(f"Duration: {t_span:.2f}s")

        if self.recorded_velocities:
            vels = np.array(self.recorded_velocities)
            print(f"\nEstimated speed: avg={np.mean(vels[:, 3]):.1f}, max={np.max(vels[:, 3]):.1f} cm/s")

        if self.recorded_predictions:
            preds = np.array(self.recorded_predictions)
            errs = np.sqrt(
                (preds[:, 0] - preds[:, 3])**2 +
                (preds[:, 1] - preds[:, 4])**2 +
                (preds[:, 2] - preds[:, 5])**2
            )
            print(f"\nPrediction error ({self.LOOKAHEAD_FRAMES}-frame):")
            print(f"  Mean:   {np.mean(errs):.2f} cm")
            print(f"  Median: {np.median(errs):.2f} cm")
            print(f"  Max:    {np.max(errs):.2f} cm")

        print(f"{'=' * 60}")

    def run(self):
        print("\n" + "=" * 60)
        print("VELOCITY & PREDICTION VALIDATION")
        print("=" * 60)

        if not self.check_calibration():
            return

        try:
            self.triangulator = StereoTriangulator(
                calibration_dir=self.calibration_dir,
                cam_left_id=self.cam_left_id,
                cam_right_id=self.cam_right_id)
        except Exception as e:
            print(f"ERROR: {e}")
            return

        self.load_thresholds()

        self.predictor = TrajectoryPredictor(
            buffer_size=10,
            min_points=3,
            velocity_method='regression',
            gravity=981.0,
            y_down=True,
            enable_drag=True
        )

        print(f"\nBaseline: {self.triangulator.get_baseline():.2f} cm")
        print(f"\nCONTROLS:")
        print(f"  n     - Arm next recording")
        print(f"  a     - Analyze last recording (plots)")
        print(f"  r     - Reset everything")
        print(f"  SPACE - Manual start/stop override")
        print(f"  q     - Quit")
        print(f"\nRecording #{self.recording_number} armed. Toss the ball to start capturing.")
        print("=" * 60)

        try:
            self.start_cameras()
        except RuntimeError as e:
            print(f"ERROR: {e}")
            return

        # FPS tracking
        fps_timestamps = []
        fps_display = 0.0

        try:
            while True:
                t_now = time.perf_counter()
                fps_timestamps.append(t_now)
                if len(fps_timestamps) > 30:
                    fps_timestamps.pop(0)
                if len(fps_timestamps) >= 2:
                    elapsed = fps_timestamps[-1] - fps_timestamps[0]
                    if elapsed > 0:
                        fps_display = (len(fps_timestamps) - 1) / elapsed

                result = self.capture_and_detect()
                if result['left_frame'] is None:
                    continue

                # Process frame
                self.process_frame(result)

                # Display
                dw = self.display_width
                dh = int(dw * self.frame_height / self.frame_width)

                left_small = cv2.resize(result['left_frame'], (dw, dh))
                right_small = cv2.resize(result['right_frame'], (dw, dh))

                self.draw_overlay(left_small, right_small, result)

                # FPS overlay
                cv2.putText(left_small, f"{fps_display:.1f} fps", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                combined = cv2.hconcat([left_small, right_small])
                cv2.imshow('Velocity Validation', combined)

                # Console
                if result['found_3d'] and self.predictor.get_velocity()['valid']:
                    x, y, z = result['position_3d']
                    v = self.predictor.get_velocity()
                    err_str = f"err={self.live_prediction_error:.1f}cm" if self.live_prediction_error else "---"
                    print(f"\rPos:({x:5.0f},{y:5.0f},{z:5.0f}) "
                          f"V:({v['vx']:5.0f},{v['vy']:5.0f},{v['vz']:5.0f}) "
                          f"Spd:{v['speed']:4.0f} {err_str}   ", end='')

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('n'):
                    # Save current recording (if any) and arm next
                    if len(self.recorded_positions) > 0:
                        self.all_recordings.append({
                            'number': self.recording_number,
                            'positions': list(self.recorded_positions),
                            'velocities': list(self.recorded_velocities),
                            'predictions': list(self.recorded_predictions),
                        })
                    self.recording = False
                    self.consecutive_losses = 0
                    self.recording_number += 1
                    self.recorded_positions = []
                    self.recorded_velocities = []
                    self.recorded_predictions = []
                    self.live_predicted_pos = None
                    self.live_prediction_error = None
                    self.predictor.reset()
                    self.armed = True
                    print(f"\n[ARMED] Recording #{self.recording_number} armed. Toss the ball.")
                elif key == ord(' '):
                    # Manual override
                    if self.recording:
                        self.recording = False
                        self.armed = False
                        self.consecutive_losses = 0
                        print(f"\n[STOPPED] {len(self.recorded_positions)} frames. Press 'n' for next, 'a' to analyze.")
                    else:
                        self.armed = True
                        print(f"\n[ARMED] Recording #{self.recording_number} armed.")
                elif key == ord('a'):
                    self.analyze()
                elif key == ord('r'):
                    self.reset_recording()
                elif key == ord('p'):
                    vel = self.predictor.get_velocity()
                    stats = self.predictor.get_stats()
                    print(f"\n[STATS] {stats}")
                    if vel['valid']:
                        print(f"  Velocity: ({vel['vx']:.1f}, {vel['vy']:.1f}, {vel['vz']:.1f}) cm/s")
                        print(f"  Speed: {vel['speed']:.1f} cm/s")

        except KeyboardInterrupt:
            pass
        finally:
            self.triangulator.stop_cameras()
            cv2.destroyAllWindows()

        # Auto-analyze on exit if we have data
        if len(self.recorded_positions) > 10:
            print("\n\nFinal analysis:")
            self.analyze()

        print("\nDone!")


if __name__ == '__main__':
    VelocityValidator().run()
