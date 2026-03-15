"""
Simple integration: stereo triangulation + robot-frame prediction + UART.

Pipeline:
  1. Triangulate ball in camera frame (cm) via StereoTriangulator
  2. Transform to robot frame (mm) via cam_to_robot rotation matrix
  3. Buffer robot-frame positions, estimate velocity via regression
  4. Predict trajectory with gravity + air drag (Euler integration)
  5. Find first point entering workspace, or clamp nearest point if none found
  6. Send (x, y, z, time) to robot via UART

v5: firmware-matching ellipse workspace, clamp-to-workspace fallback,
    simplified visualization.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from comm_function.points_based_transform import load_points_based_transform
from comm_function.transmit_over_uart import UartComm
from config.camera_config import load_camera_settings
from tracking.stereo_triangulator import StereoTriangulator


def _print(*args, **kwargs):
    print("from planner in terminal", *args, **kwargs)


# Workspace -- firmware ellipse with 5% safety margin to avoid IK rejections
ELLIPSE_A    = 790.0 * 0.95   # mm X semi-axis (750.5)
ELLIPSE_B    = 540.0 * 0.95   # mm Y semi-axis (513.0)
Z_MIN        = -1050.0 + 25   # mm (-1025, 25mm margin from robot.h limit)
Z_MAX        = -721.0  - 10   # mm (-731, 10mm margin)
MAX_CLAMP_DIST = 350.0        # mm -- max distance to clamp to workspace
ROBOT_HOME   = (0.0, 0.0, -900.0)
MAX_CART_VEL = 4000.0   # mm/s
MAX_CART_ACC = 20000.0  # mm/s^2

CM_TO_MM = 10.0
GRAVITY_Z = -9810.0  # mm/s^2, robot Z is vertical, negative = down

# Air drag for ping pong ball (mass=2.7g, diameter=40mm, Cd=0.40)
# k = 0.5 * Cd * rho_air * A / m  (in mm^-1)
#   = 0.5 * 0.40 * 1.2e-9 * pi*20^2 / 0.0027 = 1.12e-4
DRAG_K = 0.000112  # mm^-1 -- drag deceleration: a = -DRAG_K * |v| * v


def cam_to_robot(R, t, scale, cam_x, cam_y, cam_z):
    """Camera coords (cm) -> robot coords (mm) using points-based transform."""
    p = R @ (np.array([cam_x, cam_y, cam_z]) * scale) + t
    return float(p[0]), float(p[1]), float(p[2])


# ================================================================
# ROBOT-FRAME PREDICTOR
# ================================================================

class RobotPredictor:
    """
    Trajectory predictor in robot frame (mm).
    Uses Euler integration with gravity + air drag.
    """

    BUFFER_SIZE    = 15
    MIN_POINTS     = 6      # minimum for regression
    MIN_TIME_SPAN  = 0.08   # seconds -- require 80ms of data (~8 frames)
    MAX_SPEED      = 15000.0 # mm/s
    MAX_JUMP       = 400.0   # mm
    GAP_RESET      = 0.12   # seconds
    STALE_TIMEOUT  = 0.15   # seconds

    # Prediction scan
    SCAN_DURATION  = 1.5    # seconds forward
    SCAN_DT        = 0.005  # 5ms steps
    MIN_TIME_HIT   = 0.10   # minimum reaction time for robot

    # Proximity filter
    MAX_PREDICT_Y  = 1400.0  # mm -- ball must be closer than this

    # Direction filter
    MIN_APPROACH_VY = -200.0
    APPROACH_Y_THRESHOLD = 600.0

    def __init__(self):
        self.positions = deque(maxlen=self.BUFFER_SIZE)
        self.velocity = None
        self._rejected = 0
        self._accepted = 0
        self._last_reject_reason = None

    def reset(self):
        self.positions.clear()
        self.velocity = None

    def add_position(self, x, y, z, t):
        """Add a robot-frame position (mm). Returns True if accepted."""
        if self.positions:
            last = self.positions[-1]
            dt = t - last[3]
            if dt <= 0:
                self._rejected += 1
                self._last_reject_reason = "zero_dt"
                return False
            if dt > self.GAP_RESET:
                self._last_reject_reason = f"gap({dt*1000:.0f}ms)"
                self.reset()
            else:
                dx, dy, dz = x - last[0], y - last[1], z - last[2]
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                if dist > self.MAX_JUMP:
                    self._rejected += 1
                    self._last_reject_reason = f"jump({dist:.0f}mm)"
                    return False
                speed = dist / dt
                if speed > self.MAX_SPEED:
                    self._rejected += 1
                    self._last_reject_reason = f"speed({speed:.0f})"
                    return False

        self.positions.append((x, y, z, t))
        self._accepted += 1
        self._last_reject_reason = None

        if len(self.positions) >= self.MIN_POINTS:
            span = self.positions[-1][3] - self.positions[0][3]
            if span >= self.MIN_TIME_SPAN:
                self._estimate_velocity()

        return True

    def _estimate_velocity(self):
        """Estimate velocity via least-squares regression."""
        n = len(self.positions)
        pts = list(self.positions)

        t_ref = pts[-1][3]
        dt = np.array([p[3] - t_ref for p in pts])
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        zs = np.array([p[2] for p in pts])

        A = np.column_stack([dt, np.ones(n)])

        vx = float(np.linalg.lstsq(A, xs, rcond=None)[0][0])
        vy = float(np.linalg.lstsq(A, ys, rcond=None)[0][0])

        # Z with gravity: z - 0.5*g*dt^2 = z0 + vz*dt
        zs_corrected = zs - 0.5 * GRAVITY_Z * dt * dt
        vz = float(np.linalg.lstsq(A, zs_corrected, rcond=None)[0][0])

        speed = math.sqrt(vx*vx + vy*vy + vz*vz)
        if speed > self.MAX_SPEED:
            self.velocity = None
            return

        self.velocity = (vx, vy, vz)

    def is_ready(self):
        if self.velocity is None:
            return False
        if not self.positions:
            return False
        age = time.perf_counter() - self.positions[-1][3]
        return age < self.STALE_TIMEOUT

    def _ball_approaching(self):
        """Check if ball is moving toward the workspace."""
        if not self.positions or self.velocity is None:
            return False
        y_now = self.positions[-1][1]
        vy = self.velocity[1]
        if abs(y_now) < self.APPROACH_Y_THRESHOLD:
            return True
        return vy < self.MIN_APPROACH_VY

    @staticmethod
    def _step_euler(x, y, z, vx, vy, vz, dt):
        """One Euler step with gravity + air drag."""
        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        drag = DRAG_K * speed

        ax = -drag * vx
        ay = -drag * vy
        az = GRAVITY_Z - drag * vz

        vx += ax * dt
        vy += ay * dt
        vz += az * dt

        x += vx * dt
        y += vy * dt
        z += vz * dt

        return x, y, z, vx, vy, vz

    def predict_intercept(self, robot_pos=None):
        """
        Find the first future point where ball enters workspace.
        If none found, clamp the nearest trajectory point to workspace.
        """
        if not self.is_ready():
            return None

        if not self._ball_approaching():
            return None

        # Proximity filter
        y_now = self.positions[-1][1]
        if y_now > self.MAX_PREDICT_Y:
            return None

        x, y, z, _ = self.positions[-1]
        vx, vy, vz = self.velocity

        rp = robot_pos if robot_pos is not None else ROBOT_HOME

        # Track closest-to-workspace point for fallback
        best_clamp = None
        best_clamp_dist = float('inf')

        t = 0.0
        step = self.SCAN_DT
        while t <= self.SCAN_DURATION:
            if t >= self.MIN_TIME_HIT:
                if in_workspace(x, y, z):
                    return {
                        'x': x, 'y': y, 'z': z,
                        'time': t,
                        'vx': vx, 'vy': vy, 'vz': vz,
                        'clamped': False,
                    }

                # Track closest point for fallback
                xc, yc, zc, cdist = clamp_to_workspace(x, y, z)
                if cdist < best_clamp_dist:
                    best_clamp_dist = cdist
                    best_clamp = {
                        'x': xc, 'y': yc, 'z': zc,
                        'time': t,
                        'vx': vx, 'vy': vy, 'vz': vz,
                        'clamped': True,
                        'clamp_dist': cdist,
                    }

            x, y, z, vx, vy, vz = self._step_euler(x, y, z, vx, vy, vz, step)
            t += step

        # Fallback: clamp nearest point if within MAX_CLAMP_DIST
        if best_clamp is not None and best_clamp_dist <= MAX_CLAMP_DIST:
            return best_clamp

        return None

    def get_current_position(self):
        if not self.positions:
            return None
        p = self.positions[-1]
        return (p[0], p[1], p[2])

    def get_stats(self):
        y_now = self.positions[-1][1] if self.positions else 9999
        return {
            'buffer': len(self.positions),
            'accepted': self._accepted,
            'rejected': self._rejected,
            'has_vel': self.velocity is not None,
            'approaching': self._ball_approaching() if self.velocity else False,
            'close_enough': y_now <= self.MAX_PREDICT_Y,
        }


# ================================================================
# WORKSPACE CHECKS
# ================================================================

def in_workspace(x, y, z):
    """Firmware-matching ellipse check."""
    return (Z_MIN <= z <= Z_MAX and (x / ELLIPSE_A) ** 2 + (y / ELLIPSE_B) ** 2 <= 1.0)


def clamp_to_workspace(x, y, z):
    """Clamp a point to the nearest workspace boundary. Returns (x, y, z, dist)."""
    # Clamp Z
    z_c = max(Z_MIN, min(Z_MAX, z))
    # Clamp XY to ellipse
    r = math.sqrt((x / ELLIPSE_A) ** 2 + (y / ELLIPSE_B) ** 2)
    if r > 1.0:
        x_c = x / r
        y_c = y / r
    else:
        x_c, y_c = x, y
    dist = math.sqrt((x - x_c) ** 2 + (y - y_c) ** 2 + (z - z_c) ** 2)
    return x_c, y_c, z_c, dist


# ================================================================
# MAIN APPLICATION
# ================================================================

class SimpleIntegration:
    def __init__(
        self,
        uart_port: str,
        baud_rate: int = 115200,
        home_ack_timeout_s: Optional[float] = 30.0,
        uart_verbose: bool = True,
        tx_interval_s: float = 0.03,
        warmup_s: float = 2.0,
    ):
        self.script_dir = SCRIPT_DIR
        self.base_dir = PARENT_DIR
        self.calibration_dir = os.path.join(
            self.base_dir, "camera_calibration", "camera_parameters"
        )

        cam = load_camera_settings()
        self.frame_width = cam["frame_width"]
        self.frame_height = cam["frame_height"]
        self.cam_left_id = cam["camera0"]
        self.cam_right_id = cam["camera1"]

        self.triangulator: Optional[StereoTriangulator] = None
        self.predictor = RobotPredictor()
        self.uart = UartComm(port=uart_port, baud_rate=baud_rate, verbose=uart_verbose)

        # Camera->robot transform (points-based)
        tf = load_points_based_transform()
        self.R = tf["rotation"]
        self.t_vec = tf["translation"]
        self.cam_scale = tf["camera_scale_to_robot_units"]
        _print(f"Loaded points-based transform (scale={self.cam_scale})")

        self.robot_homed = False
        self.run_gate = False
        self.intercept_sent = True
        self.shutdown_requested = False
        self.shutdown_home_sent = False

        self.home_ack_timeout_s = home_ack_timeout_s
        self.tx_interval_s = tx_interval_s
        self.warmup_s = max(0.0, warmup_s)
        self.last_tx_time = 0.0
        self.last_cmd = None

        # Track robot position for reachability
        self.robot_current_pos = ROBOT_HOME

        # State machine for UART flow:
        #   None       = idle, no pending action
        #   'intercept' = sent intercept, waiting for COMPLETED Q
        #   'homing'    = sent HOME after intercept, waiting for COMPLETED Q
        self._pending_action = None
        self._stm32_moving = False  # True once STATE: MOVE received

        # Throw counter for logging
        self.throw_count = 0
        self._update_count = 0

        # Intercept log — saved to JSON on each throw
        self._intercept_log: list[dict] = []
        self._intercept_log_path = os.path.join(
            SCRIPT_DIR, "intercept_log.json"
        )

    def _log_intercept(self, intercept: dict, t_adjusted: float, n_pts: int, latency: float) -> None:
        """Append intercept to log and save to JSON file."""
        pos = self.predictor.get_current_position()
        entry = {
            "throw": self.throw_count,
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "target_x_mm": round(intercept["x"], 1),
            "target_y_mm": round(intercept["y"], 1),
            "target_z_mm": round(intercept["z"], 1),
            "time_to_intercept_ms": round(t_adjusted * 1000, 1),
            "clamped": intercept.get("clamped", False),
            "clamp_dist_mm": round(intercept.get("clamp_dist", 0), 1),
            "ball_x_mm": round(pos[0], 1) if pos else None,
            "ball_y_mm": round(pos[1], 1) if pos else None,
            "ball_z_mm": round(pos[2], 1) if pos else None,
            "vel_x_mm_s": round(self.predictor.velocity[0], 1) if self.predictor.velocity else None,
            "vel_y_mm_s": round(self.predictor.velocity[1], 1) if self.predictor.velocity else None,
            "vel_z_mm_s": round(self.predictor.velocity[2], 1) if self.predictor.velocity else None,
            "buffer_points": n_pts,
            "latency_ms": round(latency * 1000, 1),
        }
        self._intercept_log.append(entry)
        try:
            with open(self._intercept_log_path, "w", encoding="utf-8") as f:
                json.dump(self._intercept_log, f, indent=2)
        except Exception as e:
            _print(f"[WARN] Failed to save intercept log: {e}")

    # --- UART RX processing ---

    def process_uart_rx(self):
        """
        Parse UART responses from STM32.

        State machine:
          intercept sent -> _pending_action='intercept'
          STATE: MOVE -> _stm32_moving=True (stop sending updates)
          COMPLETED Q (intercept) -> send HOME -> _pending_action='homing'
          COMPLETED Q (homing) -> _pending_action=None, auto-clear, ready
          COMPLETED Q (None) -> ignore (startup noise)
        """
        for line in self.uart.poll_status_lines():
            print(f"[UART][RX] {line}")
            upper = line.upper()

            if "STATE: MOVE" in upper:
                self._stm32_moving = True

            if "COMPLETED Q" in upper:
                if self._pending_action == 'intercept':
                    _print("[AUTO] Intercept done. Sending HOME...")
                    self._pending_action = 'homing'
                    self._stm32_moving = False
                    try:
                        if self.uart.is_open and not self.shutdown_requested:
                            self.uart.send_home()
                    except Exception:
                        pass

                elif self._pending_action == 'homing':
                    self._pending_action = None
                    self._stm32_moving = False
                    self.robot_current_pos = ROBOT_HOME
                    self.predictor.reset()
                    self.intercept_sent = False
                    self._update_count = 0
                    _print("[AUTO] Home done. Ready for next throw.")

            elif "TARGET OUT OF WORKSPACE" in upper:
                _print("[WARN] STM32 rejected target (IK workspace). Auto-clearing.")
                self._pending_action = None
                self._stm32_moving = False
                self.intercept_sent = False
                self.predictor.reset()
                self.robot_current_pos = ROBOT_HOME
                self._update_count = 0

            elif "ROBOT WILL BE LATE" in upper:
                _print("[WARN] Robot will be late.")

            elif "PLANNING FAILED" in upper or "PLAN_ABORT" in upper:
                _print("[WARN] STM32 planning failed. Auto-clearing.")
                self._pending_action = None
                self._stm32_moving = False
                self.intercept_sent = False
                self.predictor.reset()
                self._update_count = 0

            elif "STATE: PLAN" in upper and not self.robot_homed:
                self.robot_homed = True

    # --- Startup ---

    def check_calibration(self) -> bool:
        for f in ["camera0_intrinsics.dat", "camera1_intrinsics.dat",
                   "camera0_rot_trans.dat", "camera1_rot_trans.dat"]:
            if not os.path.exists(os.path.join(self.calibration_dir, f)):
                _print(f"ERROR: Missing {f}")
                return False
        return True

    def init_triangulator(self) -> bool:
        try:
            self.triangulator = StereoTriangulator(
                calibration_dir=self.calibration_dir,
                cam_left_id=self.cam_left_id,
                cam_right_id=self.cam_right_id,
            )
            return True
        except Exception as e:
            _print(f"ERROR init triangulator: {e}")
            return False

    def send_home_and_wait(self) -> bool:
        try:
            self.uart.open()
            self.uart.clear_input_buffer()
            self.uart.send_home()
            ack = self.uart.wait_for_home_confirmation(timeout_s=self.home_ack_timeout_s)
        except Exception as e:
            _print(f"[UART] ERROR: {e}")
            return False
        if ack is None:
            _print("[UART] Home ACK timeout")
            return False
        self.robot_homed = True
        self.intercept_sent = True
        self.robot_current_pos = ROBOT_HOME
        _print("Robot homed. Press 'g' to start tracking.")
        return True

    def request_shutdown_home(self):
        self.shutdown_requested = True
        self.run_gate = False
        self.intercept_sent = True
        if self.shutdown_home_sent:
            return
        try:
            if self.uart.is_open:
                self.uart.send_home()
                self.shutdown_home_sent = True
        except Exception:
            pass

    def warmup_background(self) -> bool:
        if self.warmup_s <= 0:
            return True
        _print("Remove ball. Learning background (SPACE=skip)...")
        t0 = time.time()
        while time.time() - t0 < self.warmup_s:
            if not self.triangulator.cap_left.grab():
                continue
            if not self.triangulator.cap_right.grab():
                continue
            _, fl = self.triangulator.cap_left.retrieve()
            _, fr = self.triangulator.cap_right.retrieve()
            if fl is None or fr is None:
                continue
            self.triangulator.build_background(fl, fr)
            vis = cv2.resize(fl, (640, int(640 * self.frame_height / self.frame_width)))
            progress = min((time.time() - t0) / self.warmup_s, 1.0)
            h = vis.shape[0]
            bw = int(progress * (vis.shape[1] - 40))
            cv2.rectangle(vis, (20, h - 30), (20 + bw, h - 15), (0, 255, 255), -1)
            cv2.putText(vis, f"BG: {progress*100:.0f}%", (20, h - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.imshow("Simple Integration", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                break
            if key == ord("q"):
                return False
        _print("Background ready.")
        return True

    # --- UART send ---

    def maybe_send(self, intercept, frame_ts):
        if (not self.robot_homed or not self.run_gate or
                self.shutdown_requested):
            return
        if intercept is None:
            return

        # Allow updates: first send OR update while robot still in PLAN phase
        is_update = False
        if self.intercept_sent:
            if self._stm32_moving or self._pending_action != 'intercept':
                return
            # Only update if prediction changed significantly (>80mm)
            if self.last_cmd is not None:
                dx = intercept['x'] - self.last_cmd['x']
                dy = intercept['y'] - self.last_cmd['y']
                dz = intercept['z'] - self.last_cmd['z']
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                if dist < 80.0:
                    return
            is_update = True

        x, y, z = intercept['x'], intercept['y'], intercept['z']

        now = time.perf_counter()
        if (now - self.last_tx_time) < self.tx_interval_s:
            return

        time_sent = time.perf_counter()
        latency = max(0.0, time_sent - frame_ts)
        t_adjusted = max(0.0, intercept['time'] - latency)

        n_pts = len(self.predictor.positions)

        try:
            self.uart.send_intercept(
                x_mm=x, y_mm=y, z_mm=z,
                intercept_time_s=t_adjusted,
                time_sent_s=time_sent,
                timestamp_s=frame_ts,
            )
            self.last_tx_time = time_sent
            self.last_cmd = intercept
            self.robot_current_pos = (x, y, z)

            if is_update:
                self._update_count += 1
                _print(f"[UPDATE #{self._update_count}] Target(mm): "
                       f"x={x:+.0f} y={y:+.0f} z={z:+.0f}  "
                       f"t={t_adjusted*1000:.0f}ms  pts={n_pts}")
            else:
                self.intercept_sent = True
                self._pending_action = 'intercept'
                self._stm32_moving = False
                self.throw_count += 1

                clamped = intercept.get('clamped', False)
                tag = " [CLAMPED]" if clamped else ""
                _print(f"[THROW #{self.throw_count}]{tag}  Target(mm): x={x:+.0f} y={y:+.0f} z={z:+.0f}  t={t_adjusted*1000:.0f}ms")
                self._log_intercept(intercept, t_adjusted, n_pts, latency)
        except Exception as e:
            _print(f"[UART] Send failed: {e}")

    # --- Visualization ---

    def draw_intercept_marker(self, frame, intercept):
        """Draw X marker at intercept point on camera image."""
        if intercept is None:
            return
        R_inv = self.R.T
        p_robot = np.array([intercept['x'], intercept['y'], intercept['z']])
        p_cam_cm = R_inv @ (p_robot - self.t_vec) / self.cam_scale
        uv = self.triangulator.project_to_image(
            (float(p_cam_cm[0]), float(p_cam_cm[1]), float(p_cam_cm[2])),
            camera="left")
        if uv is None:
            return
        px, py = int(round(float(uv[0]))), int(round(float(uv[1])))
        cv2.line(frame, (px-12, py-12), (px+12, py+12), (0, 255, 255), 2)
        cv2.line(frame, (px-12, py+12), (px+12, py-12), (0, 255, 255), 2)
        cv2.circle(frame, (px, py), 16, (0, 255, 255), 2)
        cv2.putText(frame, f"t={intercept['time']*1000:.0f}ms",
                    (px+20, py-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    # --- Main loop ---

    def run(self):
        _print("\n" + "=" * 60)
        _print(" SIMPLE INTEGRATION v5")
        _print("=" * 60)
        _print(f"Cameras: L={self.cam_left_id} R={self.cam_right_id}")
        _print(f"Workspace: ellipse {ELLIPSE_A:.0f}x{ELLIPSE_B:.0f}mm, "
               f"Z=[{Z_MIN:.0f}, {Z_MAX:.0f}]  Drag={DRAG_K:.6f}")
        _print(f"Min points: {RobotPredictor.MIN_POINTS}, "
               f"Min time: {RobotPredictor.MIN_TIME_SPAN*1000:.0f}ms, "
               f"Gap reset: {RobotPredictor.GAP_RESET*1000:.0f}ms, "
               f"Max predict Y: {RobotPredictor.MAX_PREDICT_Y:.0f}mm")
        _print("")
        _print("HOW TO THROW:")
        _print("  - Throw ball in a gentle arc TOWARD the robot")
        _print("  - Aim for the area below the robot base plate")
        _print("  - Ball should arc DOWN into the workspace zone")
        _print("  - Gentle underhand toss works best")
        _print("")
        _print("Controls: g=gate  c=clear  r=reset  b=bg  q=quit")
        _print("  Auto-clear after each robot move -- no 'c' needed!")
        _print("=" * 60)

        if not self.send_home_and_wait():
            self.uart.close()
            return

        if not self.check_calibration():
            self.request_shutdown_home()
            self.uart.close()
            return

        if not self.init_triangulator():
            self.request_shutdown_home()
            self.uart.close()
            return

        try:
            self.triangulator.start_cameras(self.frame_width, self.frame_height)
        except Exception as e:
            _print(f"ERROR starting cameras: {e}")
            self.request_shutdown_home()
            self.uart.close()
            return

        if not self.warmup_background():
            self.request_shutdown_home()
            self.triangulator.stop_cameras()
            self.uart.close()
            cv2.destroyAllWindows()
            return

        _print("--- LIVE ---  Gate OFF. Press 'g' to start tracking.")

        fps_time = time.time()
        fps = 0.0
        frame_count = 0

        try:
            while True:
                self.process_uart_rx()

                result = self.triangulator.update()
                frame_ts = time.perf_counter()
                if result["left_frame"] is None:
                    continue

                frame_count += 1
                if frame_count % 30 == 0:
                    fps = 30.0 / max(1e-6, time.time() - fps_time)
                    fps_time = time.time()

                robot_pos = None
                intercept = None

                if self.run_gate and result["found_3d"]:
                    cx, cy, cz = result["position_3d"]
                    rx, ry, rz = cam_to_robot(self.R, self.t_vec, self.cam_scale, cx, cy, cz)
                    robot_pos = (rx, ry, rz)
                    self.predictor.add_position(rx, ry, rz, frame_ts)

                if self.run_gate and self.predictor.is_ready():
                    intercept = self.predictor.predict_intercept(
                        robot_pos=self.robot_current_pos)
                    if intercept is not None and result["found_3d"]:
                        self.maybe_send(intercept, frame_ts)

                # --- Visualization ---
                left_vis, right_vis = self.triangulator.draw_results(result)

                if self.intercept_sent and self.last_cmd is not None:
                    self.draw_intercept_marker(left_vis, self.last_cmd)

                # Overlay text
                gate_str = "ON" if self.run_gate else "OFF"
                if self._pending_action == 'intercept':
                    tx_str = "MOVING"
                elif self._pending_action == 'homing':
                    tx_str = "HOMING"
                elif self.intercept_sent:
                    tx_str = "SENT"
                else:
                    tx_str = "READY"
                stats = self.predictor.get_stats()
                appr = "Y" if stats.get('approaching') else "N"
                cv2.putText(left_vis,
                    f"FPS:{fps:.0f}  Buf:{stats['buffer']}  Gate:{gate_str}  TX:{tx_str}  Appr:{appr}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

                if robot_pos is not None:
                    cv2.putText(left_vis,
                        f"Robot(mm): X={robot_pos[0]:+.0f} Y={robot_pos[1]:+.0f} Z={robot_pos[2]:+.0f}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)

                if intercept is not None:
                    ws = "IN" if in_workspace(intercept['x'], intercept['y'], intercept['z']) else "OUT"
                    cv2.putText(left_vis,
                        f"Int(mm): X={intercept['x']:+.0f} Y={intercept['y']:+.0f} "
                        f"Z={intercept['z']:+.0f}  t={intercept['time']*1000:.0f}ms  WS:{ws}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 220), 1)

                cv2.putText(left_vis,
                    f"Throws:{self.throw_count}  g=gate c=clear r=reset b=bg q=quit",
                    (10, left_vis.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120, 120, 120), 1)

                # Terminal output — minimal, only intercept is printed in maybe_send()

                # Show
                dw = 640
                dh = int(dw * self.frame_height / self.frame_width)
                left_s = cv2.resize(left_vis, (dw, dh))
                right_s = cv2.resize(right_vis, (dw, dh))
                cv2.imshow("Simple Integration", cv2.hconcat([left_s, right_s]))

                # Key handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    _print("[QUIT] Sending home...")
                    self.request_shutdown_home()
                    break
                elif key == ord("g"):
                    self.run_gate = not self.run_gate
                    if self.run_gate:
                        self.predictor.reset()
                        self.intercept_sent = False
                        _print("[GATE] ON -- tracking active, ready to send")
                    else:
                        self.intercept_sent = True
                        _print("[GATE] OFF")
                elif key == ord("c"):
                    self.intercept_sent = False
                    self.predictor.reset()
                    _print("[CLEAR] Manual clear -- ready for next intercept")
                elif key == ord("r"):
                    self.predictor.reset()
                    self.intercept_sent = False
                    _print("[RESET] Predictor reset")
                elif key == ord("b"):
                    self.predictor.reset()
                    self.intercept_sent = False
                    _print("[BG RESET] Re-learning...")
                    if not self.warmup_background():
                        break
                elif key == ord("h"):
                    try:
                        self.uart.send_home()
                        self.robot_current_pos = ROBOT_HOME
                        _print("[HOME] Manual home sent")
                    except Exception:
                        pass

        except KeyboardInterrupt:
            _print("[CTRL-C] Sending home...")
            self.request_shutdown_home()
        finally:
            self.request_shutdown_home()
            if self.triangulator is not None:
                try:
                    self.triangulator.stop_cameras()
                except Exception:
                    pass
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            self.uart.close()

        _print(f"Done! Total throws sent: {self.throw_count}")


def main():
    parser = argparse.ArgumentParser(description="Simple stereo + robot-frame prediction + UART v5")
    parser.add_argument("--port", default=os.environ.get("STM32_UART_PORT"),
                        help="UART port (or set STM32_UART_PORT)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--home-ack-timeout", type=float, default=30.0,
                        help="Seconds to wait for home ACK (0=infinite)")
    parser.add_argument("--tx-interval-ms", type=float, default=30.0)
    parser.add_argument("--warmup-s", type=float, default=2.0)
    parser.add_argument("--quiet-uart", action="store_true")
    args = parser.parse_args()

    if not args.port:
        parser.error("UART port required. Pass --port or set STM32_UART_PORT.")

    timeout = None if args.home_ack_timeout == 0.0 else args.home_ack_timeout

    app = SimpleIntegration(
        uart_port=args.port,
        baud_rate=args.baud,
        home_ack_timeout_s=timeout,
        uart_verbose=not args.quiet_uart,
        tx_interval_s=max(0.0, args.tx_interval_ms / 1000.0),
        warmup_s=args.warmup_s,
    )
    app.run()


if __name__ == "__main__":
    raise SystemExit(main())
