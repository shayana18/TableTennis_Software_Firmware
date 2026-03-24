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
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from comm_function.points_based_transform import (
    load_points_based_transform,
    cam_to_robot,
)
from comm_function.transmit_over_uart import UartComm
from config.camera_config import load_camera_settings
from tracking.stereo_triangulator import StereoTriangulator
from trajectory.robot_predictor import RobotPredictor
from trajectory.workspace import (
    ELLIPSE_A, ELLIPSE_B, Z_MIN, Z_MAX,
    ROBOT_HOME, DRAG_K,
    in_workspace,
)


def _print(*args, **kwargs):
    print("from planner in terminal", *args, **kwargs)


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

        # === Throw data logger (detailed per-frame diagnostics) ===
        self._throw_log_path = os.path.join(SCRIPT_DIR, "throw_data_log.json")
        self._throw_log: list[dict] = []      # completed throws
        self._active_throw: Optional[dict] = None
        self._throw_lost_count = 0
        self._THROW_LOST_FRAMES = 30          # ~300ms at 100fps
        self._session_start = datetime.now().isoformat(timespec="milliseconds")
        self._prev_throw_frame_ts: Optional[float] = None


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

    # --- Throw data logger ---

    def _start_throw_log(self, frame_ts: float) -> None:
        """Begin recording a new throw."""
        throw_id = len(self._throw_log) + 1
        self._active_throw = {
            "id": throw_id,
            "start_wall": datetime.now().isoformat(timespec="milliseconds"),
            "start_ts": round(frame_ts, 6),
            "frames": [],
            "sends": [],
            "end_reason": None,
        }
        self._throw_lost_count = 0
        self._prev_throw_frame_ts = None


    def _end_throw_log(self, reason: str) -> None:
        """Finalize the active throw and save."""
        if self._active_throw is None:
            return
        throw = self._active_throw
        frames = throw["frames"]
        throw["end_reason"] = reason

        if len(frames) < 2:
            self._active_throw = None
            return

        accepted = [f for f in frames if f.get("ok")]
        throw["summary"] = {
            "n_frames": len(frames),
            "n_accepted": len(accepted),
            "n_rejected": sum(1 for f in frames if f.get("ok") is False),
            "n_no3d": sum(1 for f in frames if f.get("cam") is None),
            "duration_s": round(frames[-1]["t"] - frames[0]["t"], 4),
            "n_sends": len(throw["sends"]),
            "kf_enabled": self.predictor.get_stats().get("kf_enabled", False),
        }

        self._throw_log.append(throw)
        self._active_throw = None
        self._save_throw_log()

    def _log_throw_frame(
        self,
        frame_ts: float,
        result: dict,
        robot_pos: Optional[tuple],
        accepted: Optional[bool],
        intercept: Optional[dict],
    ) -> None:
        """Record one frame in the active throw."""
        if not self.run_gate:
            return

        found_3d = result["found_3d"]
        has_det = (result.get("left_detection") is not None or
                   result.get("right_detection") is not None)

        # Start throw on first accepted 3D detection
        if found_3d and accepted and self._active_throw is None:
            self._start_throw_log(frame_ts)

        if self._active_throw is None:
            return

        # Lost counter
        if not found_3d:
            self._throw_lost_count += 1
            if self._throw_lost_count >= self._THROW_LOST_FRAMES:
                self._end_throw_log("lost_detection")
                return
            # Only log no-3D frames that had at least a 2D detection
            if not has_det:
                return

        else:
            self._throw_lost_count = 0

        # dt from previous accepted frame
        dt_ms = None
        if found_3d and accepted and self._prev_throw_frame_ts is not None:
            dt_ms = round((frame_ts - self._prev_throw_frame_ts) * 1000, 2)
        if found_3d and accepted:
            self._prev_throw_frame_ts = frame_ts

        # Gather KF state
        vel = self.predictor.velocity
        stats = self.predictor.get_stats()
        kf_pos = None
        if self.predictor.state_estimator is not None:
            p = self.predictor.state_estimator.get_position()
            if p is not None:
                kf_pos = [round(p[0], 1), round(p[1], 1), round(p[2], 1)]

        # Build frame entry
        entry: dict = {"t": round(frame_ts, 6)}
        if dt_ms is not None:
            entry["dt"] = dt_ms

        if found_3d:
            cx, cy, cz = result["position_3d"]
            entry["cam"] = [round(cx, 2), round(cy, 2), round(cz, 2)]
        if robot_pos is not None:
            entry["rob"] = [round(robot_pos[0], 1), round(robot_pos[1], 1),
                            round(robot_pos[2], 1)]
        if accepted is not None:
            entry["ok"] = accepted
        if accepted is False:
            entry["rej"] = self.predictor._last_reject_reason

        if not found_3d and has_det:
            entry["stereo_fail"] = result.get("reject_reason", "no_match")

        if vel is not None:
            entry["vel"] = [round(vel[0], 1), round(vel[1], 1),
                            round(vel[2], 1)]
        if kf_pos is not None:
            entry["kf_pos"] = kf_pos

        entry["buf"] = stats["buffer"]
        entry["kf_rdy"] = stats["kf_ready"]
        entry["rdy"] = self.predictor.is_ready()

        disp = result.get("disparity")
        reproj = result.get("reproj_err")
        if disp is not None:
            entry["disp"] = round(disp, 1)
        if reproj is not None:
            entry["rep"] = round(reproj, 1)

        if stats.get("bounces", 0) > 0:
            entry["bnc"] = stats["bounces"]

        if intercept is not None:
            entry["pred"] = {
                "x": round(intercept["x"], 1),
                "y": round(intercept["y"], 1),
                "z": round(intercept["z"], 1),
                "t": round(intercept["time"], 4),
                "vx": round(intercept.get("vx", 0), 1),
                "vy": round(intercept.get("vy", 0), 1),
                "vz": round(intercept.get("vz", 0), 1),
                "clamp": intercept.get("clamped", False),
            }
            if intercept.get("clamped"):
                entry["pred"]["clamp_d"] = round(
                    intercept.get("clamp_dist", 0), 1)

        self._active_throw["frames"].append(entry)

    def _log_throw_send(
        self, intercept: dict, t_adjusted: float,
        n_pts: int, latency: float, is_update: bool, frame_ts: float,
    ) -> None:
        """Record a UART send event in the active throw."""
        if self._active_throw is None:
            return
        # Snapshot the predictor positions used for this send
        positions_used = [
            [round(p[0], 1), round(p[1], 1), round(p[2], 1), round(p[3], 6)]
            for p in self.predictor.positions
        ]

        self._active_throw["sends"].append({
            "t": round(frame_ts, 6),
            "target": {
                "x": round(intercept["x"], 1),
                "y": round(intercept["y"], 1),
                "z": round(intercept["z"], 1),
            },
            "t_intercept": round(t_adjusted, 4),
            "vel": [
                round(intercept.get("vx", 0), 1),
                round(intercept.get("vy", 0), 1),
                round(intercept.get("vz", 0), 1),
            ],
            "latency_ms": round(latency * 1000, 2),
            "buf_pts": n_pts,
            "is_update": is_update,
            "clamped": intercept.get("clamped", False),
            "positions_used": positions_used,
        })

    def _save_throw_log(self) -> None:
        """Write all completed throws to JSON."""
        from trajectory.workspace import GRAVITY_Z as _GZ, DRAG_K as _DK
        payload = {
            "session_start": self._session_start,
            "config": {
                "cam_scale": self.cam_scale,
                "kf_enabled": self.predictor._using_state_estimator,
                "min_points": RobotPredictor.MIN_POINTS,
                "min_time_span_s": RobotPredictor.MIN_TIME_SPAN,
                "max_speed_mm_s": RobotPredictor.MAX_SPEED,
                "max_jump_mm": RobotPredictor.MAX_JUMP,
                "gap_reset_s": RobotPredictor.GAP_RESET,
                "stale_timeout_s": RobotPredictor.STALE_TIMEOUT,
                "drag_k": _DK,
                "gravity_z_mm_s2": _GZ,
                "max_predict_y_mm": RobotPredictor.MAX_PREDICT_Y,
                "scan_duration_s": RobotPredictor.SCAN_DURATION,
                "scan_dt_s": RobotPredictor.SCAN_DT,
                "min_time_hit_s": RobotPredictor.MIN_TIME_HIT,
                "bounce_fall_z_mm": RobotPredictor.MIN_BOUNCE_FALL_Z,
                "bounce_rise_frames": RobotPredictor.BOUNCE_RISE_FRAMES,
                "post_bounce_min_updates": RobotPredictor.POST_BOUNCE_MIN_UPDATES,
                "min_send_buffer": self.MIN_SEND_BUFFER,
            },
            "throws": self._throw_log,
        }
        try:
            with open(self._throw_log_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=1)
            _print(f"[LOG] Saved {len(self._throw_log)} throws -> {self._throw_log_path}")
        except Exception as e:
            _print(f"[WARN] Failed to save throw log: {e}")

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
                    _print("[AUTO] Intercept done. Press 'h' to home.")
                    self._pending_action = None
                    self._stm32_moving = False
                    self._end_throw_log("intercept_done")
                    self.predictor.reset()
                    self.intercept_sent = False
                    self._update_count = 0

                elif self._pending_action == 'homing':
                    self._pending_action = None
                    self._stm32_moving = False
                    self.robot_current_pos = ROBOT_HOME
                    self._end_throw_log("manual_home_done")
                    self.predictor.reset()
                    self.intercept_sent = False
                    self._update_count = 0
                    _print("[HOME] Home done. Ready for next throw.")

            elif "TARGET OUT OF WORKSPACE" in upper:
                _print("[WARN] STM32 rejected target (IK workspace). Auto-clearing.")
                self._pending_action = None
                self._stm32_moving = False
                self._end_throw_log("target_out_of_workspace")
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
                self._end_throw_log("planning_failed")
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

    MIN_SEND_BUFFER = 10  # don't send until KF has enough points for stable velocity

    def maybe_send(self, intercept, frame_ts):
        if (not self.robot_homed or not self.run_gate or
                self.shutdown_requested):
            return
        if intercept is None:
            return

        # Wait for enough buffer points before first send (velocity too noisy early on)
        if not self.intercept_sent and len(self.predictor.positions) < self.MIN_SEND_BUFFER:
            return

        # Don't send while ball is rising post-bounce (VZ > 0).
        # Predicting through the peak amplifies VZ noise into large position errors.
        # Wait until VZ turns negative (ball descending toward workspace).
        if self.predictor.velocity is not None and self.predictor._bounce_count > 0:
            vz = self.predictor.velocity[2]
            if vz > 0:
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
                vx_mm_s=intercept.get('vx', 0.0),
                vy_mm_s=intercept.get('vy', 0.0),
                vz_mm_s=intercept.get('vz', 0.0),
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

            self._log_throw_send(intercept, t_adjusted, n_pts, latency, is_update, frame_ts)
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
                frame_ts = result.get("capture_time", time.perf_counter())
                if result["left_frame"] is None:
                    continue

                frame_count += 1
                if frame_count % 30 == 0:
                    fps = 30.0 / max(1e-6, time.time() - fps_time)
                    fps_time = time.time()

                robot_pos = None
                intercept = None
                pos_accepted = None

                if self.run_gate and result["found_3d"]:
                    reproj = result.get("reproj_err", 0)
                    if reproj > 1.5:
                        result["found_3d"] = False
                        result["reject_reason"] = f"reproj({reproj:.1f}px)"
                    else:
                        cx, cy, cz = result["position_3d"]
                        rx, ry, rz = cam_to_robot(self.R, self.t_vec, self.cam_scale, cx, cy, cz)
                        robot_pos = (rx, ry, rz)
                        pos_accepted = self.predictor.add_position(rx, ry, rz, frame_ts)

                if self.run_gate and self.predictor.is_ready():
                    intercept = self.predictor.predict_intercept(
                        robot_pos=self.robot_current_pos)
                    if intercept is not None and result["found_3d"]:
                        self.maybe_send(intercept, frame_ts)

                # --- Throw data logging ---
                self._log_throw_frame(
                    frame_ts, result, robot_pos, pos_accepted, intercept)

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
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

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
                        self._end_throw_log("gate_on")
                        self.predictor.reset()
                        self.intercept_sent = False
                        _print("[GATE] ON -- tracking active, ready to send")
                    else:
                        self._end_throw_log("gate_off")
                        self.intercept_sent = True
                        _print("[GATE] OFF")
                elif key == ord("c"):
                    self._end_throw_log("manual_clear")
                    self.intercept_sent = False
                    self.predictor.reset()
                    _print("[CLEAR] Manual clear -- ready for next intercept")
                elif key == ord("r"):
                    self._end_throw_log("manual_reset")
                    self.predictor.reset()
                    self.intercept_sent = False
                    _print("[RESET] Predictor reset")
                elif key == ord("b"):
                    self._end_throw_log("bg_reset")
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
            # Save any active throw before exit
            self._end_throw_log("shutdown")
            if self._throw_log:
                self._save_throw_log()

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

        _print(f"Done! Total throws sent: {self.throw_count}  "
               f"Logged throws: {len(self._throw_log)}")


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
