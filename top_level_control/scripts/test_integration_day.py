"""
Live stereo integration test: updated triangulation + robot-frame prediction + UART.

This script is the test-day integration entrypoint and keeps the same core behavior:
1) Send TARGET_HOME on startup.
2) Wait for home confirmation from STM32.
3) Keep top-level pipeline gated OFF until user opens it.
4) When gate is ON, run tracking/prediction and send one intercept at a time.
5) On quit, stop intercept TX and send one TARGET_HOME.

Uses RobotPredictor (robot-frame, mm) for all prediction logic.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
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
    robot_to_cam,
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


def _planner_print(*args, **kwargs) -> None:
    print("from planner in terminal", *args, **kwargs)


class IntegrationTestDay:
    def __init__(
        self,
        uart_port: str,
        baud_rate: int = 115200,
        home_ack_timeout_s: Optional[float] = 30.0,
        uart_verbose: bool = True,
        tx_interval_s: float = 0.03,
        warmup_s: float = 2.0,
    ) -> None:
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = os.path.dirname(self.script_dir)
        self.calibration_dir = os.path.join(
            self.base_dir, "camera_calibration", "camera_parameters"
        )

        cam_settings = load_camera_settings()
        self.frame_width = cam_settings["frame_width"]
        self.frame_height = cam_settings["frame_height"]
        self.cam_left_id = cam_settings["camera0"]
        self.cam_right_id = cam_settings["camera1"]

        self.triangulator: Optional[StereoTriangulator] = None
        self.predictor = RobotPredictor()
        self.uart = UartComm(port=uart_port, baud_rate=baud_rate, verbose=uart_verbose)

        # Points-based transform for camera -> robot conversion
        tf = load_points_based_transform()
        self.R = tf["rotation"]
        self.t_vec = tf["translation"]
        self.cam_scale = tf["camera_scale_to_robot_units"]

        self.show_velocity = True
        self.show_trajectory = True

        self.robot_homed = False
        self.accept_intercept_targets = False
        self.shutdown_requested = False
        self.shutdown_home_sent = False
        self.run_top_level = False
        self.intercept_target_sent = True

        self.home_ack_timeout_s = home_ack_timeout_s
        self.tx_interval_s = float(tx_interval_s)
        self.warmup_s = max(0.0, float(warmup_s))
        self.last_tx_time = 0.0
        self.last_cmd = None
        self.last_reject_log_t = 0.0

        # Track robot position
        self.robot_current_pos = ROBOT_HOME

    def check_calibration(self) -> bool:
        required = [
            "camera0_intrinsics.dat",
            "camera1_intrinsics.dat",
            "camera0_rot_trans.dat",
            "camera1_rot_trans.dat",
        ]
        missing = [
            f for f in required if not os.path.exists(os.path.join(self.calibration_dir, f))
        ]
        if missing:
            _planner_print("ERROR: Missing calibration files:")
            for f in missing:
                _planner_print(f"  - {f}")
            return False
        return True

    def initialize_triangulator(self) -> bool:
        try:
            self.triangulator = StereoTriangulator(
                calibration_dir=self.calibration_dir,
                cam_left_id=self.cam_left_id,
                cam_right_id=self.cam_right_id,
            )
            return True
        except Exception as exc:
            _planner_print(f"ERROR initializing triangulator: {exc}")
            _planner_print(f"Calibration path: {self.calibration_dir}")
            return False

    def send_home_and_wait_for_confirmation(self) -> bool:
        try:
            self.uart.open()
            self.uart.clear_input_buffer()
            self.uart.send_home()
            ack_line = self.uart.wait_for_home_confirmation(timeout_s=self.home_ack_timeout_s)
        except Exception as exc:
            _planner_print(f"[UART] ERROR during home handshake: {exc}")
            return False

        if ack_line is None:
            _planner_print("[UART] Timed out waiting for home confirmation")
            return False

        self.robot_homed = True
        self.accept_intercept_targets = True
        self.run_top_level = False
        self.intercept_target_sent = True
        _planner_print("Robot homed; top-level pipeline is gated OFF (press 'g' to run)")
        return True

    def request_shutdown_home(self) -> None:
        self.shutdown_requested = True
        self.accept_intercept_targets = False
        self.run_top_level = False
        self.intercept_target_sent = True
        if self.shutdown_home_sent:
            return

        try:
            if self.uart.is_open:
                self.uart.send_home()
                self.shutdown_home_sent = True
        except Exception as exc:
            _planner_print(f"[UART] Failed to send shutdown home command: {exc}")

    def clear_intercept_send_flag(self, reason: str = "user") -> None:
        self.intercept_target_sent = False
        _planner_print(
            f"[UART] Intercept-send flag cleared ({reason}); next valid target can be sent"
        )

    def set_run_top_level(self, enabled: bool, reason: str = "user") -> None:
        enabled = bool(enabled)
        if self.run_top_level == enabled:
            return

        self.run_top_level = enabled
        self.predictor.reset()

        if enabled:
            self.intercept_target_sent = False
            _planner_print(f"[GATE] run_top_level=ON ({reason})")
        else:
            self.intercept_target_sent = True
            _planner_print(f"[GATE] run_top_level=OFF ({reason})")

    def maybe_send_intercept_target(self, intercept: dict, frame_timestamp_s: float) -> None:
        if (
            not self.robot_homed
            or not self.accept_intercept_targets
            or not self.run_top_level
            or self.shutdown_requested
        ):
            return
        if intercept is None:
            return
        if self.intercept_target_sent:
            return

        rx, ry, rz = intercept['x'], intercept['y'], intercept['z']

        if not in_workspace(rx, ry, rz):
            now = time.perf_counter()
            if now - self.last_reject_log_t > 0.5:
                _planner_print(
                    f"[UART] Not sending out-of-workspace target: "
                    f"({rx:+.0f}, {ry:+.0f}, {rz:+.0f})"
                )
                self.last_reject_log_t = now
            return

        now = time.perf_counter()
        if (now - self.last_tx_time) < self.tx_interval_s:
            return

        time_sent = time.perf_counter()
        latency = max(0.0, time_sent - float(frame_timestamp_s))
        adjusted_intercept_time = max(0.0, float(intercept['time']) - latency)

        try:
            self.uart.send_intercept(
                x_mm=rx,
                y_mm=ry,
                z_mm=rz,
                vx_mm_s=intercept.get('vx', 0.0),
                vy_mm_s=intercept.get('vy', 0.0),
                vz_mm_s=intercept.get('vz', 0.0),
                intercept_time_s=adjusted_intercept_time,
                time_sent_s=time_sent,
                timestamp_s=float(frame_timestamp_s),
            )
            self.last_tx_time = time_sent
            self.last_cmd = intercept
            self.robot_current_pos = (rx, ry, rz)
            self.intercept_target_sent = True
            clamped = intercept.get('clamped', False)
            tag = " [CLAMPED]" if clamped else ""
            _planner_print(
                f"[UART]{tag} Intercept sent "
                f"(robot mm: x={rx:+.1f}, y={ry:+.1f}, z={rz:+.1f}; "
                f"t={adjusted_intercept_time*1000.0:.0f} ms)"
            )
            _planner_print("[UART] Waiting for manual clear before next send")
        except Exception as exc:
            _planner_print(f"[UART] Failed to send intercept target: {exc}")

    def warmup_background(self) -> bool:
        if self.warmup_s <= 0:
            return True

        _planner_print("Remove ball from view. Learning background (SPACE=skip)...")
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

            progress = min((time.time() - t0) / self.warmup_s, 1.0)
            vis = cv2.resize(fl, (640, int(640 * self.frame_height / self.frame_width)))
            h = vis.shape[0]
            bw = int(progress * (vis.shape[1] - 40))
            cv2.rectangle(vis, (20, h - 30), (20 + bw, h - 15), (0, 255, 255), -1)
            cv2.putText(
                vis,
                f"BG: {progress*100:.0f}%",
                (20, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
            )
            cv2.imshow("Integration Test Day", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                break
            if key == ord("q"):
                return False

        _planner_print("Background warmup ready.")
        return True

    def draw_intercept_marker(self, frame, intercept: dict) -> None:
        """Draw X marker at intercept point on camera image."""
        if intercept is None:
            return
        # Convert robot coords back to camera coords for projection
        cx, cy, cz = robot_to_cam(
            self.R, self.t_vec, self.cam_scale,
            intercept['x'], intercept['y'], intercept['z'])
        uv = self.triangulator.project_to_image((cx, cy, cz), camera="left")
        if uv is None:
            return
        u, v = float(uv[0]), float(uv[1])
        if not math.isfinite(u) or not math.isfinite(v):
            return
        px = int(round(u))
        py = int(round(v))
        if px < -5000 or py < -5000 or px > frame.shape[1] + 5000 or py > frame.shape[0] + 5000:
            return

        clamped = intercept.get('clamped', False)
        color = (0, 200, 255) if clamped else (0, 255, 255)
        cv2.line(frame, (px - 12, py - 12), (px + 12, py + 12), color, 2)
        cv2.line(frame, (px - 12, py + 12), (px + 12, py - 12), color, 2)
        cv2.circle(frame, (px, py), 16, color, 2)
        cv2.putText(
            frame,
            f"t={intercept['time']*1000:.0f}ms",
            (px + 20, py - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )

    def draw_overlay(
        self,
        left_vis,
        right_vis,
        fps: float,
        result: dict,
        intercept: Optional[dict],
        tri_robot_pos: Optional[tuple[float, float, float]],
    ) -> None:
        stats = self.predictor.get_stats()

        cv2.putText(
            left_vis,
            f"FPS:{fps:.1f}  Buf:{stats['buffer']}  Rej:{stats['rejected']}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
        )

        if self.show_velocity and self.predictor.velocity is not None:
            vx, vy, vz = self.predictor.velocity
            speed = math.sqrt(vx*vx + vy*vy + vz*vz)
            cv2.putText(
                left_vis,
                f"Vel(mm/s): X={vx:+.0f} Y={vy:+.0f} Z={vz:+.0f}  Spd={speed:.0f}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 0),
                1,
            )

        if result.get("found_3d"):
            x, y, z = result["position_3d"]
            cv2.putText(
                left_vis,
                f"3D: X={x:.1f} Y={y:.1f} Z={z:.1f} cm",
                (10, left_vis.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        if self.shutdown_requested or not self.accept_intercept_targets:
            tx_state = "OFF"
        elif not self.run_top_level:
            tx_state = "GATED"
        elif self.intercept_target_sent:
            tx_state = "WAIT_CLEAR"
        else:
            tx_state = "READY"
        cv2.putText(
            right_vis,
            f"Homed:{self.robot_homed}  Gate:{'ON' if self.run_top_level else 'OFF'}  TX:{tx_state}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0) if self.robot_homed else (0, 0, 255),
            1,
        )

        if tri_robot_pos is not None:
            cv2.putText(
                right_vis,
                (
                    f"Robot(mm): X={tri_robot_pos[0]:+.0f} "
                    f"Y={tri_robot_pos[1]:+.0f} Z={tri_robot_pos[2]:+.0f}"
                ),
                (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 180, 255),
                1,
            )
            cmd_row = 62
        else:
            cmd_row = 42

        if intercept is not None:
            ws = "IN" if in_workspace(intercept['x'], intercept['y'], intercept['z']) else "OUT"
            cv2.putText(
                right_vis,
                f"Int(mm): X={intercept['x']:+.0f} Y={intercept['y']:+.0f} Z={intercept['z']:+.0f}",
                (10, cmd_row),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 220, 220),
                1,
            )
            cv2.putText(
                right_vis,
                f"t={intercept['time']*1000:.0f}ms  WS:{ws}",
                (10, cmd_row + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (220, 220, 220),
                1,
            )

        cv2.putText(
            right_vis,
            "q quit(home) | g gate | c clearTX | r reset | v vel | b bg",
            (10, right_vis.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.33,
            (120, 120, 120),
            1,
        )

    def run(self) -> None:
        _planner_print("\n" + "=" * 72)
        _planner_print(" TEST INTEGRATION DAY: Stereo + RobotPredictor + UART ")
        _planner_print("=" * 72)
        _planner_print(f"Cameras: L={self.cam_left_id} R={self.cam_right_id}")
        _planner_print(f"Workspace: ellipse {ELLIPSE_A:.0f}x{ELLIPSE_B:.0f}mm, "
                       f"Z=[{Z_MIN:.0f}, {Z_MAX:.0f}]  Drag={DRAG_K:.6f}")
        _planner_print(f"Transform scale={self.cam_scale:.4f}")
        _planner_print("Controls: q g c r v b")
        _planner_print("=" * 72)

        if not self.send_home_and_wait_for_confirmation():
            self.uart.close()
            return

        if not self.check_calibration():
            self.request_shutdown_home()
            self.uart.close()
            return

        if not self.initialize_triangulator():
            self.request_shutdown_home()
            self.uart.close()
            return

        try:
            self.triangulator.start_cameras(self.frame_width, self.frame_height)
        except Exception as exc:
            _planner_print(f"ERROR starting cameras: {exc}")
            self.request_shutdown_home()
            self.uart.close()
            return

        if not self.warmup_background():
            self.request_shutdown_home()
            self.triangulator.stop_cameras()
            self.uart.close()
            cv2.destroyAllWindows()
            return

        _planner_print("--- LIVE TRACKING + UART TARGET TX ---")
        _planner_print("Top-level pipeline is gated OFF. Press 'g' to start tracking/planning/TX.")

        fps_time = time.time()
        fps = 0.0
        frame_count = 0

        try:
            while True:
                self.uart.print_pending_status()

                result = self.triangulator.update()
                frame_timestamp = result.get("capture_time", time.perf_counter())
                if result["left_frame"] is None:
                    continue

                tri_robot_pos = None
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = 30.0 / max(1e-6, (time.time() - fps_time))
                    fps_time = time.time()

                intercept = None
                if self.run_top_level:
                    if result["found_3d"]:
                        cx, cy, cz = result["position_3d"]
                        rx, ry, rz = cam_to_robot(
                            self.R, self.t_vec, self.cam_scale, cx, cy, cz)
                        tri_robot_pos = (rx, ry, rz)
                        self.predictor.add_position(rx, ry, rz, frame_timestamp)

                    if self.predictor.is_ready():
                        intercept = self.predictor.predict_intercept(
                            robot_pos=self.robot_current_pos)
                        if intercept is not None and result["found_3d"]:
                            self.maybe_send_intercept_target(intercept, frame_timestamp)

                left_vis, right_vis = self.triangulator.draw_results(result)

                if self.run_top_level and self.last_cmd and self.intercept_target_sent:
                    self.draw_intercept_marker(left_vis, self.last_cmd)

                self.draw_overlay(left_vis, right_vis, fps, result, intercept, tri_robot_pos)

                dw = 640
                dh = int(dw * self.frame_height / self.frame_width)
                left_small = cv2.resize(left_vis, (dw, dh))
                right_small = cv2.resize(right_vis, (dw, dh))
                cv2.imshow("Integration Test Day", cv2.hconcat([left_small, right_small]))

                if self.run_top_level and tri_robot_pos is not None:
                    line = (
                        f"\rfrom planner in terminal Robot(mm):({tri_robot_pos[0]:+6.0f},"
                        f"{tri_robot_pos[1]:+6.0f},{tri_robot_pos[2]:+7.0f})"
                    )
                    if intercept is not None:
                        line += (
                            f" Int:({intercept['x']:+6.0f},{intercept['y']:+6.0f},"
                            f"{intercept['z']:+7.0f}) t={intercept['time']*1000:5.0f}ms"
                        )
                    print(line + "   ", end="")

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    _planner_print("[QUIT] Sending home and stopping intercept transmission...")
                    self.request_shutdown_home()
                    break
                if key == ord("g"):
                    self.set_run_top_level(not self.run_top_level, reason="keyboard")
                elif key == ord("c"):
                    self.clear_intercept_send_flag(reason="keyboard")
                elif key == ord("r"):
                    self.predictor.reset()
                    self.clear_intercept_send_flag(reason="reset")
                    _planner_print("[RESET]")
                elif key == ord("v"):
                    self.show_velocity = not self.show_velocity
                elif key == ord("b"):
                    self.triangulator.reset_background()
                    self.predictor.reset()
                    self.clear_intercept_send_flag(reason="background reset")
                    _planner_print("[BG RESET] Re-learning background...")
                    if not self.warmup_background():
                        break

        except KeyboardInterrupt:
            _planner_print("[CTRL-C] Sending home and shutting down...")
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

            try:
                self.uart.print_pending_status()
            except Exception:
                pass

            self.uart.close()

        _planner_print("Done!")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integration test: stereo triangulation + RobotPredictor + UART"
    )
    parser.add_argument(
        "--port",
        default=os.environ.get("STM32_UART_PORT"),
        help="UART port for STM32 (or set STM32_UART_PORT env var)",
    )
    parser.add_argument("--baud", type=int, default=115200, help="UART baud (default: 115200)")
    parser.add_argument(
        "--home-ack-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for home confirmation (0 = no timeout)",
    )
    parser.add_argument(
        "--tx-interval-ms",
        type=float,
        default=30.0,
        help="Minimum interval between intercept messages (default: 30ms)",
    )
    parser.add_argument(
        "--warmup-s",
        type=float,
        default=2.0,
        help="Background warmup duration in seconds (default: 2.0, 0 disables)",
    )
    parser.add_argument("--quiet-uart", action="store_true", help="Reduce UART debug prints")
    args = parser.parse_args()

    if not args.port:
        parser.error("UART port is required. Pass --port or set STM32_UART_PORT.")
    return args


def main() -> int:
    args = _parse_args()
    home_ack_timeout = None if float(args.home_ack_timeout) == 0.0 else float(args.home_ack_timeout)

    app = IntegrationTestDay(
        uart_port=args.port,
        baud_rate=args.baud,
        home_ack_timeout_s=home_ack_timeout,
        uart_verbose=(not args.quiet_uart),
        tx_interval_s=max(0.0, float(args.tx_interval_ms) / 1000.0),
        warmup_s=args.warmup_s,
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
