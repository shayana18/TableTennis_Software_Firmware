"""
Live stereo integration test: updated triangulation + updated trajectory planner + UART.

This script is the test-day integration entrypoint and keeps the same core behavior:
1) Send TARGET_HOME on startup.
2) Wait for home confirmation from STM32.
3) Run live stereo tracking and trajectory prediction.
4) Send latency-compensated TARGET_INTERCEPT messages while running.
5) On quit, stop intercept TX and send one TARGET_HOME.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from comm_function.transmit_over_uart import UartComm
from config.camera_config import load_camera_settings
from tracking.stereo_triangulator import StereoTriangulator
from trajectory.trajectory_predictor import TrajectoryPredictor


def _planner_print(*args, **kwargs) -> None:
    print("from planner in terminal", *args, **kwargs)


class IntegrationTestDay:
    def __init__(
        self,
        uart_port: str,
        baud_rate: int = 115200,
        robot_x_cm: float = 50.0,
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

        # Updated planner intercepts on camera-X plane (cm).
        self.robot_x_cm = float(robot_x_cm)

        self.triangulator: Optional[StereoTriangulator] = None
        self.predictor: Optional[TrajectoryPredictor] = None
        self.uart = UartComm(port=uart_port, baud_rate=baud_rate, verbose=uart_verbose)

        self.show_velocity = True
        self.show_trajectory = True

        self.robot_homed = False
        self.accept_intercept_targets = False
        self.shutdown_requested = False
        self.shutdown_home_sent = False

        self.home_ack_timeout_s = home_ack_timeout_s
        self.tx_interval_s = float(tx_interval_s)
        self.warmup_s = max(0.0, float(warmup_s))
        self.last_tx_time = 0.0
        self.last_cmd = None
        self.last_reject_log_t = 0.0

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

    def initialize_predictor(self) -> None:
        self.predictor = TrajectoryPredictor(
            buffer_size=15,
            min_points=4,
            velocity_method="regression",
            gravity=981.0,
            y_down=True,
            enable_drag=True,
            robot_x_cam=self.robot_x_cm,
        )

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
        _planner_print("Robot homed, game can start now")
        return True

    def request_shutdown_home(self) -> None:
        self.shutdown_requested = True
        self.accept_intercept_targets = False
        if self.shutdown_home_sent:
            return

        try:
            if self.uart.is_open:
                self.uart.send_home()
                self.shutdown_home_sent = True
        except Exception as exc:
            _planner_print(f"[UART] Failed to send shutdown home command: {exc}")

    def maybe_send_intercept_target(self, cmd: dict, frame_timestamp_s: float) -> None:
        if not self.robot_homed or not self.accept_intercept_targets or self.shutdown_requested:
            return
        if not cmd.get("valid"):
            return
        if not cmd.get("in_workspace", False):
            now = time.perf_counter()
            if now - self.last_reject_log_t > 0.5:
                _planner_print(
                    f"[UART] Not sending out-of-workspace target: "
                    f"({cmd['robot_x']:+.0f}, {cmd['robot_y']:+.0f}, {cmd['robot_z']:+.0f})"
                )
                self.last_reject_log_t = now
            return

        now = time.perf_counter()
        if (now - self.last_tx_time) < self.tx_interval_s:
            return

        time_sent = time.perf_counter()
        latency = max(0.0, time_sent - float(frame_timestamp_s))
        adjusted_intercept_time = max(0.0, float(cmd["t"]) - latency)

        try:
            self.uart.send_intercept(
                x_mm=float(cmd["robot_x"]),
                y_mm=float(cmd["robot_y"]),
                z_mm=float(cmd["robot_z"]),
                intercept_time_s=adjusted_intercept_time,
                time_sent_s=time_sent,
                timestamp_s=float(frame_timestamp_s),
            )
            self.last_tx_time = time_sent
            self.last_cmd = cmd
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

    def draw_trajectory(self, frame, trajectory, color=(220, 160, 50)) -> None:
        if len(trajectory) < 2:
            return

        points = []
        for x, y, z, _ in trajectory:
            uv = self.triangulator.project_to_image((x, y, z), camera="left")
            if uv is None:
                continue
            points.append((int(uv[0]), int(uv[1])))

        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], color, 2, cv2.LINE_AA)
        for i in range(0, len(points), max(1, len(points) // 15)):
            cv2.circle(frame, points[i], 3, color, -1)

    def draw_intercept(self, frame, cmd: dict) -> None:
        if not cmd.get("valid"):
            return

        uv = self.triangulator.project_to_image(
            (cmd["cam_x"], cmd["cam_y"], cmd["cam_z"]), camera="left"
        )
        if uv is None:
            return
        px, py = int(uv[0]), int(uv[1])

        color = (0, 255, 0) if cmd.get("in_workspace") else (0, 0, 255)
        cv2.line(frame, (px - 12, py - 12), (px + 12, py + 12), color, 2)
        cv2.line(frame, (px - 12, py + 12), (px + 12, py - 12), color, 2)
        cv2.circle(frame, (px, py), 16, color, 2)
        cv2.putText(
            frame,
            f"t={cmd['t']*1000:.0f}ms {cmd.get('strategy','?')}",
            (px + 20, py - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )

    def draw_overlay(self, left_vis, right_vis, fps: float, result: dict, cmd: dict) -> None:
        stats = self.predictor.get_stats()
        vel = self.predictor.get_velocity()

        cv2.putText(
            left_vis,
            f"FPS:{fps:.1f}  Buf:{stats['buffer_size']}  Rej:{stats['rejected']}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
        )
        cv2.putText(
            left_vis,
            f"Robot X plane: {self.robot_x_cm:.1f} cm",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 220, 255),
            1,
        )

        if self.show_velocity and vel["valid"]:
            cv2.putText(
                left_vis,
                f"Vx={vel['vx']:+.0f} Vy={vel['vy']:+.0f} Vz={vel['vz']:+.0f}  Spd={vel['speed']:.0f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
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

        tx_state = "ON" if self.accept_intercept_targets and not self.shutdown_requested else "OFF"
        cv2.putText(
            right_vis,
            f"Homed:{self.robot_homed}  TX:{tx_state}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0) if self.robot_homed else (0, 0, 255),
            1,
        )

        if cmd.get("valid"):
            ws = "IN" if cmd.get("in_workspace") else "OUT"
            reach = "Y" if cmd.get("reachable") else "N"
            cv2.putText(
                right_vis,
                f"Robot(mm): X={cmd['robot_x']:+.0f} Y={cmd['robot_y']:+.0f} Z={cmd['robot_z']:+.0f}",
                (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 220, 220),
                1,
            )
            cv2.putText(
                right_vis,
                f"t={cmd['t']*1000:.0f}ms  {cmd.get('strategy','?')}  WS:{ws}  Reach:{reach}",
                (10, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (220, 220, 220),
                1,
            )

        cv2.putText(
            right_vis,
            "q quit(home) | r reset | v vel | t traj | z/x plane | b bg",
            (10, right_vis.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.33,
            (120, 120, 120),
            1,
        )

    def run(self) -> None:
        _planner_print("\n" + "=" * 72)
        _planner_print(" TEST INTEGRATION DAY: Updated Stereo + Updated Planner + UART ")
        _planner_print("=" * 72)
        _planner_print(f"Cameras: L={self.cam_left_id} R={self.cam_right_id}")
        _planner_print(f"Interception plane (camera X): {self.robot_x_cm:.1f} cm")
        _planner_print("Controls: q r v t p z x b")
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

        self.initialize_predictor()

        if not self.warmup_background():
            self.request_shutdown_home()
            self.triangulator.stop_cameras()
            self.uart.close()
            cv2.destroyAllWindows()
            return

        _planner_print("--- LIVE TRACKING + UART TARGET TX ---")

        fps_time = time.time()
        fps = 0.0
        frame_count = 0

        try:
            while True:
                self.uart.print_pending_status()

                result = self.triangulator.update()
                frame_timestamp = time.perf_counter()
                if result["left_frame"] is None:
                    continue

                frame_count += 1
                if frame_count % 30 == 0:
                    fps = 30.0 / max(1e-6, (time.time() - fps_time))
                    fps_time = time.time()

                if result["found_3d"]:
                    self.predictor.add_position(*result["position_3d"], timestamp=frame_timestamp)

                cmd = self.predictor.get_robot_command(target_x=self.robot_x_cm)
                if result["found_3d"] and cmd.get("valid"):
                    self.maybe_send_intercept_target(cmd, frame_timestamp)

                left_vis, right_vis = self.triangulator.draw_results(result)

                if self.show_trajectory and self.predictor.is_ready():
                    traj = self.predictor.predict_trajectory(duration=0.8, dt=0.005)
                    self.draw_trajectory(left_vis, traj)
                if cmd.get("valid"):
                    self.draw_intercept(left_vis, cmd)

                self.draw_overlay(left_vis, right_vis, fps, result, cmd)

                dw = 640
                dh = int(dw * self.frame_height / self.frame_width)
                left_small = cv2.resize(left_vis, (dw, dh))
                right_small = cv2.resize(right_vis, (dw, dh))
                cv2.imshow("Integration Test Day", cv2.hconcat([left_small, right_small]))

                if result["found_3d"] and self.predictor.get_velocity()["valid"]:
                    x, y, z = result["position_3d"]
                    v = self.predictor.get_velocity()
                    line = (
                        f"\rfrom planner in terminal Pos:({x:6.1f},{y:6.1f},{z:6.1f}) "
                        f"Vel:({v['vx']:6.1f},{v['vy']:6.1f},{v['vz']:6.1f})"
                    )
                    if cmd.get("valid"):
                        line += (
                            f" Cmd(mm):({cmd['robot_x']:+6.0f},{cmd['robot_y']:+6.0f},"
                            f"{cmd['robot_z']:+7.0f}) t={cmd['t']*1000:5.0f}ms"
                        )
                    print(line + "   ", end="")

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    _planner_print("[QUIT] Sending home and stopping intercept transmission...")
                    self.request_shutdown_home()
                    break
                if key == ord("r"):
                    self.predictor.reset()
                    _planner_print("[RESET]")
                elif key == ord("v"):
                    self.show_velocity = not self.show_velocity
                elif key == ord("t"):
                    self.show_trajectory = not self.show_trajectory
                elif key == ord("p"):
                    _planner_print(f"[STATS] {self.predictor.get_stats()}")
                elif key == ord("z"):
                    self.robot_x_cm += 5.0
                    self.predictor.set_robot_endline(self.robot_x_cm)
                    _planner_print(f"[Robot X plane = {self.robot_x_cm:.1f} cm]")
                elif key == ord("x"):
                    self.robot_x_cm -= 5.0
                    self.predictor.set_robot_endline(self.robot_x_cm)
                    _planner_print(f"[Robot X plane = {self.robot_x_cm:.1f} cm]")
                elif key == ord("b"):
                    self.triangulator.reset_background()
                    self.predictor.reset()
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
        description="Updated integration test: stereo triangulation + trajectory + UART"
    )
    parser.add_argument(
        "--port",
        default=os.environ.get("STM32_UART_PORT"),
        help="UART port for STM32 (or set STM32_UART_PORT env var)",
    )
    parser.add_argument("--baud", type=int, default=115200, help="UART baud (default: 115200)")
    parser.add_argument(
        "--robot-x-cm",
        type=float,
        default=50.0,
        help="Interception plane in camera X (cm). Default: 50",
    )
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
        robot_x_cm=args.robot_x_cm,
        home_ack_timeout_s=home_ack_timeout,
        uart_verbose=(not args.quiet_uart),
        tx_interval_s=max(0.0, float(args.tx_interval_ms) / 1000.0),
        warmup_s=args.warmup_s,
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
