"""
Live trajectory prediction + robot command transmitter (test day script).

This script ports the `test_trajectory_prediction.py` pipeline and adds:

1. UART handshake with the STM32:
   - Send `TARGET_HOME` on startup
   - Wait for a home-complete confirmation over the same UART
   - Print "Robot homed, game can start now" only after confirmation

2. UART transmission of interception targets:
   - Pack raw little-endian `float[7]` messages matching `target_t`
   - Send only `TARGET_INTERCEPT` and `TARGET_HOME`
   - Apply latency compensation to intercept time:
       latency = time_sent - timestamp
       adjusted_intercept_time = planner_time - latency

3. Quit safety behavior:
   - On quit, stop sending intercept targets and send one `TARGET_HOME`

All timestamps are laptop-side monotonic times (`time.perf_counter()` in seconds).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import cv2

# Add parent directory to path for imports (same pattern as test_trajectory_prediction.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from comm_function.transmit_over_uart import UartComm
from config.camera_config import configure_camera, load_camera_settings
from tracking.stereo_triangulator import StereoTriangulator
from trajectory.trajectory_predictor import TrajectoryPredictor


def _planner_print(*args, **kwargs) -> None:
    """Print a planner-side terminal message (not UART-received text)."""
    print("from planner in terminal", *args, **kwargs)


class MoveRobotTestDay:
    """
    Live stereo tracking + trajectory prediction + UART target transmission.

    This is intentionally kept simple and verbose for test-day debugging.
    """

    def __init__(
        self,
        uart_port: str,
        baud_rate: int = 115200,
        robot_z_cm: float = 50.0,
        home_ack_timeout_s: Optional[float] = 30.0,
        uart_verbose: bool = True,
        tx_interval_s: float = 0.03,
    ) -> None:
        # Determine base directory (parent of trajectory/)
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = os.path.dirname(self.script_dir)

        # File paths
        self.calibration_dir = os.path.join(self.base_dir, "camera_calibration", "camera_parameters")
        self.thresholds_stereo = os.path.join(self.base_dir, "config", "ball_thresholds_stereo.json")
        self.thresholds_single = os.path.join(self.base_dir, "config", "ball_thresholds.json")

        # Camera settings
        cam_settings = load_camera_settings()
        self.frame_width = cam_settings["frame_width"]
        self.frame_height = cam_settings["frame_height"]
        self.cam_left_id = cam_settings["camera0"]
        self.cam_right_id = cam_settings["camera1"]

        # Robot interception plane (triangulator/predictor use cm)
        self.robot_z = float(robot_z_cm)

        # Components
        self.triangulator: Optional[StereoTriangulator] = None
        self.predictor: Optional[TrajectoryPredictor] = None
        self.uart = UartComm(port=uart_port, baud_rate=baud_rate, verbose=uart_verbose)

        # Display options
        self.show_velocity = True
        self.show_trajectory = True

        # Runtime state
        self.robot_homed = False
        self.accept_intercept_targets = False
        self.shutdown_requested = False
        self.shutdown_home_sent = False
        self.home_ack_timeout_s = home_ack_timeout_s
        self.tx_interval_s = float(tx_interval_s)
        self.last_tx_time = 0.0
        self.last_measurement_timestamp = None

        self.load_config()

    def load_config(self) -> None:
        """Print resolved camera config for visibility."""
        _planner_print(
            f"[Config] Left={self.cam_left_id}, Right={self.cam_right_id}, "
            f"{self.frame_width}x{self.frame_height}"
        )

    def load_thresholds(self) -> None:
        """Legacy hook kept for parity with the original test script."""
        _planner_print("[Detection] Using MOG2 background subtraction (no thresholds needed)")

    def initialize_triangulator(self) -> bool:
        """Create the stereo triangulator after the robot is homed."""
        try:
            self.triangulator = StereoTriangulator(
                calibration_dir=self.calibration_dir,
                cam_left_id=self.cam_left_id,
                cam_right_id=self.cam_right_id,
            )
            return True
        except Exception as exc:
            _planner_print(f"\nERROR initializing triangulator: {exc}")
            _planner_print(f"  Calibration path: {self.calibration_dir}")
            return False

    def initialize_predictor(self) -> None:
        """Create the trajectory predictor used for interception planning."""
        self.predictor = TrajectoryPredictor(
            buffer_size=10,
            min_points=3,
            velocity_method="regression",
            gravity=981.0,
            y_down=True,
            enable_drag=True,
        )

    def start_cameras(self) -> None:
        """Open and configure stereo cameras."""
        if self.triangulator is None:
            raise RuntimeError("Triangulator must be initialized before starting cameras")

        _planner_print("\n[Cameras] Opening with DirectShow backend...")
        self.triangulator.cap_left = cv2.VideoCapture(self.cam_left_id, cv2.CAP_DSHOW)
        self.triangulator.cap_right = cv2.VideoCapture(self.cam_right_id, cv2.CAP_DSHOW)

        # Fallback to default backend if DirectShow is unavailable
        if not self.triangulator.cap_left.isOpened():
            self.triangulator.cap_left = cv2.VideoCapture(self.cam_left_id)
        if not self.triangulator.cap_right.isOpened():
            self.triangulator.cap_right = cv2.VideoCapture(self.cam_right_id)

        if not self.triangulator.cap_left.isOpened() or not self.triangulator.cap_right.isOpened():
            raise RuntimeError("Failed to open cameras")

        s_left = configure_camera(self.triangulator.cap_left, self.frame_width, self.frame_height)
        s_right = configure_camera(self.triangulator.cap_right, self.frame_width, self.frame_height)

        _planner_print(f"  LEFT:  {s_left['width']}x{s_left['height']} @ {s_left['fps']:.0f}fps")
        _planner_print(f"  RIGHT: {s_right['width']}x{s_right['height']} @ {s_right['fps']:.0f}fps")

        self.frame_width = s_left["width"]
        self.frame_height = s_left["height"]

    def draw_trajectory(self, frame, trajectory, color=(255, 100, 0)) -> None:
        """Draw predicted trajectory onto the left camera image."""
        if len(trajectory) < 2:
            return

        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        scale = 3.0  # pixels per cm

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

    def draw_intercept(self, frame, x, y, t_ms, color=(0, 0, 255)) -> None:
        """Draw the planned interception point and time on the image."""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        scale = 3.0

        px = int(cx + x * scale)
        py = int(cy + y * scale)
        px = max(20, min(w - 20, px))
        py = max(20, min(h - 20, py))

        cv2.line(frame, (px - 15, py - 15), (px + 15, py + 15), color, 3)
        cv2.line(frame, (px - 15, py + 15), (px + 15, py - 15), color, 3)
        cv2.circle(frame, (px, py), 20, color, 2)
        cv2.putText(
            frame,
            f"({x:.0f},{y:.0f}) {t_ms:.0f}ms",
            (px + 25, py),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    @staticmethod
    def _cm_to_mm(value_cm: float) -> float:
        """Convert centimeters to millimeters."""
        return float(value_cm) * 10.0

    def send_home_and_wait_for_confirmation(self) -> bool:
        """
        Send startup home command and wait for firmware confirmation.

        Returns:
            True when home confirmation is received, False on timeout/error.
        """
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

    def maybe_send_intercept_target(self, pred: dict, frame_timestamp_s: float) -> None:
        """
        Send a latency-compensated intercept target over UART.

        This uses the user-requested compensation:
            latency = time_sent - timestamp
            adjusted_intercept_time = intercept_time_from_planner - latency

        Notes:
            - `TrajectoryPredictor.predict()` exposes the planner time as
              `pred['time_to_intercept']` (seconds). We also check `pred['time']`
              for compatibility with any alternate planner output shape.
            - Position is converted from cm (planner units) to mm (STM32 target_t).
        """
        if not self.robot_homed or not self.accept_intercept_targets or self.shutdown_requested:
            return

        now = time.perf_counter()
        if (now - self.last_tx_time) < self.tx_interval_s:
            return

        intercept_time_from_planner = pred.get("time")
        if intercept_time_from_planner is None:
            intercept_time_from_planner = pred.get("time_to_intercept")

        if intercept_time_from_planner is None:
            return

        time_sent = time.perf_counter()
        latency = max(0.0, time_sent - float(frame_timestamp_s))
        adjusted_intercept_time = max(0.0, float(intercept_time_from_planner) - latency)

        try:
            self.uart.send_intercept(
                x_mm=self._cm_to_mm(pred["intercept_x"]),
                y_mm=self._cm_to_mm(pred["intercept_y"]),
                z_mm=self._cm_to_mm(pred["intercept_z"]),
                intercept_time_s=adjusted_intercept_time,
                time_sent_s=time_sent,
                timestamp_s=float(frame_timestamp_s),
            )
            self.last_tx_time = time_sent
        except Exception as exc:
            _planner_print(f"\n[UART] Failed to send intercept target: {exc}")

    def request_shutdown_home(self) -> None:
        """
        Stop sending intercepts and send one HOME command.

        This method is idempotent so it is safe to call from both key handling
        and `finally`.
        """
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

    def run(self) -> None:
        """Main application loop."""
        _planner_print("\n" + "=" * 72)
        _planner_print(" MOVE ROBOT TEST DAY: Stereo Tracking + Prediction + UART Target TX ")
        _planner_print("=" * 72)
        _planner_print(f"Robot Z plane: {self.robot_z:.1f} cm")
        _planner_print("Controls: q=quit(home) r=reset v=velocity t=trajectory p=stats z/x=adjust Z b=reset-bg")
        _planner_print("=" * 72)

        # Startup safety handshake: home robot before enabling tracking and interception.
        if not self.send_home_and_wait_for_confirmation():
            self.uart.close()
            return

        if not self.initialize_triangulator():
            self.request_shutdown_home()
            self.uart.close()
            return

        self.load_thresholds()

        try:
            self.start_cameras()
        except Exception as exc:
            _planner_print(f"\nERROR: {exc}")
            self.request_shutdown_home()
            self.uart.close()
            return

        self.initialize_predictor()

        _planner_print("\n--- LIVE TRACKING + UART TARGET TX ---\n")

        fps_time = time.time()
        fps = 0.0
        frame_count = 0

        try:
            while True:
                # Drain any robot debug/status messages (non-blocking)
                self.uart.print_pending_status()

                result = self.triangulator.update()
                frame_timestamp = time.perf_counter()

                if result["left_frame"] is None:
                    continue

                frame_count += 1
                if frame_count % 30 == 0:
                    fps = 30.0 / max(1e-6, (time.time() - fps_time))
                    fps_time = time.time()

                # Add triangulated point to predictor using an explicit timestamp
                if result["found_3d"]:
                    self.last_measurement_timestamp = frame_timestamp
                    self.predictor.add_position(*result["position_3d"], timestamp=frame_timestamp)

                pred = self.predictor.predict(target_z=self.robot_z)

                # Send intercept only when we have a fresh 3D observation and a valid prediction
                if result["found_3d"] and pred["valid"]:
                    self.maybe_send_intercept_target(pred, frame_timestamp)

                left_vis, right_vis = self.triangulator.draw_results(result)

                # Warmup overlay
                warmup = self.triangulator.warmup_status()
                if not warmup["left_ready"]:
                    cv2.putText(
                        left_vis,
                        f"L warming up... {warmup['left_progress']*100:.0f}%",
                        (10, left_vis.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )
                if not warmup["right_ready"]:
                    cv2.putText(
                        right_vis,
                        f"R warming up... {warmup['right_progress']*100:.0f}%",
                        (10, right_vis.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

                if self.show_trajectory and self.predictor.is_ready():
                    traj = self.predictor.predict_trajectory(duration=0.5, dt=0.01)
                    self.draw_trajectory(left_vis, traj)
                    if pred["valid"]:
                        self.draw_intercept(
                            left_vis,
                            pred["intercept_x"],
                            pred["intercept_y"],
                            pred["time_to_intercept"] * 1000.0,
                        )

                if self.show_velocity:
                    vel = self.predictor.get_velocity()
                    stats = self.predictor.get_stats()
                    y = 90
                    cv2.putText(
                        left_vis,
                        f"Buffer: {stats['buffer_size']}/10  FPS: {fps:.0f}",
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        1,
                    )
                    if vel["valid"]:
                        cv2.putText(
                            left_vis,
                            f"Velocity: Vz={vel['vz']:.0f} Speed={vel['speed']:.0f} cm/s",
                            (10, y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 0),
                            1,
                        )

                # UART status overlay
                tx_state = "ON" if self.accept_intercept_targets and not self.shutdown_requested else "OFF"
                cv2.putText(
                    left_vis,
                    f"Robot homed: {self.robot_homed}  TX: {tx_state}",
                    (10, left_vis.shape[0] - 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0) if self.robot_homed else (0, 0, 255),
                    2,
                )

                if pred["valid"]:
                    strat = pred.get("strategy", "z_plane")
                    if strat == "apex":
                        osd_text = (
                            f"[APEX] X={pred['intercept_x']:.0f} "
                            f"Y={pred['intercept_y']:.0f} Z={pred['intercept_z']:.0f} "
                            f"in {pred['time_to_intercept']*1000:.0f}ms"
                        )
                    else:
                        osd_text = (
                            f"[Z-PLANE] Z={self.robot_z:.0f}: "
                            f"X={pred['intercept_x']:.0f} Y={pred['intercept_y']:.0f} "
                            f"in {pred['time_to_intercept']*1000:.0f}ms"
                        )
                    cv2.putText(
                        left_vis,
                        osd_text,
                        (10, left_vis.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                # Display
                dw = 640
                dh = int(dw * self.frame_height / self.frame_width)
                left_vis = cv2.resize(left_vis, (dw, dh))
                right_vis = cv2.resize(right_vis, (dw, dh))
                cv2.imshow("Move Robot Test Day", cv2.hconcat([left_vis, right_vis]))

                # Console output (kept close to original script for familiarity)
                if result["found_3d"] and self.predictor.get_velocity()["valid"]:
                    x, y, z = result["position_3d"]
                    v = self.predictor.get_velocity()
                    print(
                        f"\rfrom planner in terminal Pos:({x:5.0f},{y:5.0f},{z:5.0f}) "
                        f"Vel:({v['vx']:5.0f},{v['vy']:5.0f},{v['vz']:5.0f}) ",
                        end="",
                    )
                    if pred["valid"]:
                        print(
                            f"Intercept:({pred['intercept_x']:5.0f},{pred['intercept_y']:5.0f},"
                            f"{pred['intercept_z']:5.0f}) in {pred['time_to_intercept']*1000:4.0f}ms",
                            end="",
                        )
                    print("   ", end="")

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    _planner_print("\n[QUIT] Sending home and stopping intercept transmission...\n")
                    self.request_shutdown_home()
                    break
                if key == ord("r"):
                    self.predictor.reset()
                    _planner_print("\n[RESET]\n")
                elif key == ord("v"):
                    self.show_velocity = not self.show_velocity
                elif key == ord("t"):
                    self.show_trajectory = not self.show_trajectory
                elif key == ord("p"):
                    _planner_print(f"\n[STATS] {self.predictor.get_stats()}\n")
                elif key == ord("z"):
                    self.robot_z += 10.0
                    _planner_print(f"\n[Robot Z = {self.robot_z}]\n")
                elif key == ord("x"):
                    self.robot_z = max(10.0, self.robot_z - 10.0)
                    _planner_print(f"\n[Robot Z = {self.robot_z}]\n")
                elif key == ord("b"):
                    self.triangulator.reset_background()
                    self.predictor.reset()
                    _planner_print("\n[BG RESET] Background model reset, warming up...\n")

        except KeyboardInterrupt:
            _planner_print("\n[CTRL-C] Sending home and shutting down...\n")
            self.request_shutdown_home()
        finally:
            # Final safety: make sure intercept TX stays off and HOME is sent once.
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
                # Drain any final status prints if the MCU replies quickly.
                self.uart.print_pending_status()
            except Exception:
                pass

            self.uart.close()

        _planner_print("\nDone!")


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for UART and runtime tuning."""
    parser = argparse.ArgumentParser(
        description="Live trajectory tracking + UART transmission to STM32 (test day script)"
    )
    parser.add_argument(
        "--port",
        default=os.environ.get("STM32_UART_PORT"),
        help="UART port for STM32 (or set STM32_UART_PORT env var)",
    )
    parser.add_argument("--baud", type=int, default=115200, help="UART baud rate (default: 115200)")
    parser.add_argument(
        "--robot-z-cm",
        type=float,
        default=50.0,
        help="Robot interception Z plane in cm (default: 50)",
    )
    parser.add_argument(
        "--home-ack-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for home confirmation (0 = no timeout, default: 30)",
    )
    parser.add_argument(
        "--tx-interval-ms",
        type=float,
        default=30.0,
        help="Minimum interval between intercept target transmissions (default: 30ms)",
    )
    parser.add_argument(
        "--quiet-uart",
        action="store_true",
        help="Reduce UART debug prints",
    )
    args = parser.parse_args()

    if not args.port:
        parser.error("UART port is required. Pass --port or set STM32_UART_PORT.")
    return args


def main() -> int:
    """CLI entrypoint."""
    args = _parse_args()
    home_ack_timeout = None if float(args.home_ack_timeout) == 0.0 else float(args.home_ack_timeout)

    app = MoveRobotTestDay(
        uart_port=args.port,
        baud_rate=args.baud,
        robot_z_cm=args.robot_z_cm,
        home_ack_timeout_s=home_ack_timeout,
        uart_verbose=(not args.quiet_uart),
        tx_interval_s=max(0.0, float(args.tx_interval_ms) / 1000.0),
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
