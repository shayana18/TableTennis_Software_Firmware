"""Production-oriented threaded game pipeline.

Pipeline ownership:
1) Vision thread: stereo triangulation + camera->robot transform
2) Predictor thread: temporal acceptance + trajectory planning
3) UART thread: command TX + status RX parsing
4) Main thread: game state machine + UI/controls
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ball_tracking.stereo_triangulator import StereoTriangulator
from comm_functions.points_based_transform import load_points_based_transform
from config.camera_config import load_camera_settings
from pipeline.messages import (
    PredictorCommand,
    PredictorCommandKind,
    PredictorUpdate,
    UartCommand,
    UartCommandKind,
    UartEvent,
    UartEventKind,
)
from pipeline.predictor_worker import PredictorWorker
from pipeline.uart_worker import UartWorker
from pipeline.vision_worker import VisionConfig, VisionWorker


def _print(*args, **kwargs):
    print("from planner in terminal", *args, **kwargs)


def _safe_put(q: queue.Queue, item) -> bool:
    try:
        q.put_nowait(item)
        return True
    except queue.Full:
        return False


@dataclass
class RuntimeState:
    """Global game-state snapshot owned by the main thread."""

    gate_on: bool = False
    startup_homing_done: bool = False
    robot_at_home: bool = False
    robot_state: str = "UNKNOWN"
    intercept_inflight: bool = False
    throws_sent: int = 0
    updates_sent: int = 0
    last_latency_ms: float = 0.0
    last_adjusted_ms: float = 0.0
    last_uart_line: str = ""


class GameApp:
    """Thread orchestrator + top-level state machine.

    This class keeps logic intentionally close to integration_simple while
    splitting responsibilities across workers for production readiness.
    """

    MIN_SEND_BUFFER = 5
    TIME_AGGRESSION = 1.0

    def __init__(
        self,
        *,
        uart_port: str,
        baud_rate: int,
        home_ack_timeout_s: Optional[float],
        warmup_s: float,
        tx_interval_s: float,
        uart_verbose: bool,
    ) -> None:
        self._home_ack_timeout_s = home_ack_timeout_s
        self._state = RuntimeState()
        self._stop_event = threading.Event()

        cam = load_camera_settings()
        self._frame_w = int(cam["frame_width"])
        self._frame_h = int(cam["frame_height"])
        self._cam_left_id = int(cam["camera0"])
        self._cam_right_id = int(cam["camera1"])

        transform = load_points_based_transform()
        self._R = transform["rotation"]
        self._t = transform["translation"]
        self._cam_scale = float(transform["camera_scale_to_robot_units"])

        calibration_dir = SCRIPT_DIR / "camera_params" / "camera_properties"
        self._triangulator = StereoTriangulator(
            calibration_dir=str(calibration_dir),
            cam_left_id=self._cam_left_id,
            cam_right_id=self._cam_right_id,
        )

        self._position_queue: queue.Queue = queue.Queue(maxsize=128)
        self._observation_queue: queue.Queue = queue.Queue(maxsize=2)
        self._predictor_cmd_queue: queue.Queue = queue.Queue(maxsize=16)
        self._predictor_update_queue: queue.Queue = queue.Queue(maxsize=16)
        self._uart_cmd_queue: queue.Queue = queue.Queue(maxsize=16)
        self._uart_event_queue: queue.Queue = queue.Queue(maxsize=256)

        self._vision_worker = VisionWorker(
            triangulator=self._triangulator,
            transform_rotation=self._R,
            transform_translation=self._t,
            transform_scale=self._cam_scale,
            position_queue=self._position_queue,
            observation_queue=self._observation_queue,
            stop_event=self._stop_event,
            config=VisionConfig(
                frame_width=self._frame_w,
                frame_height=self._frame_h,
                reproj_err_max_px=10.0,
                warmup_s=warmup_s,
            ),
            status_printer=_print,
        )
        self._predictor_worker = PredictorWorker(
            position_queue=self._position_queue,
            predictor_cmd_queue=self._predictor_cmd_queue,
            predictor_update_queue=self._predictor_update_queue,
            stop_event=self._stop_event,
            status_printer=_print,
        )
        self._uart_worker = UartWorker(
            port=uart_port,
            baud_rate=baud_rate,
            verbose=uart_verbose,
            uart_cmd_queue=self._uart_cmd_queue,
            uart_event_queue=self._uart_event_queue,
            stop_event=self._stop_event,
            tx_interval_s=tx_interval_s,
            time_aggression=self.TIME_AGGRESSION,
            status_printer=_print,
        )

        self._latest_obs = None
        self._latest_predictor_update: Optional[PredictorUpdate] = None
        self._latest_intercept = None
        self._last_sent_intercept = None
        # Allow exactly one intercept send per robot IDLE cycle.
        self._idle_send_armed = False
        self._fps = 0.0
        self._fps_count = 0
        self._fps_t0 = time.perf_counter()

    def _send_predictor(self, kind: PredictorCommandKind) -> None:
        _safe_put(self._predictor_cmd_queue, PredictorCommand(kind=kind))

    def _send_uart(self, kind: UartCommandKind, intercept=None) -> None:
        _safe_put(self._uart_cmd_queue, UartCommand(kind=kind, intercept=intercept))

    def _wait_for_startup_home(self) -> bool:
        """Startup handshake:
        1) wait for UART open
        2) send HOME
        3) wait until home completion is confirmed

        We require the post-command completion signal to avoid accepting
        stale IDLE lines printed before HOME was issued.
        """
        deadline = None
        if self._home_ack_timeout_s is not None:
            deadline = time.perf_counter() + float(self._home_ack_timeout_s)

        opened = False
        home_cmd_sent = False
        while not self._stop_event.is_set():
            if deadline is not None and time.perf_counter() > deadline and not self._state.startup_homing_done:
                _print("[startup] Home/idle confirmation timeout.")
                return False
            try:
                evt: UartEvent = self._uart_event_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if evt.kind == UartEventKind.UART_OPENED:
                opened = True
                self._send_uart(UartCommandKind.HOME)
                continue
            if evt.kind == UartEventKind.TX_HOME:
                home_cmd_sent = True
                continue

            if evt.kind == UartEventKind.UART_ERROR:
                _print(f"[startup] UART error: {evt.data}")
                return False

            if evt.line:
                self._state.last_uart_line = evt.line
                print(f"[UART][RX] {evt.line}")

            if evt.kind == UartEventKind.HOME_DONE:
                self._state.startup_homing_done = True
                self._state.robot_at_home = True
                self._state.robot_state = "IDLE"
                _print("[startup] Home complete, robot is IDLE. Starting pipeline.")
                return True

            if (
                evt.kind == UartEventKind.STATE_IDLE
                and opened
                and home_cmd_sent
                and not self._state.startup_homing_done
            ):
                # Conservative fallback if firmware reaches IDLE before HOME_DONE event.
                self._state.startup_homing_done = True
                self._state.robot_at_home = True
                self._state.robot_state = "IDLE"
                _print("[startup] Robot reached IDLE. Starting pipeline.")
                return True

        return False

    def _handle_uart_event(self, evt: UartEvent) -> None:
        # Print all raw firmware lines for easy field debugging.
        if evt.line:
            self._state.last_uart_line = evt.line
            print(f"[UART][RX] {evt.line}")

        if evt.kind == UartEventKind.STATE_OFF:
            self._state.robot_state = "OFF"
        elif evt.kind == UartEventKind.STATE_PLAN:
            self._state.robot_state = "PLAN"
        elif evt.kind == UartEventKind.STATE_MOVE:
            self._state.robot_state = "MOVE"
            self._state.robot_at_home = False
        elif evt.kind == UartEventKind.STATE_IDLE:
            self._state.robot_state = "IDLE"
        elif evt.kind == UartEventKind.TX_HOME:
            self._state.robot_at_home = False
        elif evt.kind == UartEventKind.TX_INTERCEPT:
            self._state.intercept_inflight = True
            self._state.robot_at_home = False
            self._idle_send_armed = False
            is_update = bool(evt.data.get("is_update", False))
            if is_update:
                self._state.updates_sent += 1
            else:
                self._state.throws_sent += 1
            self._state.last_latency_ms = float(evt.data.get("latency_ms", 0.0))
            self._state.last_adjusted_ms = float(evt.data.get("adjusted_time_ms", 0.0))
        elif evt.kind == UartEventKind.INTERCEPT_DONE:
            # Guard against duplicate done events (COMPLETED Q + STATE: IDLE).
            if not self._state.intercept_inflight:
                return
            self._state.intercept_inflight = False
            self._send_uart(UartCommandKind.HOME)
            self._send_predictor(PredictorCommandKind.RESET)
            self._latest_intercept = None
            self._last_sent_intercept = None
        elif evt.kind == UartEventKind.HOME_DONE:
            # Guard against duplicate done events.
            if self._state.robot_at_home:
                return
            self._state.robot_at_home = True
            self._state.intercept_inflight = False
            self._state.robot_state = "IDLE"
            self._idle_send_armed = True
            self._send_predictor(PredictorCommandKind.RESET)
            self._last_sent_intercept = None
        elif evt.kind in (UartEventKind.TARGET_REJECTED, UartEventKind.PLAN_FAILED):
            # Treat planner rejects/failures as recoverable and keep the game loop
            # live. Do not block on an additional HOME handshake here.
            self._state.intercept_inflight = False
            self._state.robot_at_home = True
            self._state.robot_state = "IDLE"
            self._idle_send_armed = True
            self._send_predictor(PredictorCommandKind.RESET)
            self._latest_intercept = None
            self._last_sent_intercept = None
        elif evt.kind == UartEventKind.UART_ERROR:
            _print(f"[uart] ERROR: {evt.data}")

    def _handle_predictor_update(self, update: PredictorUpdate) -> None:
        self._latest_predictor_update = update
        if update.intercept is not None:
            self._latest_intercept = update.intercept

    def _intercept_common_ready(self) -> bool:
        if self._latest_intercept is None:
            return False
        if self._latest_intercept.buffer_points < self.MIN_SEND_BUFFER:
            return False
        if self._latest_intercept.bounce_count > 0 and self._latest_predictor_update is not None:
            vel = self._latest_predictor_update.velocity
            if vel is not None and vel[2] > 0:
                return False
        return True

    def _can_send_initial_intercept(self) -> bool:
        """Legacy first-send gate: requires robot at home + no in-flight command."""
        if not self._state.startup_homing_done:
            return False
        if not self._state.gate_on:
            return False
        if not self._idle_send_armed:
            return False
        if not self._state.robot_at_home:
            return False
        if self._state.intercept_inflight:
            return False
        return self._intercept_common_ready()

    def _can_send_update_intercept(self) -> bool:
        # One target per IDLE cycle: no pre-MOVE target overwrite updates.
        return False

    def _draw_overlay(self):
        """Main-thread visualization only (OpenCV UI should stay on main thread)."""
        if self._latest_obs is None or self._latest_obs.left_frame is None:
            return
        vis = self._latest_obs.left_frame.copy()

        gate_txt = "ON" if self._state.gate_on else "OFF"
        cv2.putText(
            vis,
            f"FPS:{self._fps:.0f} Gate:{gate_txt} Robot:{self._state.robot_state}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0) if self._state.gate_on else (0, 0, 255),
            1,
        )
        cv2.putText(
            vis,
            f"Home:{self._state.robot_at_home} Inflight:{self._state.intercept_inflight} Throws:{self._state.throws_sent} Updates:{self._state.updates_sent}",
            (10, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        if self._latest_predictor_update is not None:
            s = self._latest_predictor_update
            cv2.putText(
                vis,
                f"Buf:{s.buffer_points} Ready:{s.ready} Accepted:{s.accepted}",
                (10, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 255, 200),
                1,
            )
            if s.sample is not None:
                cv2.putText(
                    vis,
                    f"Ball(mm): X={s.sample.x_mm:+.0f} Y={s.sample.y_mm:+.0f} Z={s.sample.z_mm:+.0f}",
                    (10, 91),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

        if self._latest_intercept is not None:
            ip = self._latest_intercept
            cv2.putText(
                vis,
                f"Target(mm): X={ip.x_mm:+.0f} Y={ip.y_mm:+.0f} Z={ip.z_mm:+.0f} t={ip.intercept_time_s*1000:.0f}ms",
                (10, 114),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 220, 220),
                1,
            )

        cv2.putText(
            vis,
            f"Latency:{self._state.last_latency_ms:.1f}ms  Adjusted:{self._state.last_adjusted_ms:.1f}ms",
            (10, 137),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (255, 220, 120),
            1,
        )
        cv2.putText(
            vis,
            "Controls: g=toggle gameplay  h=manual home  r=reset predictor  q=quit",
            (10, vis.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (180, 180, 180),
            1,
        )

        dw = 960
        dh = int(dw * self._frame_h / self._frame_w)
        cv2.imshow("Game", cv2.resize(vis, (dw, dh)))

    def run(self) -> int:
        _print("\n" + "=" * 70)
        _print("THREADED GAME PIPELINE")
        _print("=" * 70)
        _print(f"Cameras: L={self._cam_left_id} R={self._cam_right_id}")
        _print(f"Transform scale: {self._cam_scale}")
        _print("Startup rule: gameplay starts only after HOME completes and robot is IDLE")
        _print("=" * 70)

        self._uart_worker.start()
        if not self._wait_for_startup_home():
            self._stop_event.set()
            return 1
        self._idle_send_armed = True

        # Start compute workers only after startup homing is confirmed complete.
        self._predictor_worker.start()
        self._vision_worker.start()
        self._send_predictor(PredictorCommandKind.DISABLE)
        self._send_predictor(PredictorCommandKind.RESET)

        try:
            while not self._stop_event.is_set():
                drained_any = False
                while True:
                    try:
                        evt = self._uart_event_queue.get_nowait()
                    except queue.Empty:
                        break
                    drained_any = True
                    self._handle_uart_event(evt)

                while True:
                    try:
                        upd = self._predictor_update_queue.get_nowait()
                    except queue.Empty:
                        break
                    drained_any = True
                    self._handle_predictor_update(upd)

                while True:
                    try:
                        obs = self._observation_queue.get_nowait()
                    except queue.Empty:
                        break
                    drained_any = True
                    self._latest_obs = obs
                    self._fps_count += 1
                    if self._fps_count % 30 == 0:
                        now = time.perf_counter()
                        self._fps = 30.0 / max(1e-6, now - self._fps_t0)
                        self._fps_t0 = now

                if self._can_send_initial_intercept() or self._can_send_update_intercept():
                    # Send first intercept from HOME, then allow legacy PLAN-phase
                    # updates until firmware reports MOVE.
                    self._send_uart(UartCommandKind.INTERCEPT, intercept=self._latest_intercept)
                    self._last_sent_intercept = self._latest_intercept
                    self._state.intercept_inflight = True
                    self._latest_intercept = None

                self._draw_overlay()
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    _print("[quit] Stopping game, sending HOME.")
                    self._send_uart(UartCommandKind.HOME)
                    break
                elif key == ord("g"):
                    self._state.gate_on = not self._state.gate_on
                    if self._state.gate_on:
                        # Fresh throw: clear old state, then enable predictor.
                        self._send_predictor(PredictorCommandKind.RESET)
                        self._send_predictor(PredictorCommandKind.ENABLE)
                        if self._state.robot_at_home and not self._state.intercept_inflight:
                            self._idle_send_armed = True
                        _print("[gate] ON")
                    else:
                        # Gate-off should stop prediction immediately.
                        self._send_predictor(PredictorCommandKind.DISABLE)
                        self._send_predictor(PredictorCommandKind.RESET)
                        self._latest_intercept = None
                        self._last_sent_intercept = None
                        _print("[gate] OFF")
                elif key == ord("h"):
                    self._state.gate_on = False
                    self._state.intercept_inflight = False
                    self._idle_send_armed = False
                    self._latest_intercept = None
                    self._last_sent_intercept = None
                    self._send_predictor(PredictorCommandKind.DISABLE)
                    self._send_predictor(PredictorCommandKind.RESET)
                    self._send_uart(UartCommandKind.HOME)
                    _print("[home] Manual HOME command sent.")
                elif key == ord("r"):
                    self._latest_intercept = None
                    self._last_sent_intercept = None
                    self._send_predictor(PredictorCommandKind.RESET)
                    _print("[predictor] RESET")

                if not drained_any:
                    time.sleep(0.002)

        except KeyboardInterrupt:
            _print("[ctrl-c] stopping...")
        finally:
            self._stop_event.set()
            self._send_predictor(PredictorCommandKind.SHUTDOWN)
            self._send_uart(UartCommandKind.SHUTDOWN)
            self._vision_worker.join(timeout=2.0)
            self._predictor_worker.join(timeout=2.0)
            self._uart_worker.join(timeout=2.0)
            cv2.destroyAllWindows()

        _print(
            f"Done. Throws sent={self._state.throws_sent} Updates sent={self._state.updates_sent} "
            f"Last latency={self._state.last_latency_ms:.1f}ms"
        )
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Threaded game pipeline")
    parser.add_argument(
        "--port",
        default=os.environ.get("STM32_UART_PORT"),
        help="UART port (or set STM32_UART_PORT)",
    )
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument(
        "--home-ack-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for startup home->IDLE (0=infinite)",
    )
    parser.add_argument("--tx-interval-ms", type=float, default=30.0)
    parser.add_argument("--warmup-s", type=float, default=2.0)
    parser.add_argument("--quiet-uart", action="store_true")
    args = parser.parse_args()

    if not args.port:
        parser.error("UART port required. Pass --port or set STM32_UART_PORT.")

    timeout = None if args.home_ack_timeout == 0.0 else args.home_ack_timeout

    app = GameApp(
        uart_port=args.port,
        baud_rate=args.baud,
        home_ack_timeout_s=timeout,
        warmup_s=max(0.0, args.warmup_s),
        tx_interval_s=max(0.0, args.tx_interval_ms / 1000.0),
        uart_verbose=not args.quiet_uart,
    )
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
