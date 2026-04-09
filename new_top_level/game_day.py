"""Rally-oriented game pipeline for game day use.

This is intentionally close to `game.py`, but changes the top-level state
machine so we do not introduce extra downtime between shots:

1. Do not send an extra HOME after an intercept cycle completes.
2. Do not reset the predictor after every hit/home cycle.
3. Keep collecting ball data continuously while gameplay is enabled.
4. Send the next fresh intercept immediately once the robot is back home.

Important firmware constraint:
The current STM32 firmware only consumes new targets in `STATE_IDLE`, and
`set_idle()` clears the mailbox when HOME completes. That means targets sent
while the robot is auto-homing cannot truly preload the next strike with the
current firmware. This script still improves rally behavior by preserving the
predictor state and removing the extra Python-side reset/home latency.
"""

from __future__ import annotations

import argparse
import os
import time

import cv2

from game import GameApp, _print
from pipeline.messages import PredictorCommandKind, UartCommandKind, UartEvent, UartEventKind


class GameDayApp(GameApp):
    """Variant of GameApp tuned for repeated rally attempts."""

    MAX_READY_INTERCEPT_AGE_S = 0.15
    UPDATE_DISTANCE_MM = 80.0

    def __init__(self, *args, rally_enabled: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._rally_enabled = bool(rally_enabled)
        self._manual_home_requested = False
        self._returning_home = False
        self._last_sent_capture_time = -1.0

    def _intercept_is_fresh(self) -> bool:
        if self._latest_intercept is None:
            return False
        if self._latest_intercept.source_capture_time <= (self._last_sent_capture_time + 1e-6):
            return False
        age_s = time.perf_counter() - self._latest_intercept.source_capture_time
        return age_s <= self.MAX_READY_INTERCEPT_AGE_S

    def _handle_uart_event(self, evt: UartEvent) -> None:
        if not self._rally_enabled:
            super()._handle_uart_event(evt)
            return

        if evt.kind == UartEventKind.RAW_LINE and evt.line:
            self._state.last_uart_line = evt.line
            print(f"[UART][RX] {evt.line}")

        upper = evt.line.upper() if evt.line else ""
        if "STRIKE DONE -> HOME" in upper:
            self._returning_home = True
            self._state.robot_state = "RETURN_HOME"
            self._send_predictor(PredictorCommandKind.RESET)
            self._latest_intercept = None
            self._last_sent_intercept = None
        if "REACHED HOME" in upper:
            self._returning_home = False
            self._state.robot_at_home = True

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
            self._manual_home_requested = True
            self._state.robot_at_home = False
            self._state.intercept_inflight = False
        elif evt.kind == UartEventKind.TX_INTERCEPT:
            self._state.intercept_inflight = True
            self._state.robot_at_home = False
            self._returning_home = False
            is_update = bool(evt.data.get("is_update", False))
            if is_update:
                self._state.updates_sent += 1
            else:
                self._state.throws_sent += 1
            self._state.last_latency_ms = float(evt.data.get("latency_ms", 0.0))
            self._state.last_adjusted_ms = float(evt.data.get("adjusted_time_ms", 0.0))
        elif evt.kind == UartEventKind.INTERCEPT_DONE:
            self._state.intercept_inflight = False
            self._state.robot_state = "IDLE"
            self._state.robot_at_home = True
            self._returning_home = False
            self._last_sent_intercept = None
        elif evt.kind == UartEventKind.HOME_DONE:
            self._state.robot_at_home = True
            self._state.intercept_inflight = False
            self._state.robot_state = "IDLE"
            self._returning_home = False
            if self._manual_home_requested:
                self._send_predictor(PredictorCommandKind.RESET)
                self._latest_intercept = None
                self._last_sent_intercept = None
                self._last_sent_capture_time = -1.0
                self._manual_home_requested = False
        elif evt.kind in (UartEventKind.TARGET_REJECTED, UartEventKind.PLAN_FAILED):
            self._state.intercept_inflight = False
            self._latest_intercept = None
            self._last_sent_intercept = None
            self._returning_home = False
        elif evt.kind == UartEventKind.ROBOT_LATE:
            pass
        elif evt.kind == UartEventKind.UART_ERROR:
            _print(f"[uart] ERROR: {evt.data}")

    def _can_send_initial_intercept(self) -> bool:
        if not self._rally_enabled:
            return super()._can_send_initial_intercept()
        if not self._state.startup_homing_done:
            return False
        if not self._state.gate_on:
            return False
        if not self._state.robot_at_home:
            return False
        if self._state.robot_state != "IDLE":
            return False
        if self._state.intercept_inflight:
            return False
        if not self._intercept_common_ready():
            return False
        return self._intercept_is_fresh()

    def _can_send_update_intercept(self) -> bool:
        if not self._rally_enabled:
            return super()._can_send_update_intercept()
        if not self._state.gate_on:
            return False
        if not self._state.intercept_inflight:
            return False
        if not self._returning_home:
            return False
        if self._last_sent_intercept is None:
            return False
        if not self._intercept_common_ready():
            return False
        if self._latest_intercept is None:
            return False
        if self._latest_intercept.source_capture_time <= (self._last_sent_capture_time + 1e-6):
            return False

        dx = self._latest_intercept.x_mm - self._last_sent_intercept.x_mm
        dy = self._latest_intercept.y_mm - self._last_sent_intercept.y_mm
        dz = self._latest_intercept.z_mm - self._last_sent_intercept.z_mm
        dist = (dx * dx + dy * dy + dz * dz) ** 0.5
        return dist >= self.UPDATE_DISTANCE_MM

    def _draw_overlay(self):
        if self._latest_obs is None or self._latest_obs.left_frame is None:
            return
        vis = self._latest_obs.left_frame.copy()

        gate_txt = "ON" if self._state.gate_on else "OFF"
        rally_txt = "ON" if self._rally_enabled else "OFF"
        cv2.putText(
            vis,
            f"FPS:{self._fps:.0f} Gate:{gate_txt} Rally:{rally_txt} Robot:{self._state.robot_state}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0) if self._state.gate_on else (0, 0, 255),
            1,
        )
        cv2.putText(
            vis,
            (
                f"Home:{self._state.robot_at_home} Inflight:{self._state.intercept_inflight} "
                f"ReturnHome:{self._returning_home} Throws:{self._state.throws_sent} "
                f"Updates:{self._state.updates_sent}"
            ),
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
            age_ms = max(0.0, (time.perf_counter() - ip.source_capture_time) * 1000.0)
            cv2.putText(
                vis,
                (
                    f"Target(mm): X={ip.x_mm:+.0f} Y={ip.y_mm:+.0f} Z={ip.z_mm:+.0f} "
                    f"t={ip.intercept_time_s*1000:.0f}ms age={age_ms:.0f}ms"
                ),
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
        cv2.imshow("Game Day", cv2.resize(vis, (dw, dh)))

    def run(self) -> int:
        if self._rally_enabled:
            _print("[mode] Rally mode active: keeping predictor/live tracking through consecutive shots.")
        return super().run()


def main() -> int:
    parser = argparse.ArgumentParser(description="Rally-oriented game day pipeline")
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
    parser.add_argument(
        "--no-rally",
        action="store_true",
        help="Fall back to the original single-shot send gating inside game.py.",
    )
    args = parser.parse_args()

    if not args.port:
        parser.error("UART port required. Pass --port or set STM32_UART_PORT.")

    timeout = None if args.home_ack_timeout == 0.0 else args.home_ack_timeout

    app = GameDayApp(
        uart_port=args.port,
        baud_rate=args.baud,
        home_ack_timeout_s=timeout,
        warmup_s=max(0.0, args.warmup_s),
        tx_interval_s=max(0.0, args.tx_interval_ms / 1000.0),
        uart_verbose=not args.quiet_uart,
        rally_enabled=not args.no_rally,
    )
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
