"""Game-day variant of `game.py` with minimal rally support.

This file intentionally stays as close to `game.py` as possible. The only
behavioral differences in rally mode are:

1. When firmware reports `STRIKE DONE -> HOME`, reset the predictor so the next
   incoming ball starts from a clean trajectory state.
2. When the intercept cycle completes, do not send an extra software HOME.
   The firmware already auto-homes after strike, so we simply mark the robot as
   ready once that cycle finishes.
"""

from __future__ import annotations

import argparse
import os

from game import GameApp, _print
from pipeline.messages import PredictorCommandKind, UartEvent, UartEventKind, UartCommandKind


class GameDayApp(GameApp):
    """`game.py` behavior with only the minimal rally-specific differences."""

    MIN_RALLY_RESEND_DISTANCE_MM = 80.0

    def __init__(self, *args, rally_enabled: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._rally_enabled = bool(rally_enabled)
        self._last_sent_capture_time = -1.0
        self._rally_send_armed = False
        # Game-day rally mode is allowed to overwrite the single-slot STM32
        # mailbox even while the current move is active; tx_interval still
        # limits how often we transmit.
        self._uart_worker._allow_intercept_overwrite_while_moving = bool(rally_enabled)

    def _has_new_valid_intercept(self) -> bool:
        if not self._state.startup_homing_done:
            return False
        if not self._state.gate_on:
            return False
        if self._latest_intercept is None:
            return False
        if not self._intercept_common_ready():
            return False
        if self._latest_intercept.source_capture_time <= (self._last_sent_capture_time + 1e-6):
            return False
        if self._last_sent_intercept is None:
            return True

        dx = self._latest_intercept.x_mm - self._last_sent_intercept.x_mm
        dy = self._latest_intercept.y_mm - self._last_sent_intercept.y_mm
        dz = self._latest_intercept.z_mm - self._last_sent_intercept.z_mm
        dist = (dx * dx + dy * dy + dz * dz) ** 0.5
        return dist >= self.MIN_RALLY_RESEND_DISTANCE_MM

    def _handle_uart_event(self, evt: UartEvent) -> None:
        if not self._rally_enabled:
            super()._handle_uart_event(evt)
            return

        # Keep the same raw logging style as game.py for field debugging.
        if evt.line:
            self._state.last_uart_line = evt.line
            print(f"[UART][RX] {evt.line}")

        upper = evt.line.upper() if evt.line else ""
        if "STRIKE DONE -> HOME" in upper:
            # Old shot is over; clear trajectory state so the next valid
            # predictor output belongs to the next rally exchange.
            self._rally_send_armed = True
            self._send_predictor(PredictorCommandKind.RESET)
            self._latest_intercept = None
            self._last_sent_intercept = None

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
            self._rally_send_armed = False
            self._last_sent_capture_time = float(
                evt.data.get("source_capture_time", self._last_sent_capture_time)
            )
            is_update = bool(evt.data.get("is_update", False))
            if is_update:
                self._state.updates_sent += 1
            else:
                self._state.throws_sent += 1
            self._state.last_latency_ms = float(evt.data.get("latency_ms", 0.0))
            self._state.last_adjusted_ms = float(evt.data.get("adjusted_time_ms", 0.0))
        elif evt.kind == UartEventKind.INTERCEPT_DONE:
            # In rally mode the firmware already transitioned through strike and
            # auto-home, so do not send another HOME from software.
            if not self._state.intercept_inflight:
                return
            self._state.intercept_inflight = False
            self._state.robot_at_home = True
            self._state.robot_state = "IDLE"
            self._last_sent_intercept = None
        elif evt.kind == UartEventKind.HOME_DONE:
            if self._state.robot_at_home:
                return
            self._state.robot_at_home = True
            self._state.intercept_inflight = False
            self._state.robot_state = "IDLE"
            self._rally_send_armed = False
            self._send_predictor(PredictorCommandKind.RESET)
            self._latest_intercept = None
            self._last_sent_intercept = None
        elif evt.kind in (UartEventKind.TARGET_REJECTED, UartEventKind.PLAN_FAILED):
            self._state.intercept_inflight = False
            self._state.robot_at_home = True
            self._state.robot_state = "IDLE"
            self._rally_send_armed = True
            self._latest_intercept = None
            self._last_sent_intercept = None
        elif evt.kind == UartEventKind.UART_ERROR:
            _print(f"[uart] ERROR: {evt.data}")

    def _can_send_initial_intercept(self) -> bool:
        if not self._rally_enabled:
            return super()._can_send_initial_intercept()
        # First shot still behaves like game.py.
        if self._state.throws_sent == 0:
            if self._state.robot_at_home and not self._state.intercept_inflight:
                return self._has_new_valid_intercept()
            return False
        if self._rally_send_armed:
            return self._has_new_valid_intercept()
        return False

    def _can_send_update_intercept(self) -> bool:
        if not self._rally_enabled:
            return super()._can_send_update_intercept()
        return False

    def run(self) -> int:
        if self._rally_enabled:
            _print(
                "[mode] Rally mode active: game.py behavior plus one post-strike "
                "preloaded target when a new incoming ball is detected."
            )
        return super().run()


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal-rally game pipeline")
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
        help="Fall back to the original single-shot behavior inside game.py.",
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
