from __future__ import annotations

import queue
import threading
import time

from comm_functions.transmit_over_uart import UartComm

from .messages import (
    UartCommand,
    UartCommandKind,
    UartEvent,
    UartEventKind,
)


def _push_latest(q: queue.Queue, item) -> None:
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass


class UartWorker(threading.Thread):
    """UART transport + firmware status parser.

    Owns:
    - Serial open/close
    - Binary target packet writes
    - Status line parsing from robot_runtime_send_status()
    """

    def __init__(
        self,
        *,
        port: str,
        baud_rate: int,
        verbose: bool,
        uart_cmd_queue: queue.Queue,
        uart_event_queue: queue.Queue,
        stop_event: threading.Event,
        tx_interval_s: float = 0.03,
        time_aggression: float = 1.0,
        status_printer=print,
    ) -> None:
        super().__init__(name="UartWorker", daemon=True)
        self._uart = UartComm(port=port, baud_rate=baud_rate, verbose=verbose)
        self._cmd_queue = uart_cmd_queue
        self._evt_queue = uart_event_queue
        self._stop_event = stop_event
        self._tx_interval_s = max(0.0, float(tx_interval_s))
        self._time_aggression = float(time_aggression)
        self._print = status_printer
        self._last_tx_time = 0.0
        self._pending_action = None  # 'home' | 'intercept' | None
        self._stm32_moving = False
        self._allow_intercept_overwrite_while_moving = False

    def _emit(self, kind: UartEventKind, line: str | None = None, **data) -> None:
        _push_latest(
            self._evt_queue,
            UartEvent(kind=kind, timestamp=time.perf_counter(), line=line, data=data),
        )

    def _parse_line(self, line: str) -> None:
        # Normalize once so matching is robust to case differences.
        upper = line.upper()
        self._emit(UartEventKind.RAW_LINE, line=line)

        # Firmware prints "COMPLETED Q: ..." at move completion in current branch.
        # We emit this explicitly for parity with legacy integration scripts.
        if "COMPLETED Q" in upper:
            self._emit(UartEventKind.COMPLETED_Q, line=line)
            if self._pending_action == "home":
                self._pending_action = None
                self._stm32_moving = False
                self._emit(UartEventKind.HOME_DONE, line=line)
            elif self._pending_action == "intercept":
                self._pending_action = None
                self._stm32_moving = False
                self._emit(UartEventKind.INTERCEPT_DONE, line=line)

        if "STATE: OFF" in upper:
            self._emit(UartEventKind.STATE_OFF, line=line)
        if "STATE: PLAN" in upper:
            self._emit(UartEventKind.STATE_PLAN, line=line)
        if "STATE: MOVE" in upper:
            self._stm32_moving = True
            self._emit(UartEventKind.STATE_MOVE, line=line)
        if "STRIKE DONE -> HOME" in upper:
            # Strike motion has completed and firmware is transitioning into
            # the auto-home phase. Treat this as the end of the "moving"
            # window so game-day rally code may send the next candidate target.
            self._stm32_moving = False
        if "STATE: IDLE" in upper:
            self._emit(UartEventKind.STATE_IDLE, line=line)
            # Keep IDLE-based completion as a fallback in case firmware branch
            # does not emit COMPLETED Q for a particular move type.
            if self._pending_action == "home":
                self._pending_action = None
                self._stm32_moving = False
                self._emit(UartEventKind.HOME_DONE, line=line)
            elif self._pending_action == "intercept":
                self._pending_action = None
                self._stm32_moving = False
                self._emit(UartEventKind.INTERCEPT_DONE, line=line)

        if "TARGET OUT OF WORKSPACE" in upper:
            self._pending_action = None
            self._stm32_moving = False
            self._emit(UartEventKind.TARGET_REJECTED, line=line)
        if (
            "PLANNING FAILED" in upper
            or "PLAN FAILED" in upper
            or "PLAN_ABORT" in upper
            or "STRIKE SWEEP OUT OF WORKSPACE" in upper
            or "STRIKE SWEEP TOO SHORT" in upper
        ):
            self._pending_action = None
            self._stm32_moving = False
            self._emit(UartEventKind.PLAN_FAILED, line=line)
        if "ROBOT WILL BE LATE" in upper:
            self._emit(UartEventKind.ROBOT_LATE, line=line)
        if "FAULT" in upper:
            self._emit(UartEventKind.FAULT, line=line)

    def _send_home(self) -> None:
        self._uart.send_home()
        self._last_tx_time = time.perf_counter()
        self._pending_action = "home"
        self._emit(UartEventKind.TX_HOME)

    def _send_intercept(self, cmd: UartCommand, *, is_update: bool) -> None:
        if cmd.intercept is None:
            return

        # Match legacy integration behavior: enforce minimum transmit interval.
        now = time.perf_counter()
        if (now - self._last_tx_time) < self._tx_interval_s:
            return

        # End-to-end latency compensation:
        # subtract elapsed time between camera capture and just-before-UART-send
        # from predictor's intercept horizon.
        latency_s = max(0.0, now - cmd.intercept.source_capture_time)
        adjusted_t = max(
            0.0, (cmd.intercept.intercept_time_s - latency_s) * self._time_aggression
        )

        self._uart.send_intercept(
            x_mm=cmd.intercept.x_mm,
            y_mm=cmd.intercept.y_mm,
            z_mm=cmd.intercept.z_mm,
            vx_mm_s=cmd.intercept.vx_mm_s,
            vy_mm_s=cmd.intercept.vy_mm_s,
            vz_mm_s=cmd.intercept.vz_mm_s,
            intercept_time_s=adjusted_t,
            time_sent_s=now,
            timestamp_s=cmd.intercept.source_capture_time,
        )
        self._print("INTERCEPTION TARGET SENT")
        self._last_tx_time = now
        if self._pending_action is None:
            self._pending_action = "intercept"
            self._stm32_moving = False
        self._emit(
            UartEventKind.TX_INTERCEPT,
            latency_ms=latency_s * 1000.0,
            adjusted_time_ms=adjusted_t * 1000.0,
            frame_id=cmd.intercept.source_frame_id,
            source_capture_time=cmd.intercept.source_capture_time,
            clamped=cmd.intercept.clamped,
            confidence=cmd.intercept.confidence,
            is_update=is_update,
        )

    def _handle_command(self, cmd: UartCommand) -> None:
        if cmd.kind == UartCommandKind.SHUTDOWN:
            self._stop_event.set()
            return
        if cmd.kind == UartCommandKind.HOME:
            self._send_home()
            return
        if cmd.kind == UartCommandKind.INTERCEPT:
            # Legacy behavior: allow target updates while still in PLAN
            # (pending intercept exists, but robot has not entered MOVE yet).
            if self._pending_action is None:
                self._send_intercept(cmd, is_update=False)
                return
            if (
                self._pending_action == "intercept"
                and (
                    self._allow_intercept_overwrite_while_moving
                    or not self._stm32_moving
                )
            ):
                self._send_intercept(cmd, is_update=True)

    def run(self) -> None:
        try:
            self._uart.open()
            self._uart.clear_input_buffer()
            self._emit(UartEventKind.UART_OPENED)

            while not self._stop_event.is_set():
                try:
                    cmd: UartCommand = self._cmd_queue.get_nowait()
                except queue.Empty:
                    cmd = None
                if cmd is not None:
                    try:
                        self._handle_command(cmd)
                    except Exception as exc:
                        self._emit(UartEventKind.UART_ERROR, error=str(exc))

                for line in self._uart.poll_status_lines():
                    try:
                        self._parse_line(str(line))
                    except Exception as exc:
                        # Never let a malformed/unexpected status line stop
                        # the UART thread; report and continue.
                        self._emit(UartEventKind.UART_ERROR, line=str(line), error=f"parse_line: {exc}")

                time.sleep(0.002)
        except Exception as exc:
            self._emit(UartEventKind.UART_ERROR, error=str(exc))
            self._print(f"[uart] ERROR: {exc}")
        finally:
            try:
                self._uart.close()
            except Exception:
                pass
            self._emit(UartEventKind.UART_CLOSED)
