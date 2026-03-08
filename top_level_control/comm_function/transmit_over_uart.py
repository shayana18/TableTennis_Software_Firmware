"""
UART helpers for sending robot targets from laptop to STM32.

This module packs messages exactly how `mailbox_parse_target()` expects them on the
STM32 side: a raw little-endian array of 7 floats:

    [type, x_mm, y_mm, z_mm, intercept_time_s, time_sent_s, timestamp_s]

It also provides a simple line-based status reader for ASCII debug/status messages
coming back from the STM32 (for example home completion confirmation).
"""

from __future__ import annotations

import struct
import time
from typing import List, Optional

try:
    import serial
except ImportError as exc:  # pragma: no cover - depends on local environment
    serial = None
    _SERIAL_IMPORT_ERROR = exc
else:
    _SERIAL_IMPORT_ERROR = None


TARGET_MSG_FLOAT_COUNT = 7
TARGET_INTERCEPT = 1
TARGET_HOME = 3

_PACKET_STRUCT = struct.Struct("<7f")
_PLANNER_TERMINAL_PREFIX = "from planner in terminal"
_HOME_ACK_TOKENS = (
    "STATE: PLAN",  # User-defined startup gate: start game once FSM enters PLAN after HOME command.
    "HOMED",
    "REACHED HOME",
    "COMPLETED Q",  # Current firmware prints this on move completion.
)


class UartComm:
    """
    Simple UART transport for target messages and status lines.

    The TX direction is binary floats (STM32 mailbox format).
    The RX direction is treated as ASCII status/debug text for easy debugging.
    """

    def __init__(
        self,
        port: str,
        baud_rate: int = 115200,
        timeout_s: float = 0.0,
        verbose: bool = True,
    ) -> None:
        self.port = port
        self.baud_rate = int(baud_rate)
        self.timeout_s = float(timeout_s)
        self.verbose = bool(verbose)

        self._ser = None
        self._rx_buffer = bytearray()

    @staticmethod
    def _local_print(*args, **kwargs) -> None:
        """Print a local planner-side message (not UART-received payload text)."""
        print(_PLANNER_TERMINAL_PREFIX, *args, **kwargs)

    @property
    def is_open(self) -> bool:
        """Return True when the serial port is open."""
        return bool(self._ser is not None and self._ser.is_open)

    def open(self) -> None:
        """Open the serial port."""
        if serial is None:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "pyserial is not installed. Install it with `pip install pyserial`."
            ) from _SERIAL_IMPORT_ERROR

        if self.is_open:
            return

        self._ser = serial.Serial(
            port=self.port,
            baudrate=self.baud_rate,
            timeout=self.timeout_s,
        )

        if self.verbose:
            self._local_print(f"[UART] Opened {self.port} @ {self.baud_rate} baud")

    def close(self) -> None:
        """Close the serial port."""
        if self._ser is None:
            return
        try:
            if self._ser.is_open:
                self._ser.close()
        finally:
            if self.verbose:
                self._local_print("[UART] Closed")
            self._ser = None

    def clear_input_buffer(self) -> None:
        """
        Clear any pending STM32 output before starting a handshake.

        This avoids treating stale boot logs (or an old completion print) as a new ACK.
        """
        self._rx_buffer.clear()
        if self.is_open:
            self._ser.reset_input_buffer()

    def _write_packet(self, values: List[float]) -> None:
        """Pack and send a 7-float packet."""
        if not self.is_open:
            raise RuntimeError("UART port is not open")
        if len(values) != TARGET_MSG_FLOAT_COUNT:
            raise ValueError(f"Expected {TARGET_MSG_FLOAT_COUNT} floats, got {len(values)}")

        payload = _PACKET_STRUCT.pack(*values)
        self._ser.write(payload)
        self._ser.flush()

    def send_target(
        self,
        target_type: int,
        x_mm: float = 0.0,
        y_mm: float = 0.0,
        z_mm: float = 0.0,
        intercept_time_s: float = 0.0,
        time_sent_s: Optional[float] = None,
        timestamp_s: Optional[float] = None,
    ) -> List[float]:
        """
        Send a `target_t` packet to the STM32 and return the float payload for logging.

        Args:
            target_type: Numeric target enum (`TARGET_INTERCEPT` or `TARGET_HOME`)
            x_mm, y_mm, z_mm: Target position in millimeters
            intercept_time_s: Arrival time horizon in seconds
            time_sent_s: Laptop monotonic timestamp when packet is sent
            timestamp_s: Laptop monotonic timestamp of source frame/measurement
        """
        now = time.perf_counter() if time_sent_s is None else float(time_sent_s)
        timestamp = now if timestamp_s is None else float(timestamp_s)

        values = [
            float(target_type),
            float(x_mm),
            float(y_mm),
            float(z_mm),
            float(intercept_time_s),
            float(now),
            float(timestamp),
        ]
        self._write_packet(values)
        return values

    def send_home(self) -> List[float]:
        """
        Send a `TARGET_HOME` command.

        Position/time fields are zeroed because the firmware only needs `type`.
        """
        values = self.send_target(target_type=TARGET_HOME)
        if self.verbose:
            self._local_print("[UART][TX] TARGET_HOME")
        return values

    def send_intercept(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        intercept_time_s: float,
        time_sent_s: float,
        timestamp_s: float,
    ) -> List[float]:
        """Send a `TARGET_INTERCEPT` command with position/time metadata."""
        values = self.send_target(
            target_type=TARGET_INTERCEPT,
            x_mm=x_mm,
            y_mm=y_mm,
            z_mm=z_mm,
            intercept_time_s=intercept_time_s,
            time_sent_s=time_sent_s,
            timestamp_s=timestamp_s,
        )
        if self.verbose:
            self._local_print(
                "[UART][TX] TARGET_INTERCEPT "
                f"x={x_mm:.1f} y={y_mm:.1f} z={z_mm:.1f} mm "
                f"t={intercept_time_s*1000:.1f} ms "
                f"sent={time_sent_s:.6f} frame_ts={timestamp_s:.6f}"
            )
        return values

    def poll_status_lines(self) -> List[str]:
        """
        Read any available UART RX bytes and return complete ASCII lines.

        Incomplete trailing bytes are buffered until a newline arrives.
        """
        if not self.is_open:
            return []

        chunks = bytearray()

        try:
            waiting = int(self._ser.in_waiting)
        except Exception:
            waiting = 0

        if waiting > 0:
            chunks.extend(self._ser.read(waiting))
        else:
            chunk = self._ser.read(256)
            if chunk:
                chunks.extend(chunk)

        if not chunks:
            return []

        self._rx_buffer.extend(chunks)
        raw_lines = self._rx_buffer.split(b"\n")
        self._rx_buffer = bytearray(raw_lines.pop() if raw_lines else b"")

        lines: List[str] = []
        for raw in raw_lines:
            line = raw.decode("utf-8", errors="ignore").strip("\r\n\t \x00")
            if line:
                lines.append(line)
        return lines

    @staticmethod
    def is_home_confirmation(line: str) -> bool:
        """Return True if an incoming status line should count as home-complete ACK."""
        upper = line.upper()
        return any(token in upper for token in _HOME_ACK_TOKENS)

    def wait_for_home_confirmation(
        self,
        timeout_s: Optional[float] = 30.0,
        poll_sleep_s: float = 0.01,
    ) -> Optional[str]:
        """
        Wait for a home completion message from the STM32.

        Returns:
            The matching status line, or None on timeout.
        """
        deadline = None if timeout_s is None else (time.perf_counter() + float(timeout_s))

        if self.verbose:
            if timeout_s is None:
                self._local_print("[UART] Waiting for home confirmation (no timeout)...")
            else:
                self._local_print(f"[UART] Waiting for home confirmation (timeout {timeout_s:.1f}s)...")

        while True:
            for line in self.poll_status_lines():
                if self.verbose:
                    print(f"[UART][RX] {line}")
                if self.is_home_confirmation(line):
                    return line

            if deadline is not None and time.perf_counter() >= deadline:
                return None

            time.sleep(poll_sleep_s)

    def print_pending_status(self) -> None:
        """Drain and print any pending UART status lines (debug helper)."""
        for line in self.poll_status_lines():
            print(f"[UART][RX] {line}")
