#!/usr/bin/env python3
"""Interactive sender + live serial monitor for STM32 robot target messages."""

import argparse
import struct
import threading
import time
from typing import Optional

import serial


TARGET_MSG_FLOAT_COUNT = 7
DEFAULT_HOME = (0.0, 0.0, -1000.0)

ser: Optional[serial.Serial] = None
_stop_reader = threading.Event()
_print_lock = threading.Lock()


def safe_print(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def format_rx_chunk(data: bytes) -> str:
    # Prefer readable ASCII when bytes look text-like; otherwise show hex.
    if not data:
        return ""
    printable = sum(1 for b in data if b in (9, 10, 13) or 32 <= b <= 126)
    if printable >= int(0.85 * len(data)):
        text = data.decode("utf-8", errors="replace").replace("\r", "\\r").replace("\n", "\\n")
        return f"[RX-TXT] {text}"
    return f"[RX-HEX] {data.hex(' ').upper()}"


def reader_loop() -> None:
    assert ser is not None
    while not _stop_reader.is_set():
        try:
            if ser.in_waiting > 0:
                chunk = ser.read(ser.in_waiting)
                safe_print(format_rx_chunk(chunk))
            else:
                time.sleep(0.02)
        except serial.SerialException as exc:
            safe_print(f"[ERR] Serial read failed: {exc}")
            break


def send_message(msg_type: float, x: float, y: float, z: float, arrival_time: float) -> None:
    assert ser is not None
    msg = struct.pack("<7f", msg_type, x, y, z, arrival_time, 0.0, 0.0)
    ser.write(msg)
    ser.flush()
    safe_print(
        f"[TX] type={msg_type:.1f} x={x:.2f} y={y:.2f} z={z:.2f} t={arrival_time:.2f}s"
    )


def send_home(home_xyz: tuple[float, float, float]) -> None:
    send_message(3.0, home_xyz[0], home_xyz[1], home_xyz[2], 1.0)


def send_intercept(x: float, y: float, z: float, t_s: float) -> None:
    send_message(1.0, x, y, z, t_s)


def show_menu() -> None:
    safe_print("\n" + "=" * 54)
    safe_print("ROBOT TARGET SENDER + SERIAL MONITOR")
    safe_print("=" * 54)
    safe_print("1) Send HOME")
    safe_print("2) Send INTERCEPT (custom position)")
    safe_print("3) Send INTERCEPT (quick default)")
    safe_print("4) Exit")
    safe_print("=" * 54)


def main() -> int:
    global ser

    parser = argparse.ArgumentParser(description="Send target messages and monitor STM32 serial output.")
    parser.add_argument("--port", default="COM9", help="Serial port, e.g. COM9")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate for laptop/USB UART")
    args = parser.parse_args()

    safe_print(f"[INFO] Connecting to {args.port} @ {args.baud}...")
    ser = serial.Serial(args.port, args.baud, timeout=0.05)
    time.sleep(1.0)
    safe_print("[INFO] Connected.")

    reader = threading.Thread(target=reader_loop, daemon=True)
    reader.start()

    try:
        while True:
            show_menu()
            choice = input("Enter choice (1-4): ").strip()

            if choice == "1":
                send_home(DEFAULT_HOME)
            elif choice == "2":
                try:
                    x = float(input("  X position (mm) [default 50.0]: ") or "50.0")
                    y = float(input("  Y position (mm) [default -25.0]: ") or "-25.0")
                    z = float(input("  Z position (mm) [default -800.0]: ") or "-800.0")
                    t = float(input("  Arrival time (s) [default 0.5]: ") or "0.5")
                    send_intercept(x, y, z, t)
                except ValueError:
                    safe_print("[WARN] Invalid input: numbers only.")
            elif choice == "3":
                send_intercept(50.0, -25.0, -800.0, 0.5)
            elif choice == "4":
                safe_print("[INFO] Exiting.")
                break
            else:
                safe_print("[WARN] Invalid choice. Enter 1-4.")

            time.sleep(0.15)
    except KeyboardInterrupt:
        safe_print("\n[INFO] Interrupted by user.")
    finally:
        _stop_reader.set()
        time.sleep(0.1)
        if ser is not None:
            ser.close()
        safe_print("[INFO] Serial closed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
