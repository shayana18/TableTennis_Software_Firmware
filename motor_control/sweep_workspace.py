#!/usr/bin/env python3
"""Trace the EE workspace perimeter at top and bottom with TARGET_TEST moves.

The script sends ellipse-perimeter points at:
  z = z_max (top loop)
  z = z_min (bottom loop)

Each point is sent as TARGET_TEST (type 4.0), so no strike/interception
velocity planning is required on the firmware side.
"""

from __future__ import annotations

import argparse
import math
import struct
import time
from dataclasses import dataclass
from typing import List

import serial


TARGET_MSG_FLOAT_COUNT = 10
TARGET_TYPE_HOME = 3.0
TARGET_TYPE_TEST = 4.0

DEFAULT_PORT = "COM9"
DEFAULT_BAUD = 115200

# From app/robot.h:
# ROBOT_EE_ELLIPSE_RADIUS_X = PADDLE_ELLIPSE_RADIUS_X - PADDLE_ARM_OFFSET
#                           = 790 - 206 = 584
# ROBOT_EE_ELLIPSE_RADIUS_Y = 540
# ROBOT_EE_LIMIT_POS_Z      = PADDLE_LIMIT_POS_Z - PADDLE_OFFSET_Z = -760 - (-50) = -710
# ROBOT_EE_LIMIT_NEG_Z      = PADDLE_LIMIT_NEG_Z - PADDLE_OFFSET_Z = -1020 - (-50) = -970
DEFAULT_RX = 584.0
DEFAULT_RY = 540.0
DEFAULT_Z_MIN = -970.0
DEFAULT_Z_MAX = -710.0

DEFAULT_HOME_XYZ = (0.0, 0.0, -900.0)


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    z: float


def generate_ellipse_loop(rx: float, ry: float, z: float, point_count: int, clockwise: bool) -> List[Point]:
    """Generate one perimeter loop around x^2/rx^2 + y^2/ry^2 = 1 at fixed z."""
    if point_count < 4:
        raise ValueError("points-per-loop must be >= 4")

    points: List[Point] = []
    for i in range(point_count):
        theta = (2.0 * math.pi * i) / point_count
        if clockwise:
            theta = -theta
        points.append(Point(x=rx * math.cos(theta), y=ry * math.sin(theta), z=z))
    return points


def generate_perimeter_points(
    rx: float,
    ry: float,
    z_min: float,
    z_max: float,
    points_per_loop: int,
) -> List[Point]:
    """Generate top and bottom ellipse perimeter loops."""
    top = generate_ellipse_loop(rx=rx, ry=ry, z=z_max, point_count=points_per_loop, clockwise=False)
    bottom = generate_ellipse_loop(rx=rx, ry=ry, z=z_min, point_count=points_per_loop, clockwise=True)
    return top + bottom


def pack_target_message(msg_type: float, point: Point, arrival_time_s: float) -> bytes:
    now_s = time.time()
    return struct.pack(
        f"<{TARGET_MSG_FLOAT_COUNT}f",
        msg_type,
        point.x,
        point.y,
        point.z,
        0.0,  # vx
        0.0,  # vy
        0.0,  # vz
        arrival_time_s,
        now_s,
        0.0,
    )


def send_target(ser: serial.Serial, msg_type: float, point: Point, arrival_time_s: float) -> None:
    msg = pack_target_message(msg_type, point, arrival_time_s)
    ser.write(msg)
    ser.flush()
    print(
        "[TX] "
        f"type={msg_type:.1f} "
        f"x={point.x:.1f} y={point.y:.1f} z={point.z:.1f} "
        f"t={arrival_time_s:.2f}s",
        flush=True,
    )


def wait_for_idle(ser: serial.Serial, timeout_s: float) -> bool:
    """Wait for firmware status line containing 'STATE: IDLE'."""
    deadline = time.monotonic() + timeout_s
    pending = bytearray()

    while time.monotonic() < deadline:
        chunk = ser.read(ser.in_waiting or 1)
        if not chunk:
            continue
        pending.extend(chunk)

        while b"\n" in pending:
            line_raw, _, rest = pending.partition(b"\n")
            pending = bytearray(rest)
            line = line_raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            print(f"[RX] {line}", flush=True)
            if "STATE: IDLE" in line:
                return True

    return False


def validate_positive(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send TARGET_TEST commands along top/bottom EE ellipse perimeters."
    )
    parser.add_argument("--port", default=DEFAULT_PORT, help="Serial port (e.g. COM9)")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Serial baud rate")
    parser.add_argument(
        "--points-per-loop",
        type=int,
        default=48,
        help="Number of points on each ellipse perimeter loop",
    )

    parser.add_argument("--rx", type=float, default=DEFAULT_RX, help="Workspace ellipse X radius (mm)")
    parser.add_argument("--ry", type=float, default=DEFAULT_RY, help="Workspace ellipse Y radius (mm)")
    parser.add_argument("--z-min", type=float, default=DEFAULT_Z_MIN, help="Workspace minimum Z (mm)")
    parser.add_argument("--z-max", type=float, default=DEFAULT_Z_MAX, help="Workspace maximum Z (mm)")

    parser.add_argument(
        "--arrival-time",
        type=float,
        default=1.2,
        help="Target arrival time sent in message (seconds)",
    )
    parser.add_argument(
        "--wait-s",
        type=float,
        default=1.3,
        help="Fixed wait after each command when not using --wait-for-idle",
    )
    parser.add_argument(
        "--wait-for-idle",
        action="store_true",
        help="Wait for serial status 'STATE: IDLE' after each point instead of fixed wait",
    )
    parser.add_argument(
        "--idle-timeout-s",
        type=float,
        default=8.0,
        help="Timeout for each --wait-for-idle cycle",
    )

    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="If >0, only send the first N generated points",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned points but do not send")
    parser.add_argument("--home-start", action="store_true", help="Send HOME before sweep (default: off)")
    parser.add_argument("--home-end", action="store_true", help="Send HOME after sweep (default: off)")

    args = parser.parse_args()

    try:
        validate_positive("arrival-time", args.arrival_time)
        validate_positive("wait-s", args.wait_s)
        validate_positive("idle-timeout-s", args.idle_timeout_s)
        validate_positive("rx", args.rx)
        validate_positive("ry", args.ry)
        if args.points_per_loop < 4:
            raise ValueError("points-per-loop must be >= 4")
    except ValueError as exc:
        print(f"[ERR] {exc}")
        return 2

    if args.z_min > args.z_max:
        print("[ERR] z-min must be <= z-max")
        return 2

    points = generate_perimeter_points(
        rx=args.rx,
        ry=args.ry,
        z_min=args.z_min,
        z_max=args.z_max,
        points_per_loop=args.points_per_loop,
    )

    if args.max_points > 0:
        points = points[: args.max_points]

    if not points:
        print("[ERR] No points generated. Check workspace limits and step sizes.")
        return 2

    print(
        "[INFO] "
        f"Generated {len(points)} perimeter points "
        f"(rx={args.rx:.1f}, ry={args.ry:.1f}, z=[{args.z_min:.1f},{args.z_max:.1f}], "
        f"points_per_loop={args.points_per_loop})"
    )

    if args.dry_run:
        preview_count = min(12, len(points))
        print(f"[INFO] Dry run preview (first {preview_count} points):")
        for idx, point in enumerate(points[:preview_count], start=1):
            print(f"  {idx:3d}: x={point.x:7.1f} y={point.y:7.1f} z={point.z:7.1f}")
        return 0

    home_point = Point(*DEFAULT_HOME_XYZ)

    try:
        print(f"[INFO] Connecting to {args.port} @ {args.baud}...")
        with serial.Serial(args.port, args.baud, timeout=0.05) as ser:
            time.sleep(1.0)
            print("[INFO] Connected.")

            def send_and_wait(msg_type: float, point: Point, label: str) -> None:
                if args.wait_for_idle:
                    ser.reset_input_buffer()
                send_target(ser, msg_type, point, args.arrival_time)

                if args.wait_for_idle:
                    if not wait_for_idle(ser, args.idle_timeout_s):
                        print(f"[WARN] Timeout waiting for IDLE after {label}")
                else:
                    time.sleep(args.wait_s)

            if args.home_start:
                print("[INFO] Moving to HOME before sweep...")
                send_and_wait(TARGET_TYPE_HOME, home_point, "home-start")

            for idx, point in enumerate(points, start=1):
                print(f"[INFO] Point {idx}/{len(points)}")
                send_and_wait(TARGET_TYPE_TEST, point, f"point {idx}")

            if args.home_end:
                print("[INFO] Returning to HOME...")
                send_and_wait(TARGET_TYPE_HOME, home_point, "home-end")

        print("[INFO] Sweep complete.")
        return 0

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        return 130
    except serial.SerialException as exc:
        print(f"[ERR] Serial error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
