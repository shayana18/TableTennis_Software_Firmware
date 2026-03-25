#!/usr/bin/env python3
"""Sweep workspace as spirals using TARGET_TEST commands.

Path order:
1) Spiral up from z-bottom to z-top
2) Spiral down from z-top to z-bottom
"""

from __future__ import annotations

import argparse
import math
import struct
import time
from dataclasses import dataclass
from typing import Iterable, List

import serial


TARGET_MSG_FLOAT_COUNT = 10
TARGET_TYPE_TEST = 4.0
TARGET_TYPE_HOME = 3.0


# Defaults from app/robot.h (current workspace model).
DEFAULT_PORT = "COM9"
DEFAULT_BAUD = 115200
DEFAULT_HOME = (0.0, 0.0, -900.0)
DEFAULT_RING_RX = 574.0   # ROBOT_EE_ELLIPSE_RADIUS_X = 780 - 206
DEFAULT_RING_RY = 540.0   # ROBOT_EE_ELLIPSE_RADIUS_Y
DEFAULT_Z_TOP = -710.0    # ROBOT_EE_LIMIT_POS_Z
DEFAULT_Z_BOTTOM = -970.0 # ROBOT_EE_LIMIT_NEG_Z


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    z: float


def pack_target_message(
    msg_type: float,
    point: Point,
    arrival_time_s: float,
    vx: float = 0.0,
    vy: float = 0.0,
    vz: float = 0.0,
) -> bytes:
    now_s = time.time()
    return struct.pack(
        "<10f",
        msg_type,
        point.x,
        point.y,
        point.z,
        vx,
        vy,
        vz,
        arrival_time_s,
        now_s,
        0.0,
    )


def send_target(
    ser: serial.Serial,
    msg_type: float,
    point: Point,
    arrival_time_s: float,
    vx: float = 0.0,
    vy: float = 0.0,
    vz: float = 0.0,
) -> None:
    payload = pack_target_message(msg_type, point, arrival_time_s, vx, vy, vz)
    ser.write(payload)
    ser.flush()
    print(
        "[TX] "
        f"type={msg_type:.1f} "
        f"x={point.x:7.1f} y={point.y:7.1f} z={point.z:7.1f} "
        f"t={arrival_time_s:.2f}s"
    )


def wait_for_idle(ser: serial.Serial, timeout_s: float) -> bool:
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
            print(f"[RX] {line}")
            if "STATE: IDLE" in line:
                return True

    return False


def build_ring_points(
    rx: float,
    ry: float,
    z: float,
    point_count: int,
    clockwise: bool,
) -> List[Point]:
    points: List[Point] = []
    for i in range(point_count):
        theta = (2.0 * math.pi * i) / point_count
        if clockwise:
            theta = -theta
        points.append(Point(rx * math.cos(theta), ry * math.sin(theta), z))
    return points


def build_spiral_points(
    rx: float,
    ry: float,
    z_start: float,
    z_end: float,
    turns: int,
    points_per_turn: int,
    clockwise: bool,
) -> List[Point]:
    total_points = turns * points_per_turn
    points: List[Point] = []
    for i in range(total_points + 1):
        progress = i / float(total_points)
        theta = (2.0 * math.pi * turns) * progress
        if clockwise:
            theta = -theta
        z = z_start + (z_end - z_start) * progress
        points.append(Point(rx * math.cos(theta), ry * math.sin(theta), z))
    return points


def build_sweep_points(
    rx: float,
    ry: float,
    z_top: float,
    z_bottom: float,
    points_per_turn: int,
    spiral_turns: int,
    edge_scale: float,
) -> Iterable[Point]:
    rx_use = rx * edge_scale
    ry_use = ry * edge_scale
    up = build_spiral_points(
        rx=rx_use,
        ry=ry_use,
        z_start=z_bottom,
        z_end=z_top,
        turns=spiral_turns,
        points_per_turn=points_per_turn,
        clockwise=False,
    )
    down = build_spiral_points(
        rx=rx_use,
        ry=ry_use,
        z_start=z_top,
        z_end=z_bottom,
        turns=spiral_turns,
        points_per_turn=points_per_turn,
        clockwise=True,
    )
    if up and down:
        down = down[1:]
    return up + down


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep workspace as bottom-up and top-down spirals with TARGET_TEST packets."
    )
    parser.add_argument("--port", default=DEFAULT_PORT, help="Serial port (e.g. COM9)")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Serial baud rate")
    parser.add_argument("--points-per-ring", type=int, default=36, help="Points per spiral turn")
    parser.add_argument("--points", type=int, default=0, help="Alias: points per spiral turn (overrides --points-per-ring)")
    parser.add_argument("--spiral-turns", type=int, default=4, help="Turns per spiral (up and down)")
    parser.add_argument("--arrival-time", type=float, default=0.70, help="Arrival time in packet (s)")
    parser.add_argument("--speed-scale", type=float, default=1.0, help="Global speed multiplier (>1 faster, <1 slower)")
    parser.add_argument("--min-arrival", type=float, default=0.12, help="Lower clamp for scaled arrival time (s)")
    parser.add_argument("--idle-timeout", type=float, default=6.0, help="Timeout waiting for STATE: IDLE (s)")
    parser.add_argument("--wait-s", type=float, default=0.8, help="Fallback fixed wait when --no-wait-idle is set")
    parser.add_argument("--min-wait-s", type=float, default=0.04, help="Lower clamp for scaled wait when --no-wait-idle is set (s)")
    parser.add_argument("--no-wait-idle", action="store_true", help="Do not wait for STATE: IDLE after each point")
    parser.add_argument("--rx", type=float, default=DEFAULT_RING_RX, help="Ellipse X radius (mm)")
    parser.add_argument("--ry", type=float, default=DEFAULT_RING_RY, help="Ellipse Y radius (mm)")
    parser.add_argument("--z-top", type=float, default=DEFAULT_Z_TOP, help="Top ring Z (mm)")
    parser.add_argument("--z-bottom", type=float, default=DEFAULT_Z_BOTTOM, help="Bottom ring Z (mm)")
    parser.add_argument("--edge-scale", type=float, default=0.92, help="Scale ring inside workspace boundary")
    parser.add_argument("--home-start", action="store_true", help="Send HOME before sweep")
    parser.add_argument("--home-end", action="store_true", help="Send HOME after sweep")
    parser.add_argument("--dry-run", action="store_true", help="Print points only")
    args = parser.parse_args()

    points_per_turn = args.points if args.points > 0 else args.points_per_ring
    if points_per_turn < 4:
        raise SystemExit("points-per-ring must be >= 4")
    if args.spiral_turns < 1:
        raise SystemExit("spiral-turns must be >= 1")
    if not (0.1 <= args.edge_scale <= 1.0):
        raise SystemExit("edge-scale must be in [0.1, 1.0]")
    if args.speed_scale <= 0.0:
        raise SystemExit("speed-scale must be > 0")
    if args.min_arrival <= 0.0:
        raise SystemExit("min-arrival must be > 0")
    if args.min_wait_s < 0.0:
        raise SystemExit("min-wait-s must be >= 0")

    effective_arrival = max(args.min_arrival, args.arrival_time / args.speed_scale)
    effective_wait_s = max(args.min_wait_s, args.wait_s / args.speed_scale)

    points = list(
        build_sweep_points(
            rx=args.rx,
            ry=args.ry,
            z_top=args.z_top,
            z_bottom=args.z_bottom,
            points_per_turn=points_per_turn,
            spiral_turns=args.spiral_turns,
            edge_scale=args.edge_scale,
        )
    )
    print(
        f"[INFO] Generated {len(points)} spiral points "
        f"(turns={args.spiral_turns}, points/turn={points_per_turn})."
    )
    print(
        f"[INFO] Speed settings: speed_scale={args.speed_scale:.2f} "
        f"arrival={effective_arrival:.3f}s wait={effective_wait_s:.3f}s"
    )

    if args.dry_run:
        for i, p in enumerate(points[:12], start=1):
            print(f"{i:3d}: x={p.x:7.1f} y={p.y:7.1f} z={p.z:7.1f}")
        return 0

    home = Point(*DEFAULT_HOME)
    wait_idle = not args.no_wait_idle

    try:
        print(f"[INFO] Connecting to {args.port} @ {args.baud}...")
        with serial.Serial(args.port, args.baud, timeout=0.05) as ser:
            time.sleep(1.0)
            print("[INFO] Connected.")

            def send_and_wait(msg_type: float, point: Point, tag: str) -> None:
                if wait_idle:
                    ser.reset_input_buffer()
                send_target(ser, msg_type, point, effective_arrival)
                if wait_idle:
                    if not wait_for_idle(ser, args.idle_timeout):
                        print(f"[WARN] Timeout waiting for IDLE after {tag}")
                else:
                    time.sleep(effective_wait_s)

            if args.home_start:
                send_and_wait(TARGET_TYPE_HOME, home, "home-start")

            for idx, point in enumerate(points, start=1):
                print(f"[INFO] Spiral point {idx}/{len(points)}")
                send_and_wait(TARGET_TYPE_TEST, point, f"spiral-point-{idx}")

            if args.home_end:
                send_and_wait(TARGET_TYPE_HOME, home, "home-end")

        print("[INFO] Workspace ring sweep complete.")
        return 0
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
        return 130
    except serial.SerialException as exc:
        print(f"[ERR] Serial error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
