#!/usr/bin/env python3
"""Send arbitrary interception points to exercise strike planning logic."""

from __future__ import annotations

import argparse
import math
import random
import struct
import time
from dataclasses import dataclass
from typing import List

import serial


TARGET_MSG_FLOAT_COUNT = 10
TARGET_TYPE_INTERCEPT = 1.0
TARGET_TYPE_HOME = 3.0


DEFAULT_PORT = "COM9"
DEFAULT_BAUD = 115200
DEFAULT_HOME = (0.0, 0.0, -900.0)

# Conservative defaults from app/robot.h relationships.
DEFAULT_INTERCEPT_RX = 700.0
DEFAULT_INTERCEPT_RY = 460.0
DEFAULT_INTERCEPT_Z_MIN = -1020.0
DEFAULT_INTERCEPT_Z_MAX = -760.0
DEFAULT_STRIKE_TARGET_Z = -1150.0
DEFAULT_GRAVITY = 9810.0


@dataclass(frozen=True)
class InterceptSample:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    t_arrival_s: float


def pack_message(sample: InterceptSample) -> bytes:
    now_s = time.time()
    return struct.pack(
        "<10f",
        TARGET_TYPE_INTERCEPT,
        sample.x,
        sample.y,
        sample.z,
        sample.vx,
        sample.vy,
        sample.vz,
        sample.t_arrival_s,
        now_s,
        0.0,
    )


def send_intercept(ser: serial.Serial, sample: InterceptSample) -> None:
    payload = pack_message(sample)
    ser.write(payload)
    ser.flush()
    print(
        "[TX] "
        f"x={sample.x:7.1f} y={sample.y:7.1f} z={sample.z:7.1f} "
        f"vx={sample.vx:7.1f} vy={sample.vy:7.1f} vz={sample.vz:7.1f} "
        f"t={sample.t_arrival_s:.2f}s"
    )


def send_home(ser: serial.Serial, arrival_time_s: float) -> None:
    now_s = time.time()
    payload = struct.pack(
        "<10f",
        TARGET_TYPE_HOME,
        DEFAULT_HOME[0],
        DEFAULT_HOME[1],
        DEFAULT_HOME[2],
        0.0,
        0.0,
        0.0,
        arrival_time_s,
        now_s,
        0.0,
    )
    ser.write(payload)
    ser.flush()
    print(f"[TX] HOME t={arrival_time_s:.2f}s")


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


def valid_strike_sample(sample: InterceptSample, strike_target_z: float, gravity: float) -> bool:
    disc = (sample.vz * sample.vz) + (2.0 * gravity * (sample.z - strike_target_z))
    if disc <= 0.0:
        return False
    det = math.sqrt(disc)
    ball_return_time = (sample.vz + det) / gravity
    return ball_return_time > 0.0


def random_sample(
    rng: random.Random,
    rx: float,
    ry: float,
    z_min: float,
    z_max: float,
    t_min: float,
    t_max: float,
) -> InterceptSample:
    # Uniform-in-area ellipse sampling.
    theta = rng.uniform(0.0, 2.0 * math.pi)
    radius = math.sqrt(rng.uniform(0.0, 1.0)) * 0.90
    x = rx * radius * math.cos(theta)
    y = ry * radius * math.sin(theta)
    z = rng.uniform(z_min + 8.0, z_max - 8.0)

    vx = rng.uniform(-1200.0, 1200.0)
    vy = rng.uniform(-600.0, 1000.0)
    vz = rng.uniform(-1200.0, 300.0)
    t_arrival_s = rng.uniform(t_min, t_max)

    return InterceptSample(x, y, z, vx, vy, vz, t_arrival_s)


def generate_samples(
    count: int,
    seed: int,
    rx: float,
    ry: float,
    z_min: float,
    z_max: float,
    t_min: float,
    t_max: float,
    strike_target_z: float,
    gravity: float,
) -> List[InterceptSample]:
    rng = random.Random(seed)
    out: List[InterceptSample] = []
    attempts = 0
    max_attempts = count * 60

    while len(out) < count and attempts < max_attempts:
        attempts += 1
        sample = random_sample(rng, rx, ry, z_min, z_max, t_min, t_max)
        if valid_strike_sample(sample, strike_target_z, gravity):
            out.append(sample)

    if len(out) < count:
        raise RuntimeError(
            f"Only generated {len(out)} valid samples after {attempts} attempts. "
            "Try reducing count or adjusting workspace/velocity bounds."
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send random TARGET_INTERCEPT packets to showcase strike planning."
    )
    parser.add_argument("--port", default=DEFAULT_PORT, help="Serial port (e.g. COM9)")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Serial baud rate")
    parser.add_argument("--count", type=int, default=12, help="Number of interception samples")
    parser.add_argument("--seed", type=int, default=430, help="Random seed for repeatability")
    parser.add_argument("--arrival-min", type=float, default=0.70, help="Min interception time in packet (s)")
    parser.add_argument("--arrival-max", type=float, default=1.20, help="Max interception time in packet (s)")
    parser.add_argument("--idle-timeout", type=float, default=8.0, help="Timeout waiting for STATE: IDLE (s)")
    parser.add_argument("--no-wait-idle", action="store_true", help="Do not wait for STATE: IDLE")
    parser.add_argument("--wait-s", type=float, default=2.8, help="Fixed wait when --no-wait-idle is used")
    parser.add_argument("--home-start", action="store_true", help="Send HOME before test")
    parser.add_argument("--home-end", action="store_true", help="Send HOME after test")
    parser.add_argument("--home-time", type=float, default=2.0, help="Arrival time for HOME packets (s)")
    parser.add_argument("--rx", type=float, default=DEFAULT_INTERCEPT_RX, help="Interception ellipse X radius (mm)")
    parser.add_argument("--ry", type=float, default=DEFAULT_INTERCEPT_RY, help="Interception ellipse Y radius (mm)")
    parser.add_argument("--z-min", type=float, default=DEFAULT_INTERCEPT_Z_MIN, help="Interception min Z (mm)")
    parser.add_argument("--z-max", type=float, default=DEFAULT_INTERCEPT_Z_MAX, help="Interception max Z (mm)")
    parser.add_argument("--strike-target-z", type=float, default=DEFAULT_STRIKE_TARGET_Z, help="Strike target Z used by firmware")
    parser.add_argument("--gravity", type=float, default=DEFAULT_GRAVITY, help="Gravity used by firmware (mm/s^2)")
    parser.add_argument("--dry-run", action="store_true", help="Generate and print samples only")
    args = parser.parse_args()

    if args.count <= 0:
        raise SystemExit("count must be > 0")
    if args.arrival_min <= 0.0 or args.arrival_max <= 0.0 or args.arrival_min > args.arrival_max:
        raise SystemExit("arrival-min/max must be > 0 and min <= max")
    if args.z_min >= args.z_max:
        raise SystemExit("z-min must be < z-max")

    samples = generate_samples(
        count=args.count,
        seed=args.seed,
        rx=args.rx,
        ry=args.ry,
        z_min=args.z_min,
        z_max=args.z_max,
        t_min=args.arrival_min,
        t_max=args.arrival_max,
        strike_target_z=args.strike_target_z,
        gravity=args.gravity,
    )
    print(f"[INFO] Generated {len(samples)} valid interception samples.")

    if args.dry_run:
        for i, s in enumerate(samples, start=1):
            print(
                f"{i:3d}: x={s.x:7.1f} y={s.y:7.1f} z={s.z:7.1f} "
                f"vx={s.vx:7.1f} vy={s.vy:7.1f} vz={s.vz:7.1f} t={s.t_arrival_s:.2f}"
            )
        return 0

    wait_idle = not args.no_wait_idle

    try:
        print(f"[INFO] Connecting to {args.port} @ {args.baud}...")
        with serial.Serial(args.port, args.baud, timeout=0.05) as ser:
            time.sleep(1.0)
            print("[INFO] Connected.")

            def wait_after_send(tag: str) -> None:
                if wait_idle:
                    if not wait_for_idle(ser, args.idle_timeout):
                        print(f"[WARN] Timeout waiting for IDLE after {tag}")
                else:
                    time.sleep(args.wait_s)

            if args.home_start:
                ser.reset_input_buffer()
                send_home(ser, args.home_time)
                wait_after_send("home-start")

            for idx, sample in enumerate(samples, start=1):
                print(f"[INFO] Intercept sample {idx}/{len(samples)}")
                ser.reset_input_buffer()
                send_intercept(ser, sample)
                wait_after_send(f"sample-{idx}")

            if args.home_end:
                ser.reset_input_buffer()
                send_home(ser, args.home_time)
                wait_after_send("home-end")

        print("[INFO] Strike showcase complete.")
        return 0
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
        return 130
    except serial.SerialException as exc:
        print(f"[ERR] Serial error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

