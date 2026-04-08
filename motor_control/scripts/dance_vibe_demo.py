#!/usr/bin/env python3
"""Fast jerk routine:
1) side-to-side fast
2) up/down fast
3) move forward, stop, wave paddle
4) then quick forward/back at the end
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
TARGET_TYPE_INTERCEPT = 1.0
TARGET_TYPE_HOME = 3.0
TARGET_TYPE_TEST = 4.0

DEFAULT_PORT = "COM9"
DEFAULT_BAUD = 115200
DEFAULT_BPM = 185.0

# Workspace defaults from app/robot.h relationships.
DEFAULT_WS_RX = 574.0
DEFAULT_WS_RY = 540.0
DEFAULT_WS_Z_MIN = -970.0
DEFAULT_WS_Z_MAX = -710.0
DEFAULT_HOME = (0.0, 0.0, -900.0)


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Step:
    msg_type: float
    beats: float
    point: Point
    tag: str
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0


def pack_message(
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


def send_step(ser: serial.Serial, step: Step, arrival_time_s: float) -> None:
    payload = pack_message(
        msg_type=step.msg_type,
        point=step.point,
        arrival_time_s=arrival_time_s,
        vx=step.vx,
        vy=step.vy,
        vz=step.vz,
    )
    ser.write(payload)
    ser.flush()

    typ = (
        "HOME"
        if step.msg_type == TARGET_TYPE_HOME
        else "INT"
        if step.msg_type == TARGET_TYPE_INTERCEPT
        else "TEST"
    )
    print(
        "[TX] "
        f"{step.tag:12s} {typ:4s} "
        f"x={step.point.x:7.1f} y={step.point.y:7.1f} z={step.point.z:7.1f} "
        f"vx={step.vx:7.1f} vy={step.vy:7.1f} vz={step.vz:7.1f} "
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


def inside_workspace(point: Point, rx: float, ry: float, z_min: float, z_max: float) -> bool:
    if point.z < z_min or point.z > z_max:
        return False
    ellipse = (point.x * point.x) / (rx * rx) + (point.y * point.y) / (ry * ry)
    return ellipse <= 1.0


def build_side_to_side(
    center: Point,
    swings: int,
    x_amp: float,
    beats: float,
) -> List[Step]:
    out: List[Step] = []
    for i in range(swings):
        x = center.x + (x_amp if (i % 2 == 0) else -x_amp)
        out.append(
            Step(
                msg_type=TARGET_TYPE_TEST,
                beats=beats,
                point=Point(x=x, y=center.y, z=center.z),
                tag=f"side-{i+1:02d}",
            )
        )
    return out


def build_up_down(
    center: Point,
    pulses: int,
    z_amp: float,
    beats: float,
) -> List[Step]:
    out: List[Step] = []
    for i in range(pulses):
        z = center.z + (z_amp if (i % 2 == 0) else -z_amp)
        out.append(
            Step(
                msg_type=TARGET_TYPE_TEST,
                beats=beats,
                point=Point(x=center.x, y=center.y, z=z),
                tag=f"updn-{i+1:02d}",
            )
        )
    return out


def build_forward_stop_and_wave(
    center: Point,
    forward_y: float,
    forward_beats: float,
    stop_beats: float,
    wave_steps: int,
    wave_beats: float,
    wave_vx_amp: float,
    wave_vy: float,
    wave_vz: float,
    end_fb_beats: float,
    end_fb_cycles: int,
) -> List[Step]:
    out: List[Step] = []
    forward_point = Point(center.x, center.y + forward_y, center.z)

    # Move forward quickly.
    out.append(
        Step(
            msg_type=TARGET_TYPE_TEST,
            beats=forward_beats,
            point=forward_point,
            tag="forward",
        )
    )
    # Stop/hold.
    out.append(
        Step(
            msg_type=TARGET_TYPE_TEST,
            beats=stop_beats,
            point=forward_point,
            tag="stop",
        )
    )

    # Paddle wave via intercept packets at fixed forward stance.
    for i in range(wave_steps):
        phase = (2.0 * math.pi * i) / float(wave_steps)
        swing = math.sin(phase)
        vx = wave_vx_amp * swing
        out.append(
            Step(
                msg_type=TARGET_TYPE_INTERCEPT,
                beats=wave_beats,
                point=forward_point,
                tag=f"wave-{i+1:02d}",
                vx=vx,
                vy=wave_vy,
                vz=wave_vz,
            )
        )

    # End sequence: quick forward/back after paddle wave.
    if end_fb_cycles > 0:
        out.append(Step(msg_type=TARGET_TYPE_TEST, beats=end_fb_beats, point=center, tag="end-back"))
        for i in range(end_fb_cycles):
            out.append(Step(msg_type=TARGET_TYPE_TEST, beats=end_fb_beats, point=forward_point, tag=f"end-fwd-{i+1:02d}"))
            out.append(Step(msg_type=TARGET_TYPE_TEST, beats=end_fb_beats, point=center, tag=f"end-back-{i+1:02d}"))
    return out


def build_routine(center: Point) -> List[Step]:
    steps: List[Step] = []
    steps.extend(
        build_side_to_side(
            center=Point(center.x, center.y, center.z - 12.0),
            swings=42,
            x_amp=255.0,
            beats=0.11,
        )
    )
    steps.extend(
        build_up_down(
            center=Point(center.x, center.y, center.z - 8.0),
            pulses=34,
            z_amp=70.0,
            beats=0.10,
        )
    )
    steps.extend(
        build_forward_stop_and_wave(
            center=Point(center.x, center.y, center.z - 8.0),
            forward_y=260.0,
            forward_beats=0.30,
            stop_beats=0.42,
            wave_steps=20,
            wave_beats=0.17,
            wave_vx_amp=1300.0,
            wave_vy=180.0,
            wave_vz=-300.0,
            end_fb_beats=0.12,
            end_fb_cycles=10,
        )
    )
    return steps


def send_home(ser: serial.Serial, home_time: float) -> None:
    send_step(
        ser=ser,
        step=Step(msg_type=TARGET_TYPE_HOME, beats=0.0, point=Point(*DEFAULT_HOME), tag="home"),
        arrival_time_s=home_time,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast jerk dance routine")
    parser.add_argument("--port", default=DEFAULT_PORT, help="Serial port (e.g. COM9)")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Serial baud rate")
    parser.add_argument("--bpm", type=float, default=DEFAULT_BPM, help="Beat tempo")
    parser.add_argument("--start-delay", type=float, default=1.5, help="Seconds before first move")
    parser.add_argument("--home-start", action="store_true", help="Send HOME before dance")
    parser.add_argument("--home-end", action="store_true", help="Send HOME after dance")
    parser.add_argument("--home-time", type=float, default=2.0, help="Arrival time for HOME packets")
    parser.add_argument("--routine-loops", type=int, default=0, help="How many times to loop routine (0 = infinite)")
    parser.add_argument("--no-wait-idle", action="store_true", help="Do not wait for STATE: IDLE")
    parser.add_argument("--idle-timeout", type=float, default=6.0, help="STATE: IDLE timeout")
    parser.add_argument("--arrival-scale", type=float, default=0.55, help="Arrival time fraction of step duration")
    parser.add_argument("--min-test-arrival", type=float, default=0.10, help="Minimum arrival for TEST steps")
    parser.add_argument("--min-intercept-arrival", type=float, default=0.32, help="Minimum arrival for INTERCEPT steps")
    parser.add_argument("--no-wait-guard", type=float, default=0.02, help="Extra sleep in --no-wait-idle mode")
    parser.add_argument("--center-x", type=float, default=0.0, help="Routine center X (mm)")
    parser.add_argument("--center-y", type=float, default=0.0, help="Routine center Y (mm)")
    parser.add_argument("--center-z", type=float, default=-860.0, help="Routine center Z (mm)")
    parser.add_argument("--ws-rx", type=float, default=DEFAULT_WS_RX, help="Workspace ellipse radius X (mm)")
    parser.add_argument("--ws-ry", type=float, default=DEFAULT_WS_RY, help="Workspace ellipse radius Y (mm)")
    parser.add_argument("--ws-z-min", type=float, default=DEFAULT_WS_Z_MIN, help="Workspace minimum Z (mm)")
    parser.add_argument("--ws-z-max", type=float, default=DEFAULT_WS_Z_MAX, help="Workspace maximum Z (mm)")
    parser.add_argument("--dry-run", action="store_true", help="Preview routine and exit")
    args = parser.parse_args()

    if args.bpm <= 0.0:
        raise SystemExit("bpm must be > 0")
    if args.arrival_scale <= 0.0 or args.arrival_scale > 1.0:
        raise SystemExit("arrival-scale must be in (0, 1]")
    if args.ws_rx <= 0.0 or args.ws_ry <= 0.0:
        raise SystemExit("workspace radii must be > 0")
    if args.ws_z_min >= args.ws_z_max:
        raise SystemExit("ws-z-min must be < ws-z-max")
    if args.routine_loops < 0:
        raise SystemExit("routine-loops must be >= 0")

    center = Point(args.center_x, args.center_y, args.center_z)
    routine = build_routine(center)

    for idx, step in enumerate(routine, start=1):
        if not inside_workspace(step.point, args.ws_rx, args.ws_ry, args.ws_z_min, args.ws_z_max):
            raise SystemExit(
                f"Step {idx} ({step.tag}) outside workspace: "
                f"x={step.point.x:.1f}, y={step.point.y:.1f}, z={step.point.z:.1f}"
            )

    if args.routine_loops == 0:
        loop_desc = "infinite"
    else:
        loop_desc = str(args.routine_loops)
    print(
        f"[INFO] Generated {len(routine)} base steps "
        "(side-fast, up/down-fast, forward-stop-wave, end forward/back). "
        f"Loops={loop_desc}."
    )
    if args.dry_run:
        for i, s in enumerate(routine[:36], start=1):
            typ = "INT" if s.msg_type == TARGET_TYPE_INTERCEPT else "TEST"
            print(
                f"{i:3d}: {s.tag:10s} {typ:4s} beats={s.beats:4.2f} "
                f"x={s.point.x:7.1f} y={s.point.y:7.1f} z={s.point.z:7.1f} "
                f"vx={s.vx:7.1f} vy={s.vy:7.1f} vz={s.vz:7.1f}"
            )
        return 0

    beat_s = 60.0 / args.bpm
    wait_idle = not args.no_wait_idle

    try:
        print(f"[INFO] Connecting to {args.port} @ {args.baud}...")
        with serial.Serial(args.port, args.baud, timeout=0.05) as ser:
            time.sleep(1.0)
            print("[INFO] Connected.")

            if args.home_start:
                if wait_idle:
                    ser.reset_input_buffer()
                send_home(ser, args.home_time)
                if wait_idle:
                    wait_for_idle(ser, max(args.idle_timeout, args.home_time + 1.0))
                else:
                    time.sleep(args.home_time)

            if args.start_delay > 0.0:
                print(f"[INFO] Start music now. Routine starts in {args.start_delay:.1f}s...")
                time.sleep(args.start_delay)

            loop_i = 0
            step_count = 0
            while True:
                loop_i += 1
                print(f"[INFO] Routine loop {loop_i}")
                for idx, step in enumerate(routine, start=1):
                    step_count += 1
                    step_duration = step.beats * beat_s
                    if step.msg_type == TARGET_TYPE_INTERCEPT:
                        arrival_time_s = max(args.min_intercept_arrival, step_duration * args.arrival_scale)
                    else:
                        arrival_time_s = max(args.min_test_arrival, step_duration * args.arrival_scale)

                    print(f"[INFO] Step {idx}/{len(routine)} ({step.tag})")
                    if wait_idle:
                        ser.reset_input_buffer()
                    send_step(ser, step, arrival_time_s)

                    if wait_idle:
                        timeout = max(args.idle_timeout, arrival_time_s + 1.0)
                        if not wait_for_idle(ser, timeout):
                            print(f"[WARN] Timeout waiting for IDLE after {step.tag}")
                    else:
                        time.sleep(max(step_duration, arrival_time_s + args.no_wait_guard))

                if args.routine_loops > 0 and loop_i >= args.routine_loops:
                    break

            if args.home_end:
                if wait_idle:
                    ser.reset_input_buffer()
                send_home(ser, args.home_time)
                if wait_idle:
                    wait_for_idle(ser, max(args.idle_timeout, args.home_time + 1.0))
                else:
                    time.sleep(args.home_time)

        print("[INFO] Routine complete.")
        return 0
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
        return 130
    except serial.SerialException as exc:
        print(f"[ERR] Serial error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
