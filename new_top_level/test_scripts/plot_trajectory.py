"""
Record live throws, review raw robot-frame points against RobotPredictor,
and optionally save approved throws as CSV files.

Workflow:
  1. Warm up stereo background subtraction.
  2. Record one throw using the same start/stop pattern as the live test scripts.
  3. Replay the raw robot-frame points through RobotPredictor.
  4. Plot raw points plus predicted trajectory in 3D, XY, XZ, and YZ.
  5. In the review plot, press one key to save, skip, or quit.
  6. Automatically return to live capture for the next throw.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
NEW_TOP_LEVEL_DIR = SCRIPT_DIR.parent
CAMERA_PROPERTIES_DIR = NEW_TOP_LEVEL_DIR / "camera_params" / "camera_properties"
DEFAULT_OUTPUT_DIR = NEW_TOP_LEVEL_DIR / "test_data" / "triangulation_cvs"

if str(NEW_TOP_LEVEL_DIR) not in sys.path:
    sys.path.insert(0, str(NEW_TOP_LEVEL_DIR))

from ball_tracking.stereo_triangulator import StereoTriangulator
from comm_functions.points_based_transform import cam_to_robot, load_points_based_transform
from comm_functions.transmit_over_uart import UartComm
from config.camera_config import load_camera_settings
from estimation.robot_predictor import RobotPredictor
from estimation.workspace import MAX_BOUNCES, clamp_to_workspace, in_workspace


DEFAULT_BAUD = 115200
REPROJ_ERR_MAX_DEFAULT = 10.0
WARMUP_S_DEFAULT = 2.0
LOST_FRAMES_TO_STOP_DEFAULT = 35
STOP_NEAR_WORKSPACE_MM_DEFAULT = 120.0
MIN_SEND_BUFFER = 5
TIME_AGGRESSION = 1.0
UPDATE_DISTANCE_MM = 80.0
MIN_CONSECUTIVE_POINTS_TO_START = 3
MIN_THROW_POINTS_TO_REVIEW = 3

CSV_COLUMNS = [
    "throw_id",
    "point_type",
    "point_index",
    "time_s",
    "time_from_start_s",
    "x_mm",
    "y_mm",
    "z_mm",
    "predictor_sample_idx",
    "predictor_start_time_s",
    "intercept_x_mm",
    "intercept_y_mm",
    "intercept_z_mm",
    "vx_mm_s",
    "vy_mm_s",
    "vz_mm_s",
    "bounces",
    "accepted",
    "reproj_err_px",
    "disparity_px",
    "clamped",
]


def _print(*args, **kwargs):
    print("[plot-trajectory]", *args, **kwargs)


@dataclass
class LiveRobotController:
    uart: UartComm
    tx_interval_s: float
    time_aggression: float
    startup_homed: bool = False
    intercept_inflight: bool = False
    pending_action: str | None = None
    stm32_moving: bool = False
    last_tx_time: float = 0.0
    last_cmd: dict[str, float] | None = None
    last_status: str = "DISABLED"
    throws_sent: int = 0
    updates_sent: int = 0
    last_latency_ms: float = 0.0
    last_adjusted_ms: float = 0.0


def _next_throw_csv_path(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "throw_"
    suffix = ".csv"
    max_idx = 0
    for name in os.listdir(output_dir):
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        idx = name[len(prefix):-len(suffix)]
        if idx.isdigit():
            max_idx = max(max_idx, int(idx))
    return output_dir / f"{prefix}{max_idx + 1:03d}.csv"


def _send_robot_home(robot: LiveRobotController) -> None:
    robot.uart.send_home()
    robot.pending_action = "home"
    robot.intercept_inflight = False
    robot.stm32_moving = False
    robot.last_cmd = None
    robot.last_status = "HOMING"
    _print("[robot] HOME sent")


def _init_robot_controller(
    port: str,
    baud_rate: int,
    tx_interval_s: float,
    home_ack_timeout_s: float | None,
    uart_verbose: bool,
) -> LiveRobotController:
    uart = UartComm(port=port, baud_rate=baud_rate, verbose=uart_verbose)
    robot = LiveRobotController(
        uart=uart,
        tx_interval_s=tx_interval_s,
        time_aggression=TIME_AGGRESSION,
    )

    uart.open()
    uart.clear_input_buffer()
    _send_robot_home(robot)
    ack = uart.wait_for_home_confirmation(timeout_s=home_ack_timeout_s)
    if ack is None:
        raise RuntimeError("Robot home confirmation timed out.")

    robot.startup_homed = True
    robot.pending_action = None
    robot.intercept_inflight = False
    robot.stm32_moving = False
    robot.last_status = "READY"
    _print(f"[robot] Home confirmed: {ack}")
    return robot


def _drain_robot_status(robot: LiveRobotController) -> None:
    for line in robot.uart.poll_status_lines():
        print(f"[UART][RX] {line}")
        upper = line.upper()

        if "STATE: OFF" in upper:
            robot.last_status = "OFF"
        if "STATE: PLAN" in upper:
            robot.last_status = "PLAN"
            robot.startup_homed = True
        if "STATE: MOVE" in upper:
            robot.last_status = "MOVE"
            robot.stm32_moving = True
        if "STATE: IDLE" in upper:
            robot.last_status = "IDLE"
            robot.stm32_moving = False
            if robot.pending_action == "home":
                robot.pending_action = None
                robot.intercept_inflight = False
                robot.last_cmd = None
                _print("[robot] Home complete.")
            elif robot.pending_action == "intercept":
                robot.pending_action = None
                robot.intercept_inflight = False
                robot.last_cmd = None
                _print("[robot] Intercept move complete.")

        if "COMPLETED Q" in upper:
            if robot.pending_action == "home":
                robot.pending_action = None
                robot.intercept_inflight = False
                robot.stm32_moving = False
                robot.last_cmd = None
                robot.last_status = "READY"
                _print("[robot] Home complete.")
            elif robot.pending_action == "intercept":
                robot.pending_action = None
                robot.intercept_inflight = False
                robot.stm32_moving = False
                robot.last_cmd = None
                robot.last_status = "READY"
                _print("[robot] Intercept move complete.")

        if "TARGET OUT OF WORKSPACE" in upper:
            robot.pending_action = None
            robot.intercept_inflight = False
            robot.stm32_moving = False
            robot.last_cmd = None
            robot.last_status = "REJECTED"
            _print("[robot] Firmware rejected target.")
        if "PLANNING FAILED" in upper or "PLAN_ABORT" in upper:
            robot.pending_action = None
            robot.intercept_inflight = False
            robot.stm32_moving = False
            robot.last_cmd = None
            robot.last_status = "PLAN_FAIL"
            _print("[robot] Firmware planning failed.")
        if "ROBOT WILL BE LATE" in upper:
            robot.last_status = "LATE"
            _print("[robot] Firmware warned robot will be late.")


def _maybe_send_live_intercept(
    robot: LiveRobotController,
    predictor: RobotPredictor,
    intercept: dict[str, float | int | bool] | None,
    frame_ts: float,
) -> bool:
    if intercept is None or not robot.startup_homed:
        return False

    if not robot.intercept_inflight and len(predictor.positions) < MIN_SEND_BUFFER:
        return False

    if predictor.velocity is not None and predictor._bounce_count > 0:
        if predictor.velocity[2] > 0:
            return False

    now = time.perf_counter()
    if (now - robot.last_tx_time) < robot.tx_interval_s:
        return False

    is_update = False
    if robot.intercept_inflight:
        if robot.stm32_moving or robot.pending_action != "intercept":
            return False
        if robot.last_cmd is not None:
            dx = float(intercept["x"]) - float(robot.last_cmd["x"])
            dy = float(intercept["y"]) - float(robot.last_cmd["y"])
            dz = float(intercept["z"]) - float(robot.last_cmd["z"])
            dist = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            if dist < UPDATE_DISTANCE_MM:
                return False
        is_update = True

    time_sent = now
    latency_s = max(0.0, time_sent - frame_ts)
    adjusted_t = max(
        0.0,
        (float(intercept["time"]) - latency_s) * robot.time_aggression,
    )

    robot.uart.send_intercept(
        x_mm=float(intercept["x"]),
        y_mm=float(intercept["y"]),
        z_mm=float(intercept["z"]),
        vx_mm_s=float(intercept.get("vx", 0.0)),
        vy_mm_s=float(intercept.get("vy", 0.0)),
        vz_mm_s=float(intercept.get("vz", 0.0)),
        intercept_time_s=adjusted_t,
        time_sent_s=time_sent,
        timestamp_s=frame_ts,
    )

    robot.last_tx_time = time_sent
    robot.last_cmd = {
        "x": float(intercept["x"]),
        "y": float(intercept["y"]),
        "z": float(intercept["z"]),
    }
    robot.last_latency_ms = latency_s * 1000.0
    robot.last_adjusted_ms = adjusted_t * 1000.0

    if is_update:
        robot.updates_sent += 1
        _print(
            f"[robot][update #{robot.updates_sent}] "
            f"x={float(intercept['x']):+.0f} "
            f"y={float(intercept['y']):+.0f} "
            f"z={float(intercept['z']):+.0f} "
            f"t={adjusted_t * 1000.0:.0f}ms"
        )
        return False
    else:
        robot.throws_sent += 1
        robot.intercept_inflight = True
        robot.pending_action = "intercept"
        robot.stm32_moving = False
        _print(
            f"[robot][throw #{robot.throws_sent}] "
            f"x={float(intercept['x']):+.0f} "
            f"y={float(intercept['y']):+.0f} "
            f"z={float(intercept['z']):+.0f} "
            f"t={adjusted_t * 1000.0:.0f}ms"
        )
        return True


def _robot_status_lines(robot: LiveRobotController | None) -> list[str]:
    if robot is None:
        return ["Robot: disabled"]

    action = robot.pending_action or "none"
    motion = "MOVE" if robot.stm32_moving else "WAIT"
    return [
        f"Robot: {robot.last_status} action={action} motion={motion}",
        (
            f"Robot TX: throws={robot.throws_sent} updates={robot.updates_sent} "
            f"lat={robot.last_latency_ms:.0f}ms adj={robot.last_adjusted_ms:.0f}ms"
        ),
    ]


def _warmup_background(
    triangulator: StereoTriangulator,
    frame_width: int,
    frame_height: int,
    warmup_s: float,
    window_name: str,
) -> bool:
    if warmup_s <= 0.0:
        return True

    _print("Remove ball. Learning background...")
    t0 = time.time()
    while time.time() - t0 < warmup_s:
        if not triangulator.cap_left.grab():
            continue
        if not triangulator.cap_right.grab():
            continue

        ok_left, frame_left = triangulator.cap_left.retrieve()
        ok_right, frame_right = triangulator.cap_right.retrieve()
        if not ok_left or not ok_right or frame_left is None or frame_right is None:
            continue

        triangulator.build_background(frame_left, frame_right)

        vis = cv2.resize(frame_left, (640, int(640 * frame_height / frame_width)))
        progress = min((time.time() - t0) / warmup_s, 1.0)
        h = vis.shape[0]
        bar_w = int(progress * (vis.shape[1] - 40))
        cv2.rectangle(vis, (20, h - 30), (20 + bar_w, h - 15), (0, 255, 255), -1)
        cv2.putText(
            vis,
            f"BG: {progress * 100:.0f}%",
            (20, h - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
        )
        cv2.imshow(window_name, vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False

    _print("Background ready.")
    return True


def _wait_while_paused(window_name: str, paused_frame: np.ndarray) -> bool:
    """Show a frozen paused frame until resumed or quit.

    Returns True if the user requested quit, else False when resumed.
    """
    while True:
        display = paused_frame.copy()
        pause_lines = [
            "PAUSED",
            "Press 'p' to resume",
            "Press 'q' to quit",
        ]
        for i, text in enumerate(pause_lines):
            y = 40 + i * 28
            cv2.putText(
                display,
                text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8 if i == 0 else 0.6,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                display,
                text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8 if i == 0 else 0.6,
                (0, 0, 0),
                1,
            )
        cv2.imshow(window_name, display)
        key = cv2.waitKeyEx(50)
        if 0 <= key <= 255:
            key = ord(chr(key).lower())
        if key == ord("p"):
            return False
        if key == ord("q"):
            return True


def _robot_to_pixel(
    triangulator: StereoTriangulator,
    rotation: np.ndarray,
    translation: np.ndarray,
    cam_scale: float,
    x_mm: float,
    y_mm: float,
    z_mm: float,
) -> tuple[int, int] | None:
    p_robot = np.array([x_mm, y_mm, z_mm], dtype=float)
    p_cam = rotation.T @ (p_robot - translation) / cam_scale
    uv = triangulator.project_to_image(
        (float(p_cam[0]), float(p_cam[1]), float(p_cam[2])),
        camera="left",
    )
    if uv is None:
        return None
    return int(round(float(uv[0]))), int(round(float(uv[1])))


def _is_near_workspace(
    x_mm: float,
    y_mm: float,
    z_mm: float,
    margin_mm: float,
) -> bool:
    if in_workspace(x_mm, y_mm, z_mm):
        return True
    _, _, _, clamp_dist = clamp_to_workspace(x_mm, y_mm, z_mm)
    return clamp_dist <= margin_mm


def _append_throw_sample(
    *,
    rows: list[dict[str, object]],
    frame_idx: int,
    last_ts: float | None,
    start_ts: float,
    frame_ts: float,
    cx: float,
    cy: float,
    cz: float,
    rx: float,
    ry: float,
    rz: float,
    reproj: float,
    disparity: float | None,
    live_predictor: RobotPredictor,
    live_intercept: dict[str, float | int | bool] | None,
    robot_controller: LiveRobotController | None,
    stop_near_workspace_mm: float,
) -> tuple[int, float, dict[str, float | int | bool] | None, bool]:
    dt = None if last_ts is None else (frame_ts - last_ts)
    last_ts = frame_ts
    frame_idx += 1

    rows.append(
        {
            "frame": frame_idx,
            "time_s": round(frame_ts, 6),
            "time_from_start_s": round(frame_ts - start_ts, 6),
            "dt_s": round(dt, 6) if dt is not None else None,
            "cam_x": round(cx, 6),
            "cam_y": round(cy, 6),
            "cam_z": round(cz, 6),
            "rob_x": round(rx, 6),
            "rob_y": round(ry, 6),
            "rob_z": round(rz, 6),
            "disparity": round(float(disparity), 6) if disparity is not None else None,
            "reproj_err": round(reproj, 6),
        }
    )

    accepted = live_predictor.add_position(rx, ry, rz, frame_ts)
    next_intercept = live_intercept
    if accepted and live_predictor.is_ready():
        next_intercept = live_predictor.predict_intercept()
        if robot_controller is not None:
            _maybe_send_live_intercept(
                robot_controller,
                live_predictor,
                next_intercept,
                frame_ts,
            )

    should_stop = _is_near_workspace(rx, ry, rz, stop_near_workspace_mm)
    return frame_idx, last_ts, next_intercept, should_stop


def _draw_live_prediction_overlay(
    frame: np.ndarray,
    triangulator: StereoTriangulator,
    rotation: np.ndarray,
    translation: np.ndarray,
    cam_scale: float,
    predictor: RobotPredictor,
    intercept: dict[str, float | int | bool] | None,
) -> None:
    state = predictor._get_prediction_state()
    if state is None or not predictor.positions or predictor.velocity is None:
        return

    span = predictor.positions[-1][3] - predictor.positions[0][3]
    if span < predictor.MIN_TIME_SPAN:
        return

    points = _simulate_trajectory(state)
    prev_px = None
    for point in points[::2]:
        px = _robot_to_pixel(
            triangulator,
            rotation,
            translation,
            cam_scale,
            float(point["x"]),
            float(point["y"]),
            float(point["z"]),
        )
        if px is None:
            prev_px = None
            continue

        if prev_px is not None:
            cv2.line(frame, prev_px, px, (0, 165, 255), 2)
        cv2.circle(frame, px, 2, (0, 165, 255), -1)
        prev_px = px

    start_px = _robot_to_pixel(
        triangulator,
        rotation,
        translation,
        cam_scale,
        float(state[0]),
        float(state[1]),
        float(state[2]),
    )
    if start_px is not None:
        cv2.circle(frame, start_px, 6, (0, 165, 255), -1)
        cv2.circle(frame, start_px, 8, (0, 0, 0), 1)

    if intercept is not None:
        target_px = _robot_to_pixel(
            triangulator,
            rotation,
            translation,
            cam_scale,
            float(intercept["x"]),
            float(intercept["y"]),
            float(intercept["z"]),
        )
        if target_px is not None:
            cv2.circle(frame, target_px, 10, (0, 255, 255), -1)
            cv2.circle(frame, target_px, 12, (0, 0, 0), 1)
            cv2.putText(
                frame,
                "pred",
                (target_px[0] + 14, target_px[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
            )


def _capture_one_throw(
    *,
    triangulator: StereoTriangulator,
    rotation: np.ndarray,
    translation: np.ndarray,
    cam_scale: float,
    frame_width: int,
    frame_height: int,
    reproj_err_max: float,
    lost_frames_to_stop: int,
    stop_near_workspace_mm: float,
    window_name: str,
    robot_controller: LiveRobotController | None,
) -> tuple[list[dict[str, object]], str, bool]:
    rows: list[dict[str, object]] = []
    throw_started = False
    candidate_samples: list[dict[str, float | None]] = []
    lost_frames = 0
    last_ts: float | None = None
    start_ts: float | None = None
    frame_idx = 0
    stop_reason = ""
    quit_requested = False
    last_status_print = time.perf_counter()
    last_frame_ts: float | None = None
    fps_smoothed: float | None = None
    live_predictor = RobotPredictor()
    live_intercept: dict[str, float | int | bool] | None = None

    if robot_controller is not None and not robot_controller.intercept_inflight:
        robot_controller.last_cmd = None

    _print("Ready. Throw one ball. Press 'p' to pause, 'h' to home, or 'q' to stop.")

    while True:
        if robot_controller is not None:
            _drain_robot_status(robot_controller)

        result = triangulator.update()
        frame_ts = result.get("capture_time", time.perf_counter())
        valid_sample = False

        if last_frame_ts is not None and frame_ts > last_frame_ts:
            inst_fps = 1.0 / (frame_ts - last_frame_ts)
            if fps_smoothed is None:
                fps_smoothed = inst_fps
            else:
                fps_smoothed = 0.85 * fps_smoothed + 0.15 * inst_fps
        last_frame_ts = frame_ts

        if result["left_frame"] is None or result["right_frame"] is None:
            continue

        left_vis, right_vis = triangulator.draw_results(result)

        if result["found_3d"]:
            reproj = float(result.get("reproj_err", 0.0))
            if reproj <= reproj_err_max:
                valid_sample = True
                cx, cy, cz = result["position_3d"]
                rx, ry, rz = cam_to_robot(rotation, translation, cam_scale, cx, cy, cz)

                sample = {
                    "time_s": frame_ts,
                    "cam_x": cx,
                    "cam_y": cy,
                    "cam_z": cz,
                    "rob_x": rx,
                    "rob_y": ry,
                    "rob_z": rz,
                    "disparity": (
                        float(result.get("disparity", 0.0))
                        if result.get("disparity") is not None
                        else None
                    ),
                    "reproj_err": reproj,
                }

                if not throw_started:
                    candidate_samples.append(sample)
                    if len(candidate_samples) >= MIN_CONSECUTIVE_POINTS_TO_START:
                        throw_started = True
                        start_ts = float(candidate_samples[0]["time_s"])
                        _print("Throw detected. Recording started.")

                        for pending in candidate_samples:
                            frame_idx, last_ts, live_intercept, should_stop = _append_throw_sample(
                                rows=rows,
                                frame_idx=frame_idx,
                                last_ts=last_ts,
                                start_ts=start_ts,
                                frame_ts=float(pending["time_s"]),
                                cx=float(pending["cam_x"]),
                                cy=float(pending["cam_y"]),
                                cz=float(pending["cam_z"]),
                                rx=float(pending["rob_x"]),
                                ry=float(pending["rob_y"]),
                                rz=float(pending["rob_z"]),
                                reproj=float(pending["reproj_err"]),
                                disparity=(
                                    float(pending["disparity"])
                                    if pending["disparity"] is not None
                                    else None
                                ),
                                live_predictor=live_predictor,
                                live_intercept=live_intercept,
                                robot_controller=robot_controller,
                                stop_near_workspace_mm=stop_near_workspace_mm,
                            )
                            if should_stop:
                                stop_reason = "near_workspace"
                                _print(
                                    "Stopping capture when measured ball reached workspace neighborhood."
                                )
                                break
                        candidate_samples.clear()
                        if stop_reason == "near_workspace":
                            break
                        lost_frames = 0
                else:
                    frame_idx, last_ts, live_intercept, should_stop = _append_throw_sample(
                        rows=rows,
                        frame_idx=frame_idx,
                        last_ts=last_ts,
                        start_ts=float(start_ts),
                        frame_ts=frame_ts,
                        cx=cx,
                        cy=cy,
                        cz=cz,
                        rx=rx,
                        ry=ry,
                        rz=rz,
                        reproj=reproj,
                        disparity=(
                            float(result.get("disparity", 0.0))
                            if result.get("disparity") is not None
                            else None
                        ),
                        live_predictor=live_predictor,
                        live_intercept=live_intercept,
                        robot_controller=robot_controller,
                        stop_near_workspace_mm=stop_near_workspace_mm,
                    )
                    if should_stop:
                        stop_reason = "near_workspace"
                        _print(
                            "Stopping capture when measured ball reached workspace neighborhood."
                        )
                        break
                    lost_frames = 0
            else:
                result["reject_reason"] = f"reproj({reproj:.1f}px)"

        if not throw_started and not valid_sample:
            candidate_samples.clear()

        if throw_started and not valid_sample:
            lost_frames += 1
            if lost_frames >= lost_frames_to_stop:
                stop_reason = "lost_after_throw"
                _print("Throw ended after tracking was lost.")
                break

        status_lines = [
            f"FPS: {fps_smoothed:.1f}" if fps_smoothed is not None else "FPS: --",
            f"Rows: {len(rows)}",
            (
                f"Recording: YES"
                if throw_started
                else f"Waiting: {len(candidate_samples)}/{MIN_CONSECUTIVE_POINTS_TO_START}"
            ),
            f"Lost frames: {lost_frames}/{lost_frames_to_stop}",
            f"Predictor: {'READY' if live_predictor.is_ready() else 'BUILDING'}",
            f"Stop near WS: {stop_near_workspace_mm:.0f}mm",
            "Keys: p=pause h=home q=quit",
        ]
        status_lines.extend(_robot_status_lines(robot_controller))
        if result.get("reject_reason"):
            status_lines.append(f"Reject: {result['reject_reason']}")

        _draw_live_prediction_overlay(
            left_vis,
            triangulator,
            rotation,
            translation,
            cam_scale,
            live_predictor,
            live_intercept,
        )

        for i, text in enumerate(status_lines):
            y = 30 + i * 24
            cv2.putText(
                left_vis,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                left_vis,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
            )

        left_vis = cv2.resize(
            left_vis, (640, int(640 * left_vis.shape[0] / left_vis.shape[1]))
        )
        right_vis = cv2.resize(
            right_vis, (640, int(640 * right_vis.shape[0] / right_vis.shape[1]))
        )
        live_view = np.hstack((left_vis, right_vis))
        cv2.imshow(window_name, live_view)

        now = time.perf_counter()
        if fps_smoothed is not None and now - last_status_print >= 1.0:
            _print(
                f"FPS:{fps_smoothed:.1f} "
                f"Rows:{len(rows)} "
                f"Recording:{'YES' if throw_started else 'WAITING'} "
                f"Lost:{lost_frames}/{lost_frames_to_stop} "
                f"Predictor:{'READY' if live_predictor.is_ready() else 'BUILDING'}"
            )
            last_status_print = now

        key = cv2.waitKeyEx(1)
        if 0 <= key <= 255:
            key = ord(chr(key).lower())
        if key == ord("p"):
            _print("Paused live capture.")
            quit_requested = _wait_while_paused(window_name, live_view)
            if quit_requested:
                stop_reason = "manual_quit"
                _print("Capture stop requested.")
                break
            _print("Resumed live capture.")
            last_status_print = time.perf_counter()
            last_frame_ts = None
            continue
        if key == ord("h") and robot_controller is not None:
            try:
                _send_robot_home(robot_controller)
            except Exception as exc:
                _print(f"[robot] HOME failed: {exc}")
            continue
        if key == ord("q"):
            stop_reason = "manual_quit"
            quit_requested = True
            _print("Capture stop requested.")
            break

    if rows and len(rows) < MIN_THROW_POINTS_TO_REVIEW:
        _print(
            f"Ignoring tiny fragment throw ({len(rows)} point"
            f"{'' if len(rows) == 1 else 's'})."
        )
        return [], "fragment_ignored", quit_requested

    return rows, stop_reason, quit_requested


def _simulate_trajectory(
    state: tuple[float, float, float, float, float, float],
    duration_s: float = RobotPredictor.SCAN_DURATION,
    step_s: float = RobotPredictor.SCAN_DT,
) -> list[dict[str, float | int]]:
    x, y, z, vx, vy, vz = [float(v) for v in state]
    t = 0.0
    bounces = 0
    points: list[dict[str, float | int]] = []

    while t <= duration_s + 1e-9:
        points.append(
            {
                "time_s": round(t, 6),
                "x": x,
                "y": y,
                "z": z,
                "vx": vx,
                "vy": vy,
                "vz": vz,
                "bounces": bounces,
            }
        )

        x_prev, y_prev, z_prev = x, y, z
        x, y, z, vx, vy, vz = RobotPredictor._step_euler(x, y, z, vx, vy, vz, step_s)
        if bounces < MAX_BOUNCES:
            x, y, z, vx, vy, vz, did_bounce = RobotPredictor._apply_bounce(
                x_prev, y_prev, z_prev, x, y, z, vx, vy, vz, step_s
            )
            if did_bounce:
                bounces += 1

        t += step_s

    return points


def _predictor_has_offline_state(predictor: RobotPredictor) -> bool:
    """Check whether the predictor has enough buffered data to extrapolate.

    This intentionally ignores the real-time stale timeout so saved/replayed
    throws can still be reviewed after capture has finished.
    """
    if predictor.velocity is None or not predictor.positions:
        return False
    span = predictor.positions[-1][3] - predictor.positions[0][3]
    return span >= predictor.MIN_TIME_SPAN


def _build_throw_review(raw_rows: list[dict[str, object]]) -> dict[str, object]:
    predictor = RobotPredictor()
    replay_start_t = time.perf_counter()

    first_ready_state: tuple[float, float, float, float, float, float] | None = None
    first_ready_time: float | None = None
    first_ready_sample_idx: int | None = None

    latest_state: tuple[float, float, float, float, float, float] | None = None
    latest_time: float | None = None
    latest_sample_idx: int | None = None

    chosen_state: tuple[float, float, float, float, float, float] | None = None
    chosen_time: float | None = None
    chosen_sample_idx: int | None = None
    chosen_intercept: dict[str, float | int | bool] | None = None

    for idx, row in enumerate(raw_rows):
        t = replay_start_t + float(row["time_from_start_s"])
        x = float(row["rob_x"])
        y = float(row["rob_y"])
        z = float(row["rob_z"])

        accepted = predictor.add_position(x, y, z, t)
        if not accepted:
            continue

        state = predictor._get_prediction_state()
        if state is None or not _predictor_has_offline_state(predictor):
            continue

        latest_state = state
        latest_time = t
        latest_sample_idx = idx

        if first_ready_state is None:
            first_ready_state = state
            first_ready_time = t
            first_ready_sample_idx = idx

        intercept = predictor.predict_intercept()
        if intercept is not None and chosen_state is None:
            chosen_state = state
            chosen_time = t
            chosen_sample_idx = idx
            chosen_intercept = intercept

    if chosen_state is None and latest_state is not None:
        chosen_state = latest_state
        chosen_time = latest_time
        chosen_sample_idx = latest_sample_idx
    elif chosen_state is None and first_ready_state is not None:
        chosen_state = first_ready_state
        chosen_time = first_ready_time
        chosen_sample_idx = first_ready_sample_idx

    predicted_points = _simulate_trajectory(chosen_state) if chosen_state is not None else []

    return {
        "prediction_state": chosen_state,
        "prediction_start_time_s": chosen_time,
        "prediction_sample_idx": chosen_sample_idx,
        "intercept": chosen_intercept,
        "predicted_points": predicted_points,
    }


def _set_equal_3d(
    ax,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
) -> None:
    if xs.size == 0 or ys.size == 0 or zs.size == 0:
        return

    x_mid = float((xs.max() + xs.min()) / 2.0)
    y_mid = float((ys.max() + ys.min()) / 2.0)
    z_mid = float((zs.max() + zs.min()) / 2.0)
    radius = float(max(xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()) / 2.0)
    if radius <= 0.0:
        radius = 1.0

    ax.set_xlim(x_mid - radius, x_mid + radius)
    ax.set_ylim(y_mid - radius, y_mid + radius)
    ax.set_zlim(z_mid - radius, z_mid + radius)


def _plot_throw_review(
    throw_index: int,
    raw_rows: list[dict[str, object]],
    review: dict[str, object],
) -> str:
    raw_x = np.array([float(row["rob_x"]) for row in raw_rows], dtype=float)
    raw_y = np.array([float(row["rob_y"]) for row in raw_rows], dtype=float)
    raw_z = np.array([float(row["rob_z"]) for row in raw_rows], dtype=float)

    predicted_points = review["predicted_points"]
    pred_x = np.array([float(point["x"]) for point in predicted_points], dtype=float)
    pred_y = np.array([float(point["y"]) for point in predicted_points], dtype=float)
    pred_z = np.array([float(point["z"]) for point in predicted_points], dtype=float)

    intercept = review["intercept"]

    fig = plt.figure(figsize=(14, 10))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xz = fig.add_subplot(2, 2, 3)
    ax_yz = fig.add_subplot(2, 2, 4)

    ax3d.scatter(raw_x, raw_y, raw_z, color="tab:blue", s=24, label="Raw points")
    ax3d.plot(raw_x, raw_y, raw_z, color="tab:blue", alpha=0.6)

    ax_xy.scatter(raw_x, raw_y, color="tab:blue", s=20, label="Raw points")
    ax_xy.plot(raw_x, raw_y, color="tab:blue", alpha=0.6)

    ax_xz.scatter(raw_x, raw_z, color="tab:blue", s=20, label="Raw points")
    ax_xz.plot(raw_x, raw_z, color="tab:blue", alpha=0.6)

    ax_yz.scatter(raw_y, raw_z, color="tab:blue", s=20, label="Raw points")
    ax_yz.plot(raw_y, raw_z, color="tab:blue", alpha=0.6)

    if predicted_points:
        ax3d.plot(pred_x, pred_y, pred_z, color="tab:orange", linewidth=2.2, label="Predictor trajectory")
        ax3d.scatter(pred_x, pred_y, pred_z, color="tab:orange", s=10, alpha=0.35)
        ax_xy.plot(pred_x, pred_y, color="tab:orange", linewidth=2.0, label="Predictor trajectory")
        ax_xy.scatter(pred_x, pred_y, color="tab:orange", s=10, alpha=0.35)
        ax_xz.plot(pred_x, pred_z, color="tab:orange", linewidth=2.0, label="Predictor trajectory")
        ax_xz.scatter(pred_x, pred_z, color="tab:orange", s=10, alpha=0.35)
        ax_yz.plot(pred_y, pred_z, color="tab:orange", linewidth=2.0, label="Predictor trajectory")
        ax_yz.scatter(pred_y, pred_z, color="tab:orange", s=10, alpha=0.35)

        ax3d.scatter(
            [pred_x[0]],
            [pred_y[0]],
            [pred_z[0]],
            color="tab:orange",
            marker="o",
            s=60,
            label="Prediction start",
        )
        ax_xy.scatter([pred_x[0]], [pred_y[0]], color="tab:orange", marker="o", s=45, label="Prediction start")
        ax_xz.scatter([pred_x[0]], [pred_z[0]], color="tab:orange", marker="o", s=45, label="Prediction start")
        ax_yz.scatter([pred_y[0]], [pred_z[0]], color="tab:orange", marker="o", s=45, label="Prediction start")

    if intercept is not None:
        ix = float(intercept["x"])
        iy = float(intercept["y"])
        iz = float(intercept["z"])
        ax3d.scatter([ix], [iy], [iz], color="tab:red", s=90, marker="*", label="Intercept")
        ax_xy.scatter([ix], [iy], color="tab:red", s=90, marker="*", label="Intercept")
        ax_xz.scatter([ix], [iz], color="tab:red", s=90, marker="*", label="Intercept")
        ax_yz.scatter([iy], [iz], color="tab:red", s=90, marker="*", label="Intercept")

    all_x = raw_x if pred_x.size == 0 else np.concatenate((raw_x, pred_x))
    all_y = raw_y if pred_y.size == 0 else np.concatenate((raw_y, pred_y))
    all_z = raw_z if pred_z.size == 0 else np.concatenate((raw_z, pred_z))
    if intercept is not None:
        all_x = np.append(all_x, float(intercept["x"]))
        all_y = np.append(all_y, float(intercept["y"]))
        all_z = np.append(all_z, float(intercept["z"]))

    _set_equal_3d(ax3d, all_x, all_y, all_z)

    ax3d.set_title("3D")
    ax3d.set_xlabel("x (mm)")
    ax3d.set_ylabel("y (mm)")
    ax3d.set_zlabel("z (mm)")

    ax_xy.set_title("XY")
    ax_xy.set_xlabel("x (mm)")
    ax_xy.set_ylabel("y (mm)")
    ax_xy.grid(True, alpha=0.25)

    ax_xz.set_title("XZ")
    ax_xz.set_xlabel("x (mm)")
    ax_xz.set_ylabel("z (mm)")
    ax_xz.grid(True, alpha=0.25)

    ax_yz.set_title("YZ")
    ax_yz.set_xlabel("y (mm)")
    ax_yz.set_ylabel("z (mm)")
    ax_yz.grid(True, alpha=0.25)

    handles, labels = ax_xy.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    prediction_state = review["prediction_state"]
    status = "trajectory found" if predicted_points else "no predictor trajectory"
    if intercept is not None:
        status += " | intercept found"
    elif prediction_state is not None:
        status += " | no intercept"

    fig.suptitle(
        f"Throw {throw_index}: {len(raw_rows)} raw points | {status}",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.01,
        "Press S to save, N to skip, or Q to quit",
        ha="center",
        va="bottom",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    decision = {"value": "skip"}

    def _on_key(event) -> None:
        key = (event.key or "").lower()
        if key == "s":
            decision["value"] = "save"
            plt.close(fig)
        elif key == "n":
            decision["value"] = "skip"
            plt.close(fig)
        elif key == "q":
            decision["value"] = "quit"
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", _on_key)
    plt.show()
    plt.close(fig)
    return decision["value"]


def _save_review_csv(
    csv_path: Path,
    throw_id: int,
    raw_rows: list[dict[str, object]],
    review: dict[str, object],
) -> None:
    start_time_s = float(raw_rows[0]["time_s"])
    pred_start_s = review["prediction_start_time_s"]
    pred_sample_idx = review["prediction_sample_idx"]
    intercept = review["intercept"]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for i, row in enumerate(raw_rows):
            writer.writerow(
                {
                    "throw_id": throw_id,
                    "point_type": "raw",
                    "point_index": i,
                    "time_s": row["time_s"],
                    "time_from_start_s": row["time_from_start_s"],
                    "x_mm": row["rob_x"],
                    "y_mm": row["rob_y"],
                    "z_mm": row["rob_z"],
                    "accepted": 1,
                    "reproj_err_px": row["reproj_err"],
                    "disparity_px": row["disparity"],
                }
            )

        if pred_start_s is not None:
            for i, point in enumerate(review["predicted_points"]):
                point_time_s = float(pred_start_s) + float(point["time_s"])
                writer.writerow(
                    {
                        "throw_id": throw_id,
                        "point_type": "predicted",
                        "point_index": i,
                        "time_s": round(point_time_s, 6),
                        "time_from_start_s": round(point_time_s - start_time_s, 6),
                        "x_mm": round(float(point["x"]), 6),
                        "y_mm": round(float(point["y"]), 6),
                        "z_mm": round(float(point["z"]), 6),
                        "predictor_sample_idx": pred_sample_idx,
                        "predictor_start_time_s": pred_start_s,
                        "vx_mm_s": round(float(point["vx"]), 6),
                        "vy_mm_s": round(float(point["vy"]), 6),
                        "vz_mm_s": round(float(point["vz"]), 6),
                        "bounces": point["bounces"],
                    }
                )

        if intercept is not None and pred_start_s is not None:
            intercept_time_s = float(pred_start_s) + float(intercept["time"])
            writer.writerow(
                {
                    "throw_id": throw_id,
                    "point_type": "intercept",
                    "point_index": 0,
                    "time_s": round(intercept_time_s, 6),
                    "time_from_start_s": round(intercept_time_s - start_time_s, 6),
                    "x_mm": round(float(intercept["x"]), 6),
                    "y_mm": round(float(intercept["y"]), 6),
                    "z_mm": round(float(intercept["z"]), 6),
                    "predictor_sample_idx": pred_sample_idx,
                    "predictor_start_time_s": pred_start_s,
                    "intercept_x_mm": round(float(intercept["x"]), 6),
                    "intercept_y_mm": round(float(intercept["y"]), 6),
                    "intercept_z_mm": round(float(intercept["z"]), 6),
                    "vx_mm_s": round(float(intercept["vx"]), 6),
                    "vy_mm_s": round(float(intercept["vy"]), 6),
                    "vz_mm_s": round(float(intercept["vz"]), 6),
                    "bounces": intercept.get("bounces", 0),
                    "clamped": int(bool(intercept.get("clamped", False))),
                }
            )


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record live throws, plot raw data against the robot predictor, "
            "and optionally save approved trajectories."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for approved trajectory CSV files.",
    )
    parser.add_argument(
        "--warmup-s",
        type=float,
        default=WARMUP_S_DEFAULT,
        help="Background warmup duration before capture starts.",
    )
    parser.add_argument(
        "--reproj-err-max",
        type=float,
        default=REPROJ_ERR_MAX_DEFAULT,
        help="Maximum accepted reprojection error during live capture.",
    )
    parser.add_argument(
        "--lost-frames-to-stop",
        type=int,
        default=LOST_FRAMES_TO_STOP_DEFAULT,
        help="Stop the current throw after this many missed frames.",
    )
    parser.add_argument(
        "--stop-near-workspace-mm",
        type=float,
        default=STOP_NEAR_WORKSPACE_MM_DEFAULT,
        help=(
            "Stop a throw once the measured ball enters the workspace or comes "
            "within this distance of the workspace boundary."
        ),
    )
    parser.add_argument(
        "--max-throws",
        type=int,
        default=0,
        help="Optional cap on reviewed throws in one run (0 = unlimited).",
    )
    parser.add_argument(
        "--move-robot",
        action="store_true",
        help="Enable live UART intercept sends so the robot moves during capture.",
    )
    parser.add_argument(
        "--port",
        default=os.environ.get("STM32_UART_PORT"),
        help="UART port for robot control (or set STM32_UART_PORT).",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=DEFAULT_BAUD,
        help="UART baud rate for robot control.",
    )
    parser.add_argument(
        "--tx-interval-ms",
        type=float,
        default=30.0,
        help="Minimum time between UART target transmissions.",
    )
    parser.add_argument(
        "--home-ack-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for startup home confirmation (0 = infinite).",
    )
    parser.add_argument(
        "--quiet-uart",
        action="store_true",
        help="Reduce local UART TX chatter.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.move_robot and not args.port:
        raise SystemExit("Robot motion requires --port or STM32_UART_PORT.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cam = load_camera_settings()
    frame_width = cam["frame_width"]
    frame_height = cam["frame_height"]
    cam_left_id = cam["camera0"]
    cam_right_id = cam["camera1"]

    tf = load_points_based_transform()
    rotation = tf["rotation"]
    translation = tf["translation"]
    cam_scale = tf["camera_scale_to_robot_units"]

    triangulator = StereoTriangulator(
        calibration_dir=str(CAMERA_PROPERTIES_DIR),
        cam_left_id=cam_left_id,
        cam_right_id=cam_right_id,
    )

    window_name = "Plot Trajectory Capture"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 800)
    triangulator.start_cameras(frame_width, frame_height)
    robot_controller: LiveRobotController | None = None

    reviewed_throws = 0
    saved_throws = 0

    try:
        if args.move_robot:
            home_timeout = None if args.home_ack_timeout == 0.0 else args.home_ack_timeout
            robot_controller = _init_robot_controller(
                port=str(args.port),
                baud_rate=int(args.baud),
                tx_interval_s=max(0.0, args.tx_interval_ms / 1000.0),
                home_ack_timeout_s=home_timeout,
                uart_verbose=not args.quiet_uart,
            )

        if not _warmup_background(
            triangulator=triangulator,
            frame_width=frame_width,
            frame_height=frame_height,
            warmup_s=max(0.0, args.warmup_s),
            window_name=window_name,
        ):
            return 1

        while True:
            raw_rows, stop_reason, quit_requested = _capture_one_throw(
                triangulator=triangulator,
                rotation=rotation,
                translation=translation,
                cam_scale=cam_scale,
                frame_width=frame_width,
                frame_height=frame_height,
                reproj_err_max=max(0.0, args.reproj_err_max),
                lost_frames_to_stop=max(1, args.lost_frames_to_stop),
                stop_near_workspace_mm=max(0.0, args.stop_near_workspace_mm),
                window_name=window_name,
                robot_controller=robot_controller,
            )

            if raw_rows:
                reviewed_throws += 1
                review = _build_throw_review(raw_rows)
                action = _plot_throw_review(reviewed_throws, raw_rows, review)

                if action == "save":
                    csv_path = _next_throw_csv_path(output_dir)
                    _save_review_csv(csv_path, reviewed_throws, raw_rows, review)
                    saved_throws += 1
                    _print(f"Saved trajectory CSV -> {csv_path}")
                elif action == "quit":
                    _print("Quit requested from review plot.")
                    break
                else:
                    _print("Skipped saving this throw.")

                if stop_reason:
                    _print(f"Stop reason: {stop_reason}")

            elif quit_requested:
                break
            else:
                _print("No throw data captured.")

            if quit_requested:
                break
            if args.max_throws > 0 and reviewed_throws >= args.max_throws:
                break

    finally:
        if robot_controller is not None:
            try:
                _send_robot_home(robot_controller)
            except Exception as exc:
                _print(f"[robot] Shutdown HOME failed: {exc}")
            try:
                robot_controller.uart.close()
            except Exception:
                pass
        triangulator.stop_cameras()
        cv2.destroyAllWindows()

    _print(f"Session complete. Reviewed={reviewed_throws} Saved={saved_throws}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
