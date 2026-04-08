"""
Capture or replay throws and compare:
  1) Raw robot-frame positions
  2) Regression-based state (RobotPredictor legacy path)
  3) Base Kalman filter state (RobotPredictor default BallStateEstimator3D)

Default behavior:
  - Capture one live throw from stereo cameras
  - Auto-stop when throw is lost
  - Save captured throw CSV
  - Run comparison on that throw

Default output directory:
  <repo>/new_top_level/test_script/triangulation_cvs
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
NEW_TOP_LEVEL_DIR = SCRIPT_DIR.parent
CAMERA_PROPERTIES_DIR = NEW_TOP_LEVEL_DIR / "camera_params" / "camera_properties"
DEFAULT_OUTPUT_DIR = NEW_TOP_LEVEL_DIR / "test_script" / "triangulation_cvs"

if str(NEW_TOP_LEVEL_DIR) not in sys.path:
    sys.path.insert(0, str(NEW_TOP_LEVEL_DIR))

from ball_tracking.stereo_triangulator import StereoTriangulator
from comm_functions.points_based_transform import load_points_based_transform, cam_to_robot
from config.camera_config import load_camera_settings
from estimation.robot_predictor import RobotPredictor


REPROJ_ERR_MAX_DEFAULT = 10.0
WARMUP_S_DEFAULT = 2.0
LOST_FRAMES_TO_STOP_DEFAULT = 35


THROW_COLUMNS = [
    "frame",
    "time_s",
    "time_from_start_s",
    "dt_s",
    "cam_x",
    "cam_y",
    "cam_z",
    "rob_x",
    "rob_y",
    "rob_z",
    "disparity",
    "reproj_err",
]


def _print(*args, **kwargs):
    print("[compare-rvkf]", *args, **kwargs)


def _parse_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _norm3(x: float | None, y: float | None, z: float | None) -> float | None:
    if x is None or y is None or z is None:
        return None
    return math.sqrt(x * x + y * y + z * z)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _format_f(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _format_i(value: int | None) -> str:
    if value is None:
        return ""
    return str(value)


def _bool01(value: bool) -> str:
    return "1" if value else "0"


def _collect_input_files(input_path: str, pattern: str) -> list[Path]:
    p = Path(input_path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted([x for x in p.glob(pattern) if x.is_file()])
    return sorted([x for x in Path(".").glob(input_path) if x.is_file()])


def _make_regression_predictor() -> RobotPredictor:
    predictor = RobotPredictor()
    predictor.state_estimator = None
    predictor._using_state_estimator = False
    predictor.velocity = None
    return predictor


def _make_base_kf_predictor() -> RobotPredictor:
    predictor = RobotPredictor()
    if predictor.state_estimator is None:
        raise RuntimeError(
            "Kalman filter is unavailable. Install filterpy first "
            "(new_top_level/requirements.txt)."
        )
    predictor.velocity = None
    return predictor


def _reg_ready_offline(predictor: RobotPredictor) -> bool:
    if predictor.velocity is None:
        return False
    if len(predictor.positions) < predictor.MIN_POINTS:
        return False
    span = predictor.positions[-1][3] - predictor.positions[0][3]
    return span >= predictor.MIN_TIME_SPAN


def _path_stub(input_csv: Path) -> str:
    if input_csv.parent == SCRIPT_DIR:
        return input_csv.stem
    parent_stub = input_csv.parent.name
    return f"{parent_stub}_{input_csv.stem}"


def _run_one_file(input_csv: Path, output_dir: Path) -> dict[str, str]:
    reg_predictor = _make_regression_predictor()
    kf_predictor = _make_base_kf_predictor()

    output_csv = output_dir / f"{_path_stub(input_csv)}_comparison.csv"

    fieldnames = [
        "sample_idx",
        "frame",
        "time_s",
        "dt_s",
        "raw_x",
        "raw_y",
        "raw_z",
        "reg_accepted",
        "reg_reject_reason",
        "reg_buf",
        "reg_ready",
        "reg_pos_x",
        "reg_pos_y",
        "reg_pos_z",
        "reg_vx",
        "reg_vy",
        "reg_vz",
        "reg_speed",
        "kf_accepted",
        "kf_reject_reason",
        "kf_buf",
        "kf_ready",
        "kf_updates",
        "kf_pos_x",
        "kf_pos_y",
        "kf_pos_z",
        "kf_vx",
        "kf_vy",
        "kf_vz",
        "kf_speed",
        "raw_minus_reg_x",
        "raw_minus_reg_y",
        "raw_minus_reg_z",
        "raw_minus_reg_norm",
        "raw_minus_kf_x",
        "raw_minus_kf_y",
        "raw_minus_kf_z",
        "raw_minus_kf_norm",
        "reg_minus_kf_vx",
        "reg_minus_kf_vy",
        "reg_minus_kf_vz",
        "reg_minus_kf_v_norm",
    ]

    samples = 0
    reg_accepts = 0
    kf_accepts = 0
    reg_ready_count = 0
    kf_ready_count = 0

    raw_reg_norms: list[float] = []
    raw_kf_norms: list[float] = []
    reg_kf_v_norms: list[float] = []

    last_t: float | None = None

    with input_csv.open("r", newline="", encoding="utf-8") as fin, output_csv.open(
        "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            t = _parse_float(row, "time_s")
            x = _parse_float(row, "rob_x")
            y = _parse_float(row, "rob_y")
            z = _parse_float(row, "rob_z")
            if t is None or x is None or y is None or z is None:
                continue

            samples += 1
            dt_s = None if last_t is None else (t - last_t)
            last_t = t

            frame_raw = row.get("frame", "")
            frame_i = None
            if frame_raw is not None and frame_raw.strip() != "":
                try:
                    frame_i = int(float(frame_raw))
                except ValueError:
                    frame_i = None

            reg_accepted = reg_predictor.add_position(x, y, z, t)
            if reg_accepted:
                reg_accepts += 1

            kf_accepted = kf_predictor.add_position(x, y, z, t)
            if kf_accepted:
                kf_accepts += 1

            reg_ready = _reg_ready_offline(reg_predictor)
            if reg_ready:
                reg_ready_count += 1

            kf_ready = False
            kf_updates = 0
            kf_state = None
            if kf_predictor.state_estimator is not None:
                kf_ready = kf_predictor.state_estimator.is_ready()
                if kf_ready:
                    kf_ready_count += 1
                kf_updates = kf_predictor.state_estimator.update_count
                kf_state = kf_predictor.state_estimator.get_state()

            reg_pos = None
            if reg_predictor.positions:
                rp = reg_predictor.positions[-1]
                reg_pos = (float(rp[0]), float(rp[1]), float(rp[2]))

            reg_vel = None
            if reg_predictor.velocity is not None:
                rv = reg_predictor.velocity
                reg_vel = (float(rv[0]), float(rv[1]), float(rv[2]))

            reg_speed = None
            if reg_vel is not None:
                reg_speed = _norm3(reg_vel[0], reg_vel[1], reg_vel[2])

            kf_pos = None
            kf_vel = None
            if kf_state is not None:
                kf_pos = (float(kf_state[0]), float(kf_state[1]), float(kf_state[2]))
                kf_vel = (float(kf_state[3]), float(kf_state[4]), float(kf_state[5]))

            kf_speed = None
            if kf_vel is not None:
                kf_speed = _norm3(kf_vel[0], kf_vel[1], kf_vel[2])

            raw_minus_reg_x = raw_minus_reg_y = raw_minus_reg_z = None
            if reg_pos is not None:
                raw_minus_reg_x = x - reg_pos[0]
                raw_minus_reg_y = y - reg_pos[1]
                raw_minus_reg_z = z - reg_pos[2]
            raw_minus_reg_norm = _norm3(raw_minus_reg_x, raw_minus_reg_y, raw_minus_reg_z)
            if raw_minus_reg_norm is not None:
                raw_reg_norms.append(raw_minus_reg_norm)

            raw_minus_kf_x = raw_minus_kf_y = raw_minus_kf_z = None
            if kf_pos is not None:
                raw_minus_kf_x = x - kf_pos[0]
                raw_minus_kf_y = y - kf_pos[1]
                raw_minus_kf_z = z - kf_pos[2]
            raw_minus_kf_norm = _norm3(raw_minus_kf_x, raw_minus_kf_y, raw_minus_kf_z)
            if raw_minus_kf_norm is not None:
                raw_kf_norms.append(raw_minus_kf_norm)

            reg_minus_kf_vx = reg_minus_kf_vy = reg_minus_kf_vz = None
            if reg_vel is not None and kf_vel is not None:
                reg_minus_kf_vx = reg_vel[0] - kf_vel[0]
                reg_minus_kf_vy = reg_vel[1] - kf_vel[1]
                reg_minus_kf_vz = reg_vel[2] - kf_vel[2]
            reg_minus_kf_v_norm = _norm3(reg_minus_kf_vx, reg_minus_kf_vy, reg_minus_kf_vz)
            if reg_minus_kf_v_norm is not None:
                reg_kf_v_norms.append(reg_minus_kf_v_norm)

            writer.writerow(
                {
                    "sample_idx": samples,
                    "frame": _format_i(frame_i),
                    "time_s": _format_f(t),
                    "dt_s": _format_f(dt_s),
                    "raw_x": _format_f(x),
                    "raw_y": _format_f(y),
                    "raw_z": _format_f(z),
                    "reg_accepted": _bool01(reg_accepted),
                    "reg_reject_reason": reg_predictor._last_reject_reason or "",
                    "reg_buf": len(reg_predictor.positions),
                    "reg_ready": _bool01(reg_ready),
                    "reg_pos_x": _format_f(reg_pos[0]) if reg_pos is not None else "",
                    "reg_pos_y": _format_f(reg_pos[1]) if reg_pos is not None else "",
                    "reg_pos_z": _format_f(reg_pos[2]) if reg_pos is not None else "",
                    "reg_vx": _format_f(reg_vel[0]) if reg_vel is not None else "",
                    "reg_vy": _format_f(reg_vel[1]) if reg_vel is not None else "",
                    "reg_vz": _format_f(reg_vel[2]) if reg_vel is not None else "",
                    "reg_speed": _format_f(reg_speed),
                    "kf_accepted": _bool01(kf_accepted),
                    "kf_reject_reason": kf_predictor._last_reject_reason or "",
                    "kf_buf": len(kf_predictor.positions),
                    "kf_ready": _bool01(kf_ready),
                    "kf_updates": kf_updates,
                    "kf_pos_x": _format_f(kf_pos[0]) if kf_pos is not None else "",
                    "kf_pos_y": _format_f(kf_pos[1]) if kf_pos is not None else "",
                    "kf_pos_z": _format_f(kf_pos[2]) if kf_pos is not None else "",
                    "kf_vx": _format_f(kf_vel[0]) if kf_vel is not None else "",
                    "kf_vy": _format_f(kf_vel[1]) if kf_vel is not None else "",
                    "kf_vz": _format_f(kf_vel[2]) if kf_vel is not None else "",
                    "kf_speed": _format_f(kf_speed),
                    "raw_minus_reg_x": _format_f(raw_minus_reg_x),
                    "raw_minus_reg_y": _format_f(raw_minus_reg_y),
                    "raw_minus_reg_z": _format_f(raw_minus_reg_z),
                    "raw_minus_reg_norm": _format_f(raw_minus_reg_norm),
                    "raw_minus_kf_x": _format_f(raw_minus_kf_x),
                    "raw_minus_kf_y": _format_f(raw_minus_kf_y),
                    "raw_minus_kf_z": _format_f(raw_minus_kf_z),
                    "raw_minus_kf_norm": _format_f(raw_minus_kf_norm),
                    "reg_minus_kf_vx": _format_f(reg_minus_kf_vx),
                    "reg_minus_kf_vy": _format_f(reg_minus_kf_vy),
                    "reg_minus_kf_vz": _format_f(reg_minus_kf_vz),
                    "reg_minus_kf_v_norm": _format_f(reg_minus_kf_v_norm),
                }
            )

    reg_accept_rate = (reg_accepts / samples) if samples else 0.0
    kf_accept_rate = (kf_accepts / samples) if samples else 0.0
    reg_ready_rate = (reg_ready_count / samples) if samples else 0.0
    kf_ready_rate = (kf_ready_count / samples) if samples else 0.0

    return {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "samples": str(samples),
        "reg_accept_rate": _format_f(reg_accept_rate),
        "kf_accept_rate": _format_f(kf_accept_rate),
        "reg_ready_rate": _format_f(reg_ready_rate),
        "kf_ready_rate": _format_f(kf_ready_rate),
        "mean_raw_minus_reg_norm_mm": _format_f(_mean(raw_reg_norms)),
        "mean_raw_minus_kf_norm_mm": _format_f(_mean(raw_kf_norms)),
        "mean_reg_minus_kf_v_norm_mm_s": _format_f(_mean(reg_kf_v_norms)),
    }


def _write_summary(summary_path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "input_csv",
        "output_csv",
        "samples",
        "reg_accept_rate",
        "kf_accept_rate",
        "reg_ready_rate",
        "kf_ready_rate",
        "mean_raw_minus_reg_norm_mm",
        "mean_raw_minus_kf_norm_mm",
        "mean_reg_minus_kf_v_norm_mm_s",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def _save_throw_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=THROW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


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


def _capture_one_throw(
    output_dir: Path,
    warmup_s: float,
    reproj_err_max: float,
    lost_frames_to_stop: int,
) -> Path | None:
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

    rows: list[dict[str, object]] = []
    throw_started = False
    lost_frames = 0
    last_ts: float | None = None
    start_ts: float | None = None
    frame_idx = 0
    stop_reason = ""

    window_name = "Compare Regression vs KF"
    triangulator.start_cameras(frame_width, frame_height)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 800)

    try:
        if not _warmup_background(
            triangulator=triangulator,
            frame_width=frame_width,
            frame_height=frame_height,
            warmup_s=warmup_s,
            window_name=window_name,
        ):
            return None

        _print("Ready. Throw one ball. Script will auto-stop after throw leaves view.")
        _print("Press 'q' to abort.")

        while True:
            result = triangulator.update()
            frame_ts = result.get("capture_time", time.perf_counter())
            valid_sample = False

            if result["left_frame"] is None or result["right_frame"] is None:
                continue

            left_vis, right_vis = triangulator.draw_results(result)

            if result["found_3d"]:
                reproj = float(result.get("reproj_err", 0.0))
                if reproj <= reproj_err_max:
                    valid_sample = True
                    cx, cy, cz = result["position_3d"]
                    rx, ry, rz = cam_to_robot(
                        rotation, translation, cam_scale, cx, cy, cz
                    )

                    if not throw_started:
                        throw_started = True
                        start_ts = frame_ts
                        _print("Throw detected. Recording started.")

                    dt = None if last_ts is None else (frame_ts - last_ts)
                    last_ts = frame_ts

                    frame_idx += 1
                    rows.append(
                        {
                            "frame": frame_idx,
                            "time_s": round(frame_ts, 6),
                            "time_from_start_s": (
                                round(frame_ts - start_ts, 6) if start_ts is not None else ""
                            ),
                            "dt_s": round(dt, 6) if dt is not None else "",
                            "cam_x": round(cx, 6),
                            "cam_y": round(cy, 6),
                            "cam_z": round(cz, 6),
                            "rob_x": round(rx, 6),
                            "rob_y": round(ry, 6),
                            "rob_z": round(rz, 6),
                            "disparity": (
                                round(float(result.get("disparity", 0.0)), 6)
                                if result.get("disparity") is not None
                                else ""
                            ),
                            "reproj_err": round(reproj, 6),
                        }
                    )

                    lost_frames = 0
                else:
                    result["reject_reason"] = f"reproj({reproj:.1f}px)"

            if throw_started and not valid_sample:
                lost_frames += 1
                if lost_frames >= lost_frames_to_stop:
                    stop_reason = "lost_after_throw"
                    _print("Throw ended (lost track).")
                    break

            status_lines = [
                f"Rows: {len(rows)}",
                f"Recording: {'YES' if throw_started else 'WAITING FOR THROW'}",
                f"Lost frames: {lost_frames}/{lost_frames_to_stop}",
            ]
            if result.get("reject_reason"):
                status_lines.append(f"Reject: {result['reject_reason']}")

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

            key = cv2.waitKeyEx(1)
            if 0 <= key <= 255:
                key = ord(chr(key).lower())
            if key == ord("q"):
                stop_reason = "manual_quit"
                _print("Quit requested.")
                break

    finally:
        triangulator.stop_cameras()
        cv2.destroyAllWindows()

    if not rows:
        _print("No throw data captured.")
        return None

    csv_path = _next_throw_csv_path(output_dir)
    _save_throw_csv(csv_path, rows)
    _print(f"Saved throw CSV -> {csv_path}")
    if stop_reason:
        _print(f"Stop reason: {stop_reason}")
    return csv_path


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture or replay throw CSVs and compare raw vs regression vs base KF."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["capture", "replay"],
        default="capture",
        help=(
            "capture: record one live throw then compare it. "
            "replay: compare existing throw CSV files."
        ),
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "Input CSV file, directory, or glob for replay mode. "
            "Default: new_top_level/test_script/triangulation_cvs"
        ),
    )
    parser.add_argument(
        "--pattern",
        default="throw_*.csv",
        help="Filename pattern when --input is a directory (replay mode).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for captured and comparison CSV files.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on number of input files in replay mode (0 = all).",
    )
    parser.add_argument(
        "--warmup-s",
        type=float,
        default=WARMUP_S_DEFAULT,
        help="Background warmup duration before live capture.",
    )
    parser.add_argument(
        "--reproj-err-max",
        type=float,
        default=REPROJ_ERR_MAX_DEFAULT,
        help="Maximum accepted reprojection error in live capture mode.",
    )
    parser.add_argument(
        "--lost-frames-to-stop",
        type=int,
        default=LOST_FRAMES_TO_STOP_DEFAULT,
        help="End live capture after this many missed frames once throw started.",
    )
    return parser.parse_args(list(argv))


def _run_replay(args: argparse.Namespace, output_dir: Path) -> int:
    input_files = _collect_input_files(args.input, args.pattern)
    if not input_files:
        print(f"No input CSV files found for: {args.input}")
        return 1

    if args.max_files > 0:
        input_files = input_files[: args.max_files]

    summary_rows: list[dict[str, str]] = []
    for csv_path in input_files:
        summary = _run_one_file(csv_path, output_dir)
        summary_rows.append(summary)
        print(
            f"[OK] {csv_path.name} -> {Path(summary['output_csv']).name} "
            f"(samples={summary['samples']})"
        )

    summary_path = output_dir / "comparison_summary.csv"
    _write_summary(summary_path, summary_rows)
    print(f"[DONE] Wrote {len(summary_rows)} comparison CSV(s) and summary:")
    print(f"       {summary_path}")
    return 0


def _run_capture(args: argparse.Namespace, output_dir: Path) -> int:
    captured_csv = _capture_one_throw(
        output_dir=output_dir,
        warmup_s=max(0.0, args.warmup_s),
        reproj_err_max=max(0.0, args.reproj_err_max),
        lost_frames_to_stop=max(1, args.lost_frames_to_stop),
    )
    if captured_csv is None:
        return 1

    summary = _run_one_file(captured_csv, output_dir)
    summary_path = output_dir / "comparison_summary.csv"
    _write_summary(summary_path, [summary])

    print("[DONE] Live capture + comparison complete:")
    print(f"       Throw CSV:      {captured_csv}")
    print(f"       Comparison CSV: {summary['output_csv']}")
    print(f"       Summary CSV:    {summary_path}")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "capture":
        return _run_capture(args, output_dir)
    return _run_replay(args, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
