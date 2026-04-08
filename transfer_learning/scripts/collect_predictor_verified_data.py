#!/usr/bin/env python3
"""
Collect transfer-learning samples using RobotPredictor + real robot execution.

Workflow:
1) Press `g` to arm.
2) Throw ball.
3) Script captures first N accepted predictor points (default 6),
   sends predictor intercept to robot, waits for movement to complete + HOME.
4) Review mode:
   - `k`: keep sample (append row)
   - `x`: discard sample
5) Script auto-rearms for next throw after review decision.

Label target is predictor output that was sent to robot.
"""

from __future__ import annotations

import argparse
import csv
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2


def _print(*args, **kwargs) -> None:
    print("[collect_predictor]", *args, **kwargs)


def _safe_put(q: queue.Queue, item) -> bool:
    try:
        q.put_nowait(item)
        return True
    except queue.Full:
        return False


@dataclass
class RuntimeState:
    gate_on: bool = False
    startup_homing_done: bool = False
    robot_at_home: bool = False
    robot_state: str = "UNKNOWN"
    intercept_inflight: bool = False
    throws_sent: int = 0
    updates_sent: int = 0
    last_latency_ms: float = 0.0
    last_adjusted_ms: float = 0.0


@dataclass
class CaptureConfig:
    stack_root: Path
    output_dir: Path
    output_file: str
    num_points: int
    uart_port: str
    baud_rate: int
    warmup_s: float
    tx_interval_s: float
    home_ack_timeout_s: Optional[float]
    gap_reset_s: float
    preview_width: int
    uart_verbose: bool


class PredictorVerifiedCollector:
    def __init__(self, cfg: CaptureConfig) -> None:
        self.cfg = cfg
        self._state = RuntimeState()
        self._stop_event = threading.Event()
        self._status = "Press 'g' to arm."

        self.output_dir = cfg.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_csv = self.output_dir / cfg.output_file
        self.sample_id = self._next_sample_id()
        self.saved_samples = 0
        self.discarded_samples = 0
        self.failed_samples = 0

        self._stack_root = cfg.stack_root.resolve()
        self._load_stack_modules()
        self._init_workers()

        self._latest_obs = None
        self._latest_predictor_update = None
        self._latest_intercept = None
        self._last_sent_intercept = None

        self._first_points: List[object] = []
        self._last_accept_t: Optional[float] = None
        self._throw_had_send = False

        self._review_mode = False
        self._pending_row: Optional[Dict[str, float]] = None

        self._fps = 0.0
        self._fps_count = 0
        self._fps_t0 = time.perf_counter()

    def _load_stack_modules(self) -> None:
        if str(self._stack_root) not in sys.path:
            sys.path.insert(0, str(self._stack_root))

        from ball_tracking.stereo_triangulator import StereoTriangulator
        from comm_functions.points_based_transform import load_points_based_transform
        from config.camera_config import load_camera_settings
        from pipeline.messages import (
            PredictorCommand,
            PredictorCommandKind,
            UartCommand,
            UartCommandKind,
            UartEventKind,
        )
        from pipeline.predictor_worker import PredictorWorker
        from pipeline.uart_worker import UartWorker
        from pipeline.vision_worker import VisionConfig, VisionWorker

        self.StereoTriangulator = StereoTriangulator
        self.load_points_based_transform = load_points_based_transform
        self.load_camera_settings = load_camera_settings
        self.PredictorCommand = PredictorCommand
        self.PredictorCommandKind = PredictorCommandKind
        self.UartCommand = UartCommand
        self.UartCommandKind = UartCommandKind
        self.UartEventKind = UartEventKind
        self.PredictorWorker = PredictorWorker
        self.UartWorker = UartWorker
        self.VisionConfig = VisionConfig
        self.VisionWorker = VisionWorker

    def _init_workers(self) -> None:
        cam = self.load_camera_settings()
        self._frame_w = int(cam["frame_width"])
        self._frame_h = int(cam["frame_height"])
        self._cam_left_id = int(cam["camera0"])
        self._cam_right_id = int(cam["camera1"])

        transform = self.load_points_based_transform()
        self._R = transform["rotation"]
        self._t = transform["translation"]
        self._cam_scale = float(transform["camera_scale_to_robot_units"])

        calibration_dir = self._stack_root / "camera_params" / "camera_properties"
        self._triangulator = self.StereoTriangulator(
            calibration_dir=str(calibration_dir),
            cam_left_id=self._cam_left_id,
            cam_right_id=self._cam_right_id,
        )

        self._position_queue: queue.Queue = queue.Queue(maxsize=256)
        self._observation_queue: queue.Queue = queue.Queue(maxsize=2)
        self._predictor_cmd_queue: queue.Queue = queue.Queue(maxsize=16)
        self._predictor_update_queue: queue.Queue = queue.Queue(maxsize=16)
        self._uart_cmd_queue: queue.Queue = queue.Queue(maxsize=16)
        self._uart_event_queue: queue.Queue = queue.Queue(maxsize=256)

        self._vision_worker = self.VisionWorker(
            triangulator=self._triangulator,
            transform_rotation=self._R,
            transform_translation=self._t,
            transform_scale=self._cam_scale,
            position_queue=self._position_queue,
            observation_queue=self._observation_queue,
            stop_event=self._stop_event,
            config=self.VisionConfig(
                frame_width=self._frame_w,
                frame_height=self._frame_h,
                reproj_err_max_px=10.0,
                warmup_s=float(self.cfg.warmup_s),
            ),
            status_printer=_print,
        )
        self._predictor_worker = self.PredictorWorker(
            position_queue=self._position_queue,
            predictor_cmd_queue=self._predictor_cmd_queue,
            predictor_update_queue=self._predictor_update_queue,
            stop_event=self._stop_event,
            status_printer=_print,
        )
        self._uart_worker = self.UartWorker(
            port=self.cfg.uart_port,
            baud_rate=int(self.cfg.baud_rate),
            verbose=bool(self.cfg.uart_verbose),
            uart_cmd_queue=self._uart_cmd_queue,
            uart_event_queue=self._uart_event_queue,
            stop_event=self._stop_event,
            tx_interval_s=float(self.cfg.tx_interval_s),
            time_aggression=1.0,
            status_printer=_print,
        )

    def _send_predictor(self, kind) -> None:
        _safe_put(self._predictor_cmd_queue, self.PredictorCommand(kind=kind))

    def _send_uart(self, kind, intercept=None) -> None:
        _safe_put(self._uart_cmd_queue, self.UartCommand(kind=kind, intercept=intercept))

    def _next_sample_id(self) -> int:
        if not self.dataset_csv.exists():
            return 1
        max_id = 0
        try:
            with self.dataset_csv.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        max_id = max(max_id, int(float(row.get("sample_id", 0))))
                    except Exception:
                        pass
        except Exception:
            return 1
        return max_id + 1 if max_id > 0 else 1

    def _reset_throw(self, reason: str) -> None:
        self._first_points = []
        self._last_accept_t = None
        self._latest_intercept = None
        self._last_sent_intercept = None
        self._throw_had_send = False
        self._status = f"Throw reset ({reason})."

    def _arm(self) -> None:
        self._state.gate_on = True
        self._review_mode = False
        self._pending_row = None
        self._reset_throw("arm")
        self._send_predictor(self.PredictorCommandKind.RESET)
        self._send_predictor(self.PredictorCommandKind.ENABLE)
        self._status = "Armed. Throw one ball."
        _print("[gate] ON")

    def _pause(self) -> None:
        self._state.gate_on = False
        self._state.intercept_inflight = False
        self._review_mode = False
        self._pending_row = None
        self._reset_throw("pause")
        self._send_predictor(self.PredictorCommandKind.DISABLE)
        self._send_predictor(self.PredictorCommandKind.RESET)
        self._status = "Paused. Press 'g' to arm."
        _print("[gate] OFF")

    def _build_row(self, points: List[object], intercept) -> Dict[str, float]:
        p_used = points[: self.cfg.num_points]
        if len(p_used) < self.cfg.num_points:
            raise ValueError("Insufficient accepted points")

        row: Dict[str, float] = {
            "sample_id": int(self.sample_id),
            "captured_at": datetime.now().isoformat(timespec="milliseconds"),
        }

        t_vals: List[float] = []
        for i, p in enumerate(p_used, start=1):
            row[f"x{i}"] = float(p.x_mm)
            row[f"y{i}"] = float(p.y_mm)
            row[f"z{i}"] = float(p.z_mm)
            t_vals.append(float(p.capture_time))

        for i in range(1, len(t_vals)):
            row[f"dt{i}{i+1}"] = float(t_vals[i] - t_vals[i - 1])

        row["x_hit"] = float(intercept.x_mm)
        row["y_hit"] = float(intercept.y_mm)
        row["z_hit"] = float(intercept.z_mm)
        row["vx_hit"] = float(intercept.vx_mm_s)
        row["vy_hit"] = float(intercept.vy_mm_s)
        row["vz_hit"] = float(intercept.vz_mm_s)

        t_hit_abs = float(intercept.source_capture_time) + float(intercept.intercept_time_s)
        row["t_hit"] = float(t_hit_abs - t_vals[-1])
        row["is_reachable"] = 0.0 if bool(intercept.clamped) else 1.0
        row["intercept_valid"] = 1.0
        row["bounces_before_hit"] = float(intercept.bounce_count)
        return row

    def _save_row(self, row: Dict[str, float]) -> None:
        write_header = not self.dataset_csv.exists()
        with self.dataset_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        self.sample_id += 1
        self.saved_samples += 1

    def _delete_last_saved(self) -> bool:
        if not self.dataset_csv.exists():
            self._status = "No dataset file yet."
            return False
        try:
            with self.dataset_csv.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames or [])
                rows = list(reader)
        except Exception as exc:
            self._status = f"Delete failed: {exc}"
            return False

        if not rows or not fieldnames:
            self._status = "No rows to delete."
            return False

        removed = rows.pop()
        try:
            with self.dataset_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        except Exception as exc:
            self._status = f"Delete failed: {exc}"
            return False

        self.sample_id = self._next_sample_id()
        self.saved_samples = max(0, self.saved_samples - 1)
        self._status = f"Deleted sample_id={removed.get('sample_id', '?')}"
        return True

    def _wait_for_startup_home(self) -> bool:
        deadline = None
        if self.cfg.home_ack_timeout_s is not None:
            deadline = time.perf_counter() + float(self.cfg.home_ack_timeout_s)

        opened = False
        home_sent = False
        while not self._stop_event.is_set():
            if deadline is not None and time.perf_counter() > deadline and not self._state.startup_homing_done:
                _print("[startup] HOME/IDLE timeout.")
                return False
            try:
                evt = self._uart_event_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if evt.kind == self.UartEventKind.UART_OPENED:
                opened = True
                self._send_uart(self.UartCommandKind.HOME)
                continue
            if evt.kind == self.UartEventKind.TX_HOME:
                home_sent = True
                continue
            if evt.kind == self.UartEventKind.UART_ERROR:
                _print(f"[startup] UART error: {evt.data}")
                return False

            if evt.kind == self.UartEventKind.HOME_DONE:
                self._state.startup_homing_done = True
                self._state.robot_at_home = True
                self._state.robot_state = "IDLE"
                return True

            if evt.kind == self.UartEventKind.STATE_IDLE and opened and home_sent:
                self._state.startup_homing_done = True
                self._state.robot_at_home = True
                self._state.robot_state = "IDLE"
                return True

        return False

    def _handle_uart_event(self, evt) -> None:
        if evt.kind == self.UartEventKind.STATE_OFF:
            self._state.robot_state = "OFF"
        elif evt.kind == self.UartEventKind.STATE_PLAN:
            self._state.robot_state = "PLAN"
        elif evt.kind == self.UartEventKind.STATE_MOVE:
            self._state.robot_state = "MOVE"
            self._state.robot_at_home = False
        elif evt.kind == self.UartEventKind.STATE_IDLE:
            self._state.robot_state = "IDLE"
        elif evt.kind == self.UartEventKind.TX_HOME:
            self._state.robot_at_home = False
        elif evt.kind == self.UartEventKind.TX_INTERCEPT:
            is_update = bool(evt.data.get("is_update", False))
            self._state.intercept_inflight = True
            self._state.robot_at_home = False
            if is_update:
                self._state.updates_sent += 1
            else:
                self._state.throws_sent += 1
            self._state.last_latency_ms = float(evt.data.get("latency_ms", 0.0))
            self._state.last_adjusted_ms = float(evt.data.get("adjusted_time_ms", 0.0))
        elif evt.kind == self.UartEventKind.INTERCEPT_DONE:
            if self._state.intercept_inflight:
                self._state.intercept_inflight = False
                self._send_uart(self.UartCommandKind.HOME)
        elif evt.kind == self.UartEventKind.HOME_DONE:
            if self._state.robot_at_home:
                return
            self._state.robot_at_home = True
            self._state.intercept_inflight = False
            self._state.robot_state = "IDLE"

            if self._throw_had_send and self._last_sent_intercept is not None and len(self._first_points) >= self.cfg.num_points:
                try:
                    self._pending_row = self._build_row(self._first_points, self._last_sent_intercept)
                    self._review_mode = True
                    self._state.gate_on = False
                    self._send_predictor(self.PredictorCommandKind.DISABLE)
                    self._send_predictor(self.PredictorCommandKind.RESET)
                    self._status = "Review: k=keep, x=discard"
                except Exception as exc:
                    self.failed_samples += 1
                    self._status = f"Sample build failed: {exc}"
                    self._arm()
            else:
                if self._state.gate_on:
                    self.failed_samples += 1
                    self._reset_throw("home_without_sample")
                    self._send_predictor(self.PredictorCommandKind.RESET)
                    self._send_predictor(self.PredictorCommandKind.ENABLE)
                    self._status = "Throw ended without valid sample; ready for next."
                else:
                    self._status = "Robot homed."
        elif evt.kind in (self.UartEventKind.TARGET_REJECTED, self.UartEventKind.PLAN_FAILED):
            self.failed_samples += 1
            self._state.intercept_inflight = False
            self._state.robot_at_home = False
            self._send_uart(self.UartCommandKind.HOME)
            self._reset_throw("planner_reject")
            self._status = "Planner rejected throw; discarded."

    def _handle_predictor_update(self, update) -> None:
        self._latest_predictor_update = update
        if self._review_mode or not self._state.gate_on:
            return

        if update.intercept is not None:
            self._latest_intercept = update.intercept

        if not bool(update.accepted) or update.sample is None:
            return

        # Freeze sample capture after first intercept send.
        # We do not want post-send gaps to reset the throw before HOME_DONE.
        if self._throw_had_send or self._state.intercept_inflight:
            return

        t_now = float(update.sample.capture_time)
        if self._last_accept_t is not None and (t_now - self._last_accept_t) > float(self.cfg.gap_reset_s):
            self._reset_throw(f"gap>{self.cfg.gap_reset_s:.3f}s")
        self._last_accept_t = t_now

        if len(self._first_points) < self.cfg.num_points:
            self._first_points.append(update.sample)

    def _can_send_intercept(self) -> bool:
        if self._review_mode:
            return False
        if not self._state.startup_homing_done:
            return False
        if not self._state.gate_on:
            return False
        if not self._state.robot_at_home:
            return False
        if self._state.intercept_inflight:
            return False
        if self._latest_intercept is None:
            return False
        if len(self._first_points) < self.cfg.num_points:
            return False
        if int(self._latest_intercept.buffer_points) < int(self.cfg.num_points):
            return False
        return True

    def _draw_overlay(self) -> None:
        if self._latest_obs is None or self._latest_obs.left_frame is None:
            return
        vis = self._latest_obs.left_frame.copy()

        cv2.putText(
            vis,
            f"FPS:{self._fps:.0f} Gate:{'ON' if self._state.gate_on else 'OFF'} Robot:{self._state.robot_state}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0) if self._state.gate_on else (0, 0, 255),
            1,
        )
        cv2.putText(
            vis,
            f"Saved:{self.saved_samples} Discarded:{self.discarded_samples} Failed:{self.failed_samples}",
            (10, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            vis,
            f"Points:{len(self._first_points)}/{self.cfg.num_points} Inflight:{self._state.intercept_inflight}",
            (10, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (180, 255, 180),
            1,
        )

        if self._latest_predictor_update is not None and self._latest_predictor_update.sample is not None:
            s = self._latest_predictor_update.sample
            cv2.putText(
                vis,
                f"Ball(mm): X={s.x_mm:+.1f} Y={s.y_mm:+.1f} Z={s.z_mm:+.1f}",
                (10, 91),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 0),
                1,
            )

        if self._latest_intercept is not None:
            ip = self._latest_intercept
            cv2.putText(
                vis,
                f"Pred(mm): X={ip.x_mm:+.1f} Y={ip.y_mm:+.1f} Z={ip.z_mm:+.1f} t={ip.intercept_time_s*1000:.1f}ms",
                (10, 114),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 220, 220),
                1,
            )

        cv2.putText(
            vis,
            self._status,
            (10, 137),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (250, 230, 120),
            1,
        )
        cv2.putText(
            vis,
            "Controls: g=arm/pause h=home r=reset d=delete-last k=keep x=discard q=quit",
            (10, vis.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (180, 180, 180),
            1,
        )

        dw = int(self.cfg.preview_width)
        dh = int(dw * self._frame_h / self._frame_w)
        cv2.imshow("Predictor Verified Collector", cv2.resize(vis, (dw, dh)))

    def run(self) -> int:
        _print("Starting predictor-verified collector")
        _print(f"Output: {self.dataset_csv.resolve()}")

        self._uart_worker.start()
        if not self._wait_for_startup_home():
            self._stop_event.set()
            return 1

        self._predictor_worker.start()
        self._vision_worker.start()
        self._send_predictor(self.PredictorCommandKind.DISABLE)
        self._send_predictor(self.PredictorCommandKind.RESET)

        try:
            while not self._stop_event.is_set():
                drained = False

                while True:
                    try:
                        evt = self._uart_event_queue.get_nowait()
                    except queue.Empty:
                        break
                    drained = True
                    self._handle_uart_event(evt)

                while True:
                    try:
                        upd = self._predictor_update_queue.get_nowait()
                    except queue.Empty:
                        break
                    drained = True
                    self._handle_predictor_update(upd)

                while True:
                    try:
                        obs = self._observation_queue.get_nowait()
                    except queue.Empty:
                        break
                    drained = True
                    self._latest_obs = obs
                    self._fps_count += 1
                    if self._fps_count % 30 == 0:
                        now = time.perf_counter()
                        self._fps = 30.0 / max(1e-6, now - self._fps_t0)
                        self._fps_t0 = now

                if self._can_send_intercept():
                    self._send_uart(self.UartCommandKind.INTERCEPT, intercept=self._latest_intercept)
                    self._last_sent_intercept = self._latest_intercept
                    self._state.intercept_inflight = True
                    self._state.robot_at_home = False
                    self._throw_had_send = True
                    self._latest_intercept = None
                    # This collector is one-shot per throw (no pre-move update stream).
                    self._send_predictor(self.PredictorCommandKind.DISABLE)

                self._draw_overlay()
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self._send_uart(self.UartCommandKind.HOME)
                    break
                elif key == ord("g"):
                    if self._review_mode:
                        self._status = "Finish review first: k=keep or x=discard"
                    elif self._state.gate_on:
                        self._pause()
                    else:
                        self._arm()
                elif key == ord("h"):
                    self._pause()
                    self._send_uart(self.UartCommandKind.HOME)
                elif key == ord("r"):
                    if not self._review_mode:
                        self._reset_throw("manual_reset")
                        self._send_predictor(self.PredictorCommandKind.RESET)
                elif key == ord("d"):
                    self._delete_last_saved()
                elif key == ord("k"):
                    if self._review_mode and self._pending_row is not None:
                        self._save_row(self._pending_row)
                        _print(f"Saved sample #{int(self._pending_row['sample_id'])}")
                        self._arm()
                elif key == ord("x"):
                    if self._review_mode:
                        self.discarded_samples += 1
                        _print("Discarded pending sample.")
                        self._arm()

                if not drained:
                    time.sleep(0.002)

        except KeyboardInterrupt:
            _print("Interrupted by user")
        finally:
            self._stop_event.set()
            self._send_predictor(self.PredictorCommandKind.SHUTDOWN)
            self._send_uart(self.UartCommandKind.SHUTDOWN)
            self._vision_worker.join(timeout=2.0)
            self._predictor_worker.join(timeout=2.0)
            self._uart_worker.join(timeout=2.0)
            cv2.destroyAllWindows()

        _print(
            f"Done. Saved={self.saved_samples}, Discarded={self.discarded_samples}, Failed={self.failed_samples}"
        )
        return 0


def parse_args() -> CaptureConfig:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Collect predictor-verified transfer-learning data")
    parser.add_argument("--stack-root", type=str, default=str(repo_root / "new_top_level"))
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "real_data_collected"),
    )
    parser.add_argument("--output-file", type=str, default="real_transfer_predictor_verified.csv")
    parser.add_argument("--num-points", type=int, default=6)
    parser.add_argument("--port", default=os.environ.get("STM32_UART_PORT"))
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--warmup-s", type=float, default=2.0)
    parser.add_argument("--tx-interval-ms", type=float, default=30.0)
    parser.add_argument("--gap-reset-ms", type=float, default=120.0)
    parser.add_argument("--preview-width", type=int, default=960)
    parser.add_argument("--home-ack-timeout", type=float, default=30.0)
    parser.add_argument("--quiet-uart", action="store_true")
    args = parser.parse_args()

    if not args.port:
        parser.error("UART port required. Pass --port or set STM32_UART_PORT.")
    if int(args.num_points) < 4:
        parser.error("--num-points must be >= 4")

    timeout = None if float(args.home_ack_timeout) == 0.0 else float(args.home_ack_timeout)
    return CaptureConfig(
        stack_root=Path(args.stack_root),
        output_dir=Path(args.output_dir),
        output_file=str(args.output_file),
        num_points=max(4, int(args.num_points)),
        uart_port=str(args.port),
        baud_rate=int(args.baud),
        warmup_s=max(0.0, float(args.warmup_s)),
        tx_interval_s=max(0.0, float(args.tx_interval_ms) / 1000.0),
        home_ack_timeout_s=timeout,
        gap_reset_s=max(0.02, float(args.gap_reset_ms) / 1000.0),
        preview_width=max(320, int(args.preview_width)),
        uart_verbose=not bool(args.quiet_uart),
    )


def main() -> int:
    cfg = parse_args()
    app = PredictorVerifiedCollector(cfg)
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
