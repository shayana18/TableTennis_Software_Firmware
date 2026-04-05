#!/usr/bin/env python3
"""
ANN-driven game runtime:
  1) Stereo triangulation -> robot-frame points
  2) Collect first K points of a throw (default K=6)
  3) Run trained ANN to predict [x_hit, y_hit, z_hit, vx_hit, vy_hit, vz_hit, t_hit]
  4) Send prediction over UART using same worker path as new_top_level/game.py

This script intentionally bypasses RobotPredictor planning logic.
"""

from __future__ import annotations

import argparse
import os
import pickle
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def _print(*args, **kwargs):
    print("from planner in terminal", *args, **kwargs)


class IdentityScaler:
    def transform(self, x: np.ndarray) -> np.ndarray:
        return x

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x


class ArrayScaler:
    """Simple StandardScaler-compatible wrapper from saved mean/scale arrays."""

    def __init__(self, mean: np.ndarray, scale: np.ndarray):
        self.mean_ = np.asarray(mean, dtype=float)
        self.scale_ = np.asarray(scale, dtype=float)
        self.scale_[self.scale_ == 0.0] = 1.0

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.scale_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.scale_ + self.mean_


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
    last_latency_ms: float = 0.0
    last_adjusted_ms: float = 0.0
    last_uart_line: str = ""


class AnnGameApp:
    MIN_SEND_BUFFER = 4

    def __init__(
        self,
        *,
        stack_root: Path,
        model_path: Path,
        scaler_x_path: Optional[Path],
        scaler_y_path: Optional[Path],
        num_points: int,
        uart_port: str,
        baud_rate: int,
        home_ack_timeout_s: Optional[float],
        warmup_s: float,
        tx_interval_s: float,
        uart_verbose: bool,
        gap_reset_s: float,
        stale_reset_s: float,
        min_t_hit_s: float,
        max_t_hit_s: float,
        preview_width: int,
    ) -> None:
        self._stack_root = stack_root.resolve()
        self._model_path = model_path.resolve()
        self._num_points = int(num_points)
        self._home_ack_timeout_s = home_ack_timeout_s
        self._preview_width = int(preview_width)
        self._gap_reset_s = float(gap_reset_s)
        self._stale_reset_s = float(stale_reset_s)
        self._min_t_hit_s = float(min_t_hit_s)
        self._max_t_hit_s = float(max_t_hit_s)

        self._state = RuntimeState()
        self._stop_event = threading.Event()

        self._load_stack_modules()
        self._load_model_and_scalers(scaler_x_path, scaler_y_path)

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
                warmup_s=warmup_s,
            ),
            status_printer=_print,
        )
        self._uart_worker = self.UartWorker(
            port=uart_port,
            baud_rate=baud_rate,
            verbose=uart_verbose,
            uart_cmd_queue=self._uart_cmd_queue,
            uart_event_queue=self._uart_event_queue,
            stop_event=self._stop_event,
            tx_interval_s=tx_interval_s,
            time_aggression=1.0,
            status_printer=_print,
        )

        self._latest_obs = None
        self._latest_sample = None
        self._latest_intercept = None
        self._fps = 0.0
        self._fps_count = 0
        self._fps_t0 = time.perf_counter()

        self._throw_points = []
        self._last_sample_time = None
        self._last_throw_reset_reason = "init"

    # ------------------------------------------------------------------
    # Imports / model loading
    # ------------------------------------------------------------------
    def _load_stack_modules(self) -> None:
        if str(self._stack_root) not in sys.path:
            sys.path.insert(0, str(self._stack_root))

        from ball_tracking.stereo_triangulator import StereoTriangulator
        from comm_functions.points_based_transform import load_points_based_transform
        from config.camera_config import load_camera_settings
        from pipeline.messages import (
            InterceptPlan,
            UartCommand,
            UartCommandKind,
            UartEvent,
            UartEventKind,
        )
        from pipeline.uart_worker import UartWorker
        from pipeline.vision_worker import VisionConfig, VisionWorker

        self.StereoTriangulator = StereoTriangulator
        self.load_points_based_transform = load_points_based_transform
        self.load_camera_settings = load_camera_settings
        self.InterceptPlan = InterceptPlan
        self.UartCommand = UartCommand
        self.UartCommandKind = UartCommandKind
        self.UartEvent = UartEvent
        self.UartEventKind = UartEventKind
        self.UartWorker = UartWorker
        self.VisionConfig = VisionConfig
        self.VisionWorker = VisionWorker

    @staticmethod
    def _load_scaler(path: Optional[Path], name: str):
        if path is None:
            _print(f"[model] {name}: identity (no scaler path provided)")
            return IdentityScaler()

        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"{name} scaler file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".npz":
            arr = np.load(path)
            if "mean_" not in arr or "scale_" not in arr:
                raise KeyError(f"{name} .npz must contain keys 'mean_' and 'scale_': {path}")
            scaler = ArrayScaler(mean=arr["mean_"], scale=arr["scale_"])
            _print(f"[model] Loaded {name} scaler from npz: {path}")
            return scaler

        # .joblib / .pkl / generic pickle object with transform APIs
        try:
            import joblib  # type: ignore

            scaler = joblib.load(path)
            _print(f"[model] Loaded {name} scaler via joblib: {path}")
            return scaler
        except Exception:
            with path.open("rb") as f:
                scaler = pickle.load(f)
            _print(f"[model] Loaded {name} scaler via pickle: {path}")
            return scaler

    def _load_model_and_scalers(self, scaler_x_path: Optional[Path], scaler_y_path: Optional[Path]) -> None:
        import tensorflow as tf

        if not self._model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self._model_path}")

        self._model = tf.keras.models.load_model(self._model_path, compile=False)
        self._scx = self._load_scaler(scaler_x_path, "X")
        self._scy = self._load_scaler(scaler_y_path, "Y")

        try:
            in_shape = self._model.input_shape
        except Exception:
            in_shape = None
        _print(f"[model] Loaded model: {self._model_path}")
        _print(f"[model] input_shape={in_shape}")

        expected_features = 3 * self._num_points + (self._num_points - 1)
        if isinstance(in_shape, (tuple, list)) and in_shape and in_shape[-1] is not None:
            model_features = int(in_shape[-1])
            if model_features != expected_features:
                inferred_points = None
                if (model_features + 1) % 4 == 0:
                    cand = (model_features + 1) // 4
                    if cand >= 4:
                        inferred_points = int(cand)

                if inferred_points is not None:
                    _print(
                        f"[model] Auto-adjusting num_points from {self._num_points} to {inferred_points} "
                        f"based on model input size ({model_features} features)."
                    )
                    self._num_points = inferred_points
                    expected_features = model_features
                else:
                    _print(
                        f"[WARN] Model expects {model_features} features, but num_points={self._num_points} "
                        f"implies {expected_features} features."
                    )
            else:
                _print(
                    f"[model] Feature size matches num_points={self._num_points} ({expected_features} features)."
                )

    # ------------------------------------------------------------------
    # UART helpers
    # ------------------------------------------------------------------
    def _send_uart(self, kind, intercept=None) -> None:
        _safe_put(self._uart_cmd_queue, self.UartCommand(kind=kind, intercept=intercept))

    def _reset_throw(self, reason: str) -> None:
        self._throw_points = []
        self._last_sample_time = None
        self._last_throw_reset_reason = reason

    def _wait_for_startup_home(self) -> bool:
        deadline = None
        if self._home_ack_timeout_s is not None:
            deadline = time.perf_counter() + float(self._home_ack_timeout_s)

        opened = False
        home_cmd_sent = False
        while not self._stop_event.is_set():
            if deadline is not None and time.perf_counter() > deadline and not self._state.startup_homing_done:
                _print("[startup] Home/idle confirmation timeout.")
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
                home_cmd_sent = True
                continue
            if evt.kind == self.UartEventKind.UART_ERROR:
                _print(f"[startup] UART error: {evt.data}")
                return False

            if evt.line:
                self._state.last_uart_line = evt.line
                print(f"[UART][RX] {evt.line}")

            if evt.kind == self.UartEventKind.HOME_DONE:
                self._state.startup_homing_done = True
                self._state.robot_at_home = True
                self._state.robot_state = "IDLE"
                _print("[startup] Home complete, robot is IDLE. Starting ANN loop.")
                return True

            if (
                evt.kind == self.UartEventKind.STATE_IDLE
                and opened
                and home_cmd_sent
                and not self._state.startup_homing_done
            ):
                self._state.startup_homing_done = True
                self._state.robot_at_home = True
                self._state.robot_state = "IDLE"
                _print("[startup] Robot reached IDLE. Starting ANN loop.")
                return True

        return False

    def _handle_uart_event(self, evt) -> None:
        if evt.line:
            self._state.last_uart_line = evt.line
            print(f"[UART][RX] {evt.line}")

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
            self._state.intercept_inflight = True
            self._state.robot_at_home = False
            self._state.throws_sent += 1
            self._state.last_latency_ms = float(evt.data.get("latency_ms", 0.0))
            self._state.last_adjusted_ms = float(evt.data.get("adjusted_time_ms", 0.0))
        elif evt.kind == self.UartEventKind.INTERCEPT_DONE:
            if not self._state.intercept_inflight:
                return
            self._state.intercept_inflight = False
            self._send_uart(self.UartCommandKind.HOME)
            self._latest_intercept = None
            self._reset_throw("intercept_done")
        elif evt.kind == self.UartEventKind.HOME_DONE:
            if self._state.robot_at_home:
                return
            self._state.robot_at_home = True
            self._state.intercept_inflight = False
            self._state.robot_state = "IDLE"
            self._latest_intercept = None
            self._reset_throw("home_done")
        elif evt.kind in (self.UartEventKind.TARGET_REJECTED, self.UartEventKind.PLAN_FAILED):
            self._state.intercept_inflight = False
            self._state.robot_at_home = False
            self._send_uart(self.UartCommandKind.HOME)
            self._latest_intercept = None
            self._reset_throw("target_rejected_or_plan_failed")
        elif evt.kind == self.UartEventKind.UART_ERROR:
            _print(f"[uart] ERROR: {evt.data}")

    # ------------------------------------------------------------------
    # ANN inference
    # ------------------------------------------------------------------
    def _build_feature_vector(self, points) -> np.ndarray:
        coords = []
        times = []
        for p in points:
            coords.extend([float(p.x_mm), float(p.y_mm), float(p.z_mm)])
            times.append(float(p.capture_time))
        dts = [times[i] - times[i - 1] for i in range(1, len(times))]
        feat = np.asarray(coords + dts, dtype=np.float32).reshape(1, -1)
        return feat

    def _predict_from_points(self, points):
        x_raw = self._build_feature_vector(points)
        x_scaled = self._scx.transform(x_raw)
        y_scaled = self._model.predict(x_scaled, verbose=0)
        y_raw = self._scy.inverse_transform(y_scaled)
        vals = np.asarray(y_raw, dtype=float).reshape(-1)
        if vals.size < 7 or not np.all(np.isfinite(vals[:7])):
            return None

        t_hit_s = float(vals[6])
        t_hit_s = max(self._min_t_hit_s, min(self._max_t_hit_s, t_hit_s))

        p_last = points[-1]
        plan = self.InterceptPlan(
            source_frame_id=int(p_last.frame_id),
            source_capture_time=float(p_last.capture_time),
            x_mm=float(vals[0]),
            y_mm=float(vals[1]),
            z_mm=float(vals[2]),
            vx_mm_s=float(vals[3]),
            vy_mm_s=float(vals[4]),
            vz_mm_s=float(vals[5]),
            intercept_time_s=float(t_hit_s),
            confidence=1.0,
            clamped=False,
            buffer_points=len(points),
            bounce_count=0,
        )
        return plan

    def _can_send_initial_intercept(self) -> bool:
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
        required_points = max(self.MIN_SEND_BUFFER, self._num_points)
        if self._latest_intercept.buffer_points < required_points:
            return False
        return True

    def _process_sample(self, sample) -> None:
        if not self._state.gate_on:
            return
        if not self._state.startup_homing_done:
            return
        if not self._state.robot_at_home:
            return
        if self._state.intercept_inflight:
            return

        t_now = float(sample.capture_time)
        if self._last_sample_time is not None and (t_now - self._last_sample_time) > self._gap_reset_s:
            self._reset_throw(f"gap>{self._gap_reset_s:.3f}s")
        self._last_sample_time = t_now

        if len(self._throw_points) < self._num_points:
            self._throw_points.append(sample)

        if len(self._throw_points) == self._num_points and self._latest_intercept is None:
            plan = self._predict_from_points(self._throw_points)
            if plan is not None:
                self._latest_intercept = plan
                _print(
                    f"[ann] Pred target: X={plan.x_mm:+.1f} Y={plan.y_mm:+.1f} Z={plan.z_mm:+.1f} "
                    f"V=({plan.vx_mm_s:+.1f},{plan.vy_mm_s:+.1f},{plan.vz_mm_s:+.1f}) "
                    f"t_hit={plan.intercept_time_s*1000:.1f}ms"
                )
            else:
                _print("[ann] Prediction failed (non-finite output). Resetting throw buffer.")
                self._reset_throw("pred_failed")

    # ------------------------------------------------------------------
    # UI / loop
    # ------------------------------------------------------------------
    def _draw_overlay(self) -> None:
        if self._latest_obs is None or self._latest_obs.left_frame is None:
            return
        vis = self._latest_obs.left_frame.copy()

        gate_txt = "ON" if self._state.gate_on else "OFF"
        cv2.putText(
            vis,
            f"FPS:{self._fps:.0f} Gate:{gate_txt} Robot:{self._state.robot_state}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0) if self._state.gate_on else (0, 0, 255),
            1,
        )
        cv2.putText(
            vis,
            f"Home:{self._state.robot_at_home} Inflight:{self._state.intercept_inflight} Throws:{self._state.throws_sent}",
            (10, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        cv2.putText(
            vis,
            f"ANN points: {len(self._throw_points)}/{self._num_points}  Last reset: {self._last_throw_reset_reason}",
            (10, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (200, 255, 200),
            1,
        )

        if self._latest_sample is not None:
            s = self._latest_sample
            cv2.putText(
                vis,
                f"Ball(mm): X={s.x_mm:+.1f} Y={s.y_mm:+.1f} Z={s.z_mm:+.1f}",
                (10, 91),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.47,
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
                0.47,
                (0, 220, 220),
                1,
            )

        cv2.putText(
            vis,
            f"Latency:{self._state.last_latency_ms:.1f}ms Adjusted:{self._state.last_adjusted_ms:.1f}ms",
            (10, 137),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 220, 120),
            1,
        )
        cv2.putText(
            vis,
            "Controls: g=toggle gameplay  h=home  r=reset throw  q=quit",
            (10, vis.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (180, 180, 180),
            1,
        )

        dw = self._preview_width
        dh = int(dw * self._frame_h / self._frame_w)
        cv2.imshow("ANN Game", cv2.resize(vis, (dw, dh)))

    def run(self) -> int:
        _print("\n" + "=" * 70)
        _print("ANN GAME PIPELINE")
        _print("=" * 70)
        _print(f"Cameras: L={self._cam_left_id} R={self._cam_right_id}")
        _print(f"Model: {self._model_path}")
        _print(f"Input points: {self._num_points}")
        _print("=" * 70)

        self._uart_worker.start()
        if not self._wait_for_startup_home():
            self._stop_event.set()
            return 1

        self._vision_worker.start()

        try:
            while not self._stop_event.is_set():
                drained_any = False

                while True:
                    try:
                        evt = self._uart_event_queue.get_nowait()
                    except queue.Empty:
                        break
                    drained_any = True
                    self._handle_uart_event(evt)

                while True:
                    try:
                        obs = self._observation_queue.get_nowait()
                    except queue.Empty:
                        break
                    drained_any = True
                    self._latest_obs = obs
                    self._fps_count += 1
                    if self._fps_count % 30 == 0:
                        now = time.perf_counter()
                        self._fps = 30.0 / max(1e-6, now - self._fps_t0)
                        self._fps_t0 = now

                while True:
                    try:
                        sample = self._position_queue.get_nowait()
                    except queue.Empty:
                        break
                    drained_any = True
                    self._latest_sample = sample
                    self._process_sample(sample)

                if (
                    self._state.gate_on
                    and not self._state.intercept_inflight
                    and self._last_sample_time is not None
                ):
                    idle_t = time.perf_counter() - self._last_sample_time
                    if idle_t > self._stale_reset_s and self._throw_points:
                        self._reset_throw(f"stale>{self._stale_reset_s:.3f}s")

                if self._can_send_initial_intercept():
                    self._send_uart(self.UartCommandKind.INTERCEPT, intercept=self._latest_intercept)
                    self._state.intercept_inflight = True
                    self._state.robot_at_home = False

                self._draw_overlay()
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    _print("[quit] Stopping app, sending HOME.")
                    self._send_uart(self.UartCommandKind.HOME)
                    break
                elif key == ord("g"):
                    self._state.gate_on = not self._state.gate_on
                    self._reset_throw("gate_toggle")
                    self._latest_intercept = None
                    _print(f"[gate] {'ON' if self._state.gate_on else 'OFF'}")
                elif key == ord("h"):
                    self._state.gate_on = False
                    self._state.intercept_inflight = False
                    self._latest_intercept = None
                    self._reset_throw("manual_home")
                    self._send_uart(self.UartCommandKind.HOME)
                    _print("[home] Manual HOME command sent.")
                elif key == ord("r"):
                    self._latest_intercept = None
                    self._reset_throw("manual_reset")
                    _print("[ann] Throw buffer reset.")

                if not drained_any:
                    time.sleep(0.002)

        except KeyboardInterrupt:
            _print("[ctrl-c] stopping...")
        finally:
            self._stop_event.set()
            self._send_uart(self.UartCommandKind.SHUTDOWN)
            self._vision_worker.join(timeout=2.0)
            self._uart_worker.join(timeout=2.0)
            cv2.destroyAllWindows()

        _print(
            f"Done. Throws sent={self._state.throws_sent} "
            f"Last latency={self._state.last_latency_ms:.1f}ms"
        )
        return 0


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="ANN-driven ping pong runtime using first K observed points.")
    parser.add_argument(
        "--stack-root",
        type=str,
        default=str(repo_root / "new_top_level"),
        help="Path to new_top_level folder.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained Keras model (.keras/.h5).",
    )
    parser.add_argument(
        "--scaler-x",
        type=str,
        default="",
        help="Optional scaler for ANN inputs (joblib/pkl/npz with mean_/scale_).",
    )
    parser.add_argument(
        "--scaler-y",
        type=str,
        default="",
        help="Optional scaler for ANN outputs (joblib/pkl/npz with mean_/scale_).",
    )
    parser.add_argument("--num-points", type=int, default=6, help="Number of initial points fed to ANN.")
    parser.add_argument(
        "--port",
        default=os.environ.get("STM32_UART_PORT"),
        help="UART port (or set STM32_UART_PORT)",
    )
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument(
        "--home-ack-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for startup home->IDLE (0=infinite)",
    )
    parser.add_argument("--tx-interval-ms", type=float, default=30.0)
    parser.add_argument("--warmup-s", type=float, default=2.0)
    parser.add_argument("--gap-reset-ms", type=float, default=120.0, help="Reset throw if sample gap exceeds this.")
    parser.add_argument("--stale-reset-ms", type=float, default=350.0, help="Reset throw if no new sample for this long.")
    parser.add_argument("--min-t-hit-ms", type=float, default=20.0, help="Minimum commanded intercept time.")
    parser.add_argument("--max-t-hit-ms", type=float, default=2000.0, help="Maximum commanded intercept time.")
    parser.add_argument("--preview-width", type=int, default=960)
    parser.add_argument("--quiet-uart", action="store_true")
    args = parser.parse_args()

    if not args.port:
        parser.error("UART port required. Pass --port or set STM32_UART_PORT.")
    if int(args.num_points) < 4:
        parser.error("--num-points must be >= 4.")

    return args


def main() -> int:
    args = parse_args()

    timeout = None if float(args.home_ack_timeout) == 0.0 else float(args.home_ack_timeout)
    app = AnnGameApp(
        stack_root=Path(args.stack_root),
        model_path=Path(args.model),
        scaler_x_path=Path(args.scaler_x).resolve() if args.scaler_x else None,
        scaler_y_path=Path(args.scaler_y).resolve() if args.scaler_y else None,
        num_points=int(args.num_points),
        uart_port=str(args.port),
        baud_rate=int(args.baud),
        home_ack_timeout_s=timeout,
        warmup_s=max(0.0, float(args.warmup_s)),
        tx_interval_s=max(0.0, float(args.tx_interval_ms) / 1000.0),
        uart_verbose=not bool(args.quiet_uart),
        gap_reset_s=max(0.01, float(args.gap_reset_ms) / 1000.0),
        stale_reset_s=max(0.02, float(args.stale_reset_ms) / 1000.0),
        min_t_hit_s=max(0.001, float(args.min_t_hit_ms) / 1000.0),
        max_t_hit_s=max(0.01, float(args.max_t_hit_ms) / 1000.0),
        preview_width=max(320, int(args.preview_width)),
    )
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
