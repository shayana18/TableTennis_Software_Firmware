#!/usr/bin/env python3
"""
Collect real transfer-learning samples from live stereo triangulation.

Workflow:
1. Run script, wait for warmup.
2. Press `s` to arm.
3. Throw one ball.
4. Script records first 4 accepted robot-frame points + first in-workspace intercept.
5. Appends one row to a single dataset CSV.
6. Automatically re-arms for the next throw (no reset required).

No UART is used here; this is a pure data-capture tool.

t_hit definition in saved CSV:
  t_hit = time from point4 to intercept [s]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def _print(*args, **kwargs) -> None:
    print("[collect_real]", *args, **kwargs)


@dataclass
class CaptureConfig:
    stack_root: Path
    output_dir: Path
    output_file_name: str
    warmup_s: float
    reproj_err_max_px: float
    max_throw_s: float
    max_lost_frames: int
    post_throw_cooldown_s: float
    preview_width: int
    show_stereo_debug: bool


class RealTransferCollector:
    STATE_DISARMED = "DISARMED"
    STATE_ARMED = "ARMED"
    STATE_TRACKING = "TRACKING"

    def __init__(self, cfg: CaptureConfig) -> None:
        self.cfg = cfg
        self.output_dir = cfg.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_csv = self.output_dir / cfg.output_file_name
        self.sample_id = self._next_sample_id()
        self.cooldown_until_ts = 0.0

        self._load_stack_modules(cfg.stack_root)
        self._init_vision_stack(cfg.stack_root)

        self.state = self.STATE_DISARMED
        self.status_msg = "Press 's' to arm capture."
        self.saved_samples = 0
        self.failed_throws = 0

        self._reset_throw_buffers()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _load_stack_modules(self, stack_root: Path) -> None:
        stack_root = stack_root.resolve()
        if str(stack_root) not in sys.path:
            sys.path.insert(0, str(stack_root))

        try:
            from ball_tracking.stereo_triangulator import StereoTriangulator
            from comm_functions.points_based_transform import load_points_based_transform, cam_to_robot
            from config.camera_config import load_camera_settings
            from estimation.robot_predictor import RobotPredictor
            from estimation.workspace import in_workspace, ELLIPSE_A, ELLIPSE_B, Z_MIN, Z_MAX
        except Exception as exc:
            raise RuntimeError(
                f"Failed importing new_top_level modules from '{stack_root}': {exc}"
            ) from exc

        self.StereoTriangulator = StereoTriangulator
        self.load_points_based_transform = load_points_based_transform
        self.cam_to_robot = cam_to_robot
        self.load_camera_settings = load_camera_settings
        self.RobotPredictor = RobotPredictor
        self.in_workspace = in_workspace
        self.ELLIPSE_A = ELLIPSE_A
        self.ELLIPSE_B = ELLIPSE_B
        self.Z_MIN = Z_MIN
        self.Z_MAX = Z_MAX

    def _init_vision_stack(self, stack_root: Path) -> None:
        settings = self.load_camera_settings()
        self.frame_width = int(settings["frame_width"])
        self.frame_height = int(settings["frame_height"])
        self.cam_left_id = int(settings["camera0"])
        self.cam_right_id = int(settings["camera1"])

        tf = self.load_points_based_transform()
        self.R = tf["rotation"]
        self.t_vec = tf["translation"]
        self.cam_scale = float(tf["camera_scale_to_robot_units"])

        calibration_dir = stack_root / "camera_params" / "camera_properties"
        self.triangulator = self.StereoTriangulator(
            calibration_dir=str(calibration_dir),
            cam_left_id=self.cam_left_id,
            cam_right_id=self.cam_right_id,
        )
        self.predictor = self.RobotPredictor()

    # ------------------------------------------------------------------
    # Throw state
    # ------------------------------------------------------------------

    def _reset_throw_buffers(self) -> None:
        self.predictor.reset()
        self.first_points: List[Tuple[float, float, float, float]] = []
        self.tracking_start_ts: Optional[float] = None
        self.last_robot_point: Optional[Tuple[float, float, float]] = None
        self.last_intercept: Optional[Dict[str, float]] = None
        self.lost_frames = 0

    def arm(self, msg: str = "Armed. Throw one ball now.") -> None:
        self._reset_throw_buffers()
        self.state = self.STATE_ARMED
        self.status_msg = msg

    def disarm(self, msg: str = "Capture paused. Press 's' to arm.") -> None:
        self._reset_throw_buffers()
        self.state = self.STATE_DISARMED
        self.cooldown_until_ts = 0.0
        self.status_msg = msg

    def _complete_throw(self, success: bool, frame_ts: float, msg: str) -> None:
        if not success:
            self.failed_throws += 1
        self._reset_throw_buffers()
        if self.state != self.STATE_DISARMED:
            self.state = self.STATE_ARMED
            self.cooldown_until_ts = float(frame_ts + self.cfg.post_throw_cooldown_s)
            self.status_msg = f"{msg} Auto-armed for next throw."

    def stop_current_throw(self) -> None:
        self._reset_throw_buffers()
        self.state = self.STATE_ARMED
        self.cooldown_until_ts = 0.0
        self.status_msg = "Current throw stopped. Ready for next throw."

    # ------------------------------------------------------------------
    # File output
    # ------------------------------------------------------------------

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
                        continue
        except Exception:
            return 1
        return max_id + 1 if max_id > 0 else 1

    @staticmethod
    def _build_row(
        sample_id: int,
        first_points: List[Tuple[float, float, float, float]],
        intercept: Dict[str, float],
        prediction_ts: float,
    ) -> Dict[str, float]:
        t_vals = [p[3] for p in first_points]
        row: Dict[str, float] = {
            "sample_id": int(sample_id),
            "captured_at": datetime.now().isoformat(timespec="milliseconds"),
        }

        for i, (x, y, z, _t) in enumerate(first_points, start=1):
            row[f"x{i}"] = float(x)
            row[f"y{i}"] = float(y)
            row[f"z{i}"] = float(z)

        row["dt12"] = float(t_vals[1] - t_vals[0])
        row["dt23"] = float(t_vals[2] - t_vals[1])
        row["dt34"] = float(t_vals[3] - t_vals[2])

        row["x_hit"] = float(intercept["x"])
        row["y_hit"] = float(intercept["y"])
        row["z_hit"] = float(intercept["z"])
        row["vx_hit"] = float(intercept.get("vx", 0.0))
        row["vy_hit"] = float(intercept.get("vy", 0.0))
        row["vz_hit"] = float(intercept.get("vz", 0.0))
        # Predictor returns time-from-prediction-now. Convert to time-from-point4.
        t4 = float(t_vals[3])
        t_hit_abs = float(prediction_ts + float(intercept["time"]))
        row["t_hit"] = float(t_hit_abs - t4)
        row["is_reachable"] = 1.0
        row["intercept_valid"] = 1.0
        row["bounces_before_hit"] = float(intercept.get("bounces", 0))
        return row

    def save_sample(
        self,
        first_points: List[Tuple[float, float, float, float]],
        intercept: Dict[str, float],
        prediction_ts: float,
    ) -> Dict[str, float]:
        row = self._build_row(self.sample_id, first_points, intercept, prediction_ts)
        write_header = not self.dataset_csv.exists()
        with self.dataset_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        self.sample_id += 1
        self.saved_samples += 1
        return row

    # ------------------------------------------------------------------
    # Vision loop helpers
    # ------------------------------------------------------------------

    def warmup_background(self) -> bool:
        if self.cfg.warmup_s <= 0.0:
            return True

        _print("Remove moving objects/ball. Building background model (space=skip, q=quit)...")
        t0 = time.time()
        while time.time() - t0 < self.cfg.warmup_s:
            if not self.triangulator.cap_left.grab():
                continue
            if not self.triangulator.cap_right.grab():
                continue
            ret_l, frame_l = self.triangulator.cap_left.retrieve()
            ret_r, frame_r = self.triangulator.cap_right.retrieve()
            if not ret_l or not ret_r or frame_l is None or frame_r is None:
                continue

            self.triangulator.build_background(frame_l, frame_r)

            vis_l = frame_l.copy()
            vis_r = frame_r.copy()
            cv2.putText(vis_l, "LEFT", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(vis_r, "RIGHT", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            vis = self._compose_side_by_side(vis_l, vis_r)
            progress = min((time.time() - t0) / self.cfg.warmup_s, 1.0)
            h = vis.shape[0]
            bar_w = int(progress * (vis.shape[1] - 40))
            cv2.rectangle(vis, (20, h - 30), (20 + bar_w, h - 15), (0, 255, 255), -1)
            cv2.putText(vis, f"BG Warmup: {progress*100:.0f}%", (20, h - 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.imshow("Real Transfer Capture", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                break
            if key == ord("q"):
                return False

        _print("Background model ready.")
        return True

    def _compose_side_by_side(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Resize and stack left/right views while keeping total width manageable."""
        if left is None and right is None:
            raise ValueError("Both frames cannot be None.")
        if left is None:
            left = right.copy()
        if right is None:
            right = left.copy()

        target_total_w = self.cfg.preview_width
        panel_w = max(240, target_total_w // 2)

        h_l, w_l = left.shape[:2]
        h_r, w_r = right.shape[:2]
        panel_h_l = int(panel_w * h_l / max(1, w_l))
        panel_h_r = int(panel_w * h_r / max(1, w_r))

        left_rs = cv2.resize(left, (panel_w, panel_h_l))
        right_rs = cv2.resize(right, (panel_w, panel_h_r))

        panel_h = max(panel_h_l, panel_h_r)
        if panel_h_l < panel_h:
            pad = np.zeros((panel_h - panel_h_l, panel_w, 3), dtype=left_rs.dtype)
            left_rs = np.vstack([left_rs, pad])
        if panel_h_r < panel_h:
            pad = np.zeros((panel_h - panel_h_r, panel_w, 3), dtype=right_rs.dtype)
            right_rs = np.vstack([right_rs, pad])

        return np.hstack([left_rs, right_rs])

    def _process_capture_frame(self, result: dict, frame_ts: float) -> None:
        if self.state not in (self.STATE_ARMED, self.STATE_TRACKING):
            return
        if frame_ts < self.cooldown_until_ts:
            return

        if not result.get("found_3d", False):
            if self.state == self.STATE_TRACKING:
                self.lost_frames += 1
                if self.lost_frames >= self.cfg.max_lost_frames:
                    self._complete_throw(False, frame_ts, "Throw failed (lost tracking).")
            return

        reproj = float(result.get("reproj_err") or 0.0)
        if reproj > self.cfg.reproj_err_max_px:
            return

        cam_pos = result["position_3d"]
        rx, ry, rz = self.cam_to_robot(self.R, self.t_vec, self.cam_scale, cam_pos[0], cam_pos[1], cam_pos[2])
        self.last_robot_point = (rx, ry, rz)
        self.lost_frames = 0

        accepted = bool(self.predictor.add_position(rx, ry, rz, frame_ts))
        if not accepted:
            return

        if self.state == self.STATE_ARMED:
            self.state = self.STATE_TRACKING
            self.tracking_start_ts = frame_ts

        if len(self.first_points) < 4:
            self.first_points.append((rx, ry, rz, frame_ts))

        if self.state == self.STATE_TRACKING and self.tracking_start_ts is not None:
            if (frame_ts - self.tracking_start_ts) > self.cfg.max_throw_s:
                self._complete_throw(False, frame_ts, "Throw failed (timeout before valid intercept).")
                return

        if len(self.first_points) < 4:
            return

        if not self.predictor.is_ready():
            return

        intercept = self.predictor.predict_intercept()
        self.last_intercept = intercept
        if intercept is None:
            return

        # Keep only true in-workspace intercept labels; no clamp fallback here.
        if bool(intercept.get("clamped", False)):
            return
        if not self.in_workspace(float(intercept["x"]), float(intercept["y"]), float(intercept["z"])):
            return

        saved_row = self.save_sample(self.first_points, intercept, frame_ts)
        self._complete_throw(True, frame_ts, f"Saved sample #{int(float(saved_row['sample_id']))}.")
        _print(
            f"Saved sample #{self.saved_samples} -> {self.dataset_csv} | "
            f"x_hit={intercept['x']:+.1f}, y_hit={intercept['y']:+.1f}, "
            f"z_hit={intercept['z']:+.1f}, t_hit(p4)={saved_row['t_hit']*1000:.1f}ms"
        )

    def _draw_overlay(self, result: dict, fps: float) -> None:
        show_debug = self.cfg.show_stereo_debug and self.state in (self.STATE_ARMED, self.STATE_TRACKING)
        if show_debug:
            left_vis, right_vis = self.triangulator.draw_results(result)
        else:
            left_vis = result["left_frame"].copy()
            right_vis = result["right_frame"].copy()

        h, w = left_vis.shape[:2]

        state_color = {
            self.STATE_DISARMED: (120, 120, 255),
            self.STATE_ARMED: (0, 220, 255),
            self.STATE_TRACKING: (0, 255, 0),
        }.get(self.state, (255, 255, 255))

        cv2.putText(
            left_vis,
            f"State: {self.state}  FPS:{fps:.0f}  Saved:{self.saved_samples}  Failed:{self.failed_throws}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            state_color,
            1,
        )
        cv2.putText(
            left_vis,
            self.status_msg,
            (10, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (230, 230, 230),
            1,
        )

        cv2.putText(
            left_vis,
            f"First accepted points: {len(self.first_points)}/4  Predictor buffer: {len(self.predictor.positions)}",
            (10, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (180, 255, 180),
            1,
        )

        if self.last_robot_point is not None:
            rx, ry, rz = self.last_robot_point
            cv2.putText(
                left_vis,
                f"Ball(mm): X={rx:+.1f} Y={ry:+.1f} Z={rz:+.1f}",
                (10, 91),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 0),
                1,
            )

        if self.last_intercept is not None:
            ip = self.last_intercept
            cv2.putText(
                left_vis,
                f"Intercept: X={ip['x']:+.1f} Y={ip['y']:+.1f} Z={ip['z']:+.1f} t={ip['time']*1000:.0f}ms clamped={ip.get('clamped', False)}",
                (10, 114),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 230, 230),
                1,
            )

        cv2.putText(
            left_vis,
            f"Workspace: X[{self.ELLIPSE_A:.0f}] Y[{self.ELLIPSE_B:.0f}] Z[{self.Z_MIN:.0f},{self.Z_MAX:.0f}]",
            (10, 137),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (210, 210, 210),
            1,
        )
        cv2.putText(
            left_vis,
            "Keys: s=start  r=stop throw  x=pause  b=bg relearn  q=quit",
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (180, 180, 180),
            1,
        )

        cv2.putText(
            left_vis,
            "LEFT",
            (w - 70, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (120, 255, 120),
            1,
        )
        cv2.putText(
            right_vis,
            "RIGHT",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (120, 255, 120),
            1,
        )

        stacked = self._compose_side_by_side(left_vis, right_vis)
        cv2.imshow("Real Transfer Capture", stacked)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> int:
        _print("Starting real transfer-data collector")
        _print(f"Output dir: {self.output_dir.resolve()}")
        _print(f"Dataset file: {self.dataset_csv.resolve()}")
        _print(f"Cameras: L={self.cam_left_id}, R={self.cam_right_id}")
        _print("Workflow: s=start -> throw -> autosave row -> auto-ready next throw")

        try:
            self.triangulator.start_cameras(self.frame_width, self.frame_height)
        except Exception as exc:
            _print(f"Failed starting cameras: {exc}")
            return 1

        if not self.warmup_background():
            self.triangulator.stop_cameras()
            cv2.destroyAllWindows()
            return 1

        fps = 0.0
        fps_counter = 0
        fps_t0 = time.perf_counter()

        try:
            while True:
                result = self.triangulator.update()
                if result.get("left_frame") is None:
                    continue
                frame_ts = float(result.get("capture_time", time.perf_counter()))

                fps_counter += 1
                if fps_counter % 30 == 0:
                    now = time.perf_counter()
                    fps = 30.0 / max(1e-6, now - fps_t0)
                    fps_t0 = now

                self._process_capture_frame(result, frame_ts)
                self._draw_overlay(result, fps)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    _print("Quit requested.")
                    break
                if key == ord("s"):
                    if self.state == self.STATE_DISARMED:
                        self.arm("Capture running. Throw balls; rows auto-save.")
                        _print("Capture started (armed).")
                elif key == ord("r"):
                    if self.state in (self.STATE_ARMED, self.STATE_TRACKING):
                        self.stop_current_throw()
                        _print("Stopped current throw; ready for next.")
                elif key == ord("x"):
                    self.disarm()
                    _print("Paused (disarmed).")
                elif key == ord("b"):
                    self.disarm("Background reset complete. Press 's' to arm.")
                    self.triangulator.reset_background()
                    _print("Background reset requested.")
                    if not self.warmup_background():
                        break

        except KeyboardInterrupt:
            _print("Interrupted by user.")
        finally:
            try:
                self.triangulator.stop_cameras()
            except Exception:
                pass
            cv2.destroyAllWindows()

        _print(f"Done. Saved={self.saved_samples}, failed={self.failed_throws}")
        _print(f"Dataset: {self.dataset_csv.resolve()}")
        return 0


def parse_args() -> CaptureConfig:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Collect real transfer-learning samples from live stereo throws.")
    parser.add_argument(
        "--stack-root",
        type=str,
        default=str(repo_root / "new_top_level"),
        help="Path to new_top_level folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "real_transfer_data"),
        help="Directory for dataset CSV export.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="real_transfer_dataset.csv",
        help="Single CSV file name (Excel-friendly) to append all throws.",
    )
    parser.add_argument("--warmup-s", type=float, default=2.0, help="Background warmup seconds.")
    parser.add_argument("--reproj-max", type=float, default=10.0, help="Max reprojection error (px) to accept 3D point.")
    parser.add_argument("--max-throw-s", type=float, default=2.5, help="Max seconds after first accepted point before timeout.")
    parser.add_argument("--max-lost-frames", type=int, default=25, help="Consecutive no-3D frames before throw is marked failed.")
    parser.add_argument("--cooldown-s", type=float, default=0.35, help="Cooldown after each throw before next capture [s].")
    parser.add_argument("--preview-width", type=int, default=960, help="Display width of visualization window.")
    parser.add_argument(
        "--raw-view",
        action="store_true",
        help="Show raw stereo frames (disable detection/debug overlays).",
    )
    args = parser.parse_args()

    return CaptureConfig(
        stack_root=Path(args.stack_root),
        output_dir=Path(args.output_dir),
        output_file_name=str(args.output_file),
        warmup_s=max(0.0, float(args.warmup_s)),
        reproj_err_max_px=max(0.0, float(args.reproj_max)),
        max_throw_s=max(0.5, float(args.max_throw_s)),
        max_lost_frames=max(1, int(args.max_lost_frames)),
        post_throw_cooldown_s=max(0.0, float(args.cooldown_s)),
        preview_width=max(320, int(args.preview_width)),
        show_stereo_debug=not bool(args.raw_view),
    )


def main() -> int:
    cfg = parse_args()
    collector = RealTransferCollector(cfg)
    return collector.run()


if __name__ == "__main__":
    raise SystemExit(main())
