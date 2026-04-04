#!/usr/bin/env python3
"""
Collect real transfer-learning samples from live stereo triangulation.

Workflow:
1. Run script, wait for warmup.
2. Press `s` to arm.
3. Throw one ball.
4. Script records first N accepted robot-frame points (`--num-points`, default 6).
5. When throw ends, it labels using the observed point closest to workspace.
6. Appends one row to a single dataset CSV.
7. Automatically re-arms for the next throw (no reset required).

No UART is used here; this is a pure data-capture tool.

t_hit definition in saved CSV:
  t_hit = time from last input point (pointN) to selected observed hit point [s]
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
    num_input_points: int
    preview_width: int
    show_stereo_debug: bool
    det_min_area: Optional[float]
    det_max_area: Optional[float]
    det_min_circularity: Optional[float]
    det_search_radius: Optional[float]
    tri_min_disparity: Optional[float]
    tri_max_epipolar: Optional[float]
    tri_max_reproj: Optional[float]
    tri_max_z: Optional[float]
    table_length_mm: float
    table_width_mm: float


class RealTransferCollector:
    STATE_DISARMED = "DISARMED"
    STATE_ARMED = "ARMED"
    STATE_TRACKING = "TRACKING"

    def __init__(self, cfg: CaptureConfig) -> None:
        self.cfg = cfg
        self.output_dir = cfg.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_csv = self.output_dir / cfg.output_file_name
        self.plot_dir = self.output_dir / "sample_plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.sample_id = self._next_sample_id()
        self.cooldown_until_ts = 0.0
        self.last_saved_row: Optional[Dict[str, float]] = None

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
            from estimation.workspace import (
                in_workspace,
                ELLIPSE_A,
                ELLIPSE_B,
                Z_MIN,
                Z_MAX,
                Z_TABLE_SURFACE,
                ROBOT_HOME,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed importing new_top_level modules from '{stack_root}': {exc}"
            ) from exc

        self.StereoTriangulator = StereoTriangulator
        self.load_points_based_transform = load_points_based_transform
        self.cam_to_robot = cam_to_robot
        self.load_camera_settings = load_camera_settings
        self.in_workspace = in_workspace
        self.ELLIPSE_A = ELLIPSE_A
        self.ELLIPSE_B = ELLIPSE_B
        self.Z_MIN = Z_MIN
        self.Z_MAX = Z_MAX
        self.Z_TABLE_SURFACE = float(Z_TABLE_SURFACE)
        self.ROBOT_HOME = tuple(float(v) for v in ROBOT_HOME)

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
        self._apply_tracking_overrides()

    def _apply_tracking_overrides(self) -> None:
        # Detector tuning (helps when far-away ball contour gets very small).
        if self.cfg.det_min_area is not None:
            self.triangulator.detector_left.min_area = float(self.cfg.det_min_area)
            self.triangulator.detector_right.min_area = float(self.cfg.det_min_area)
        if self.cfg.det_max_area is not None:
            self.triangulator.detector_left.max_area = float(self.cfg.det_max_area)
            self.triangulator.detector_right.max_area = float(self.cfg.det_max_area)
        if self.cfg.det_min_circularity is not None:
            self.triangulator.detector_left.min_circularity = float(self.cfg.det_min_circularity)
            self.triangulator.detector_right.min_circularity = float(self.cfg.det_min_circularity)
        if self.cfg.det_search_radius is not None:
            self.triangulator.detector_left.search_radius = float(self.cfg.det_search_radius)
            self.triangulator.detector_right.search_radius = float(self.cfg.det_search_radius)

        # Triangulation sanity thresholds (far points often fail disparity/reproj).
        if self.cfg.tri_min_disparity is not None:
            self.triangulator.MIN_DISPARITY = float(self.cfg.tri_min_disparity)
        if self.cfg.tri_max_epipolar is not None:
            self.triangulator.MAX_EPIPOLAR_ERR = float(self.cfg.tri_max_epipolar)
        if self.cfg.tri_max_reproj is not None:
            self.triangulator.MAX_REPROJ_ERR = float(self.cfg.tri_max_reproj)
        if self.cfg.tri_max_z is not None:
            self.triangulator.MAX_Z = float(self.cfg.tri_max_z)

    # ------------------------------------------------------------------
    # Throw state
    # ------------------------------------------------------------------

    def _reset_throw_buffers(self) -> None:
        self.first_points: List[Tuple[float, float, float, float]] = []
        self.accepted_points: List[Tuple[float, float, float, float]] = []
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
        observed_hit: Dict[str, float],
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

        for i in range(1, len(t_vals)):
            row[f"dt{i}{i+1}"] = float(t_vals[i] - t_vals[i - 1])

        row["x_hit"] = float(observed_hit["x"])
        row["y_hit"] = float(observed_hit["y"])
        row["z_hit"] = float(observed_hit["z"])
        row["vx_hit"] = float(observed_hit.get("vx", 0.0))
        row["vy_hit"] = float(observed_hit.get("vy", 0.0))
        row["vz_hit"] = float(observed_hit.get("vz", 0.0))
        t_ref = float(t_vals[-1])
        t_hit_abs = float(observed_hit["t_abs"])
        row["t_hit"] = float(t_hit_abs - t_ref)
        row["is_reachable"] = float(observed_hit.get("is_reachable", 0.0))
        row["intercept_valid"] = 1.0
        row["bounces_before_hit"] = float(observed_hit.get("bounces_before_hit", 0.0))
        return row

    def save_sample(
        self,
        first_points: List[Tuple[float, float, float, float]],
        observed_hit: Dict[str, float],
    ) -> Dict[str, float]:
        row = self._build_row(self.sample_id, first_points, observed_hit)
        write_header = not self.dataset_csv.exists()
        with self.dataset_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        self.last_saved_row = dict(row)
        self.sample_id += 1
        self.saved_samples += 1
        return row

    def delete_last_sample(self) -> bool:
        """Delete last row from dataset CSV and rewind sample_id."""
        if not self.dataset_csv.exists():
            self.status_msg = "No dataset file yet; nothing to delete."
            return False

        try:
            with self.dataset_csv.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames or [])
                rows = list(reader)
        except Exception as exc:
            self.status_msg = f"Delete failed (read error): {exc}"
            return False

        if not fieldnames or not rows:
            self.status_msg = "Dataset has no rows; nothing to delete."
            return False

        removed = rows.pop()
        removed_id = removed.get("sample_id", "?")

        try:
            with self.dataset_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        except Exception as exc:
            self.status_msg = f"Delete failed (write error): {exc}"
            return False

        self.sample_id = self._next_sample_id()
        self.saved_samples = max(0, self.saved_samples - 1)
        self.last_saved_row = None
        self.status_msg = f"Deleted last sample (sample_id={removed_id})."
        return True

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def _load_last_saved_row(self) -> Optional[Dict[str, float]]:
        if self.last_saved_row is not None:
            return dict(self.last_saved_row)
        if not self.dataset_csv.exists():
            return None
        try:
            with self.dataset_csv.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if not rows:
                return None
            return rows[-1]
        except Exception:
            return None

    @staticmethod
    def _safe_float(value: object) -> float:
        try:
            return float(value)
        except Exception:
            return float("nan")

    def plot_last_sample(self) -> Optional[Path]:
        row = self._load_last_saved_row()
        if row is None:
            self.status_msg = "No saved sample to plot yet."
            return None

        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            self.status_msg = f"Plot unavailable (matplotlib import failed: {exc})"
            return None

        n = int(self.cfg.num_input_points)
        pts = []
        for i in range(1, n + 1):
            x = self._safe_float(row.get(f"x{i}"))
            y = self._safe_float(row.get(f"y{i}"))
            z = self._safe_float(row.get(f"z{i}"))
            if np.isnan(x) or np.isnan(y) or np.isnan(z):
                self.status_msg = f"Plot failed: missing x{i}/y{i}/z{i}."
                return None
            pts.append((x, y, z))
        pts_np = np.asarray(pts, dtype=float)

        # Reconstruct observation times from dt12..dt(N-1)N (relative to p1).
        t_obs = np.zeros(n, dtype=float)
        dt_ok = True
        for i in range(2, n + 1):
            dt_key = f"dt{i-1}{i}"
            dt_val = self._safe_float(row.get(dt_key))
            if np.isnan(dt_val) or dt_val <= 0.0:
                dt_ok = False
                break
            t_obs[i - 1] = t_obs[i - 2] + dt_val
        if not dt_ok or float(np.ptp(t_obs)) < 1e-6:
            # Fallback: unit-spaced pseudo-time if dt columns are unavailable.
            t_obs = np.arange(n, dtype=float)

        xh = self._safe_float(row.get("x_hit"))
        yh = self._safe_float(row.get("y_hit"))
        zh = self._safe_float(row.get("z_hit"))
        if np.isnan(xh) or np.isnan(yh) or np.isnan(zh):
            self.status_msg = "Plot failed: missing x_hit/y_hit/z_hit."
            return None

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        # Input points p1..pN and line.
        ax.plot(pts_np[:, 0], pts_np[:, 1], pts_np[:, 2], "-o", color="tab:blue", linewidth=2.0, label=f"Observed p1..p{n}")
        for i, (x, y, z) in enumerate(pts_np, start=1):
            ax.text(x, y, z, f"p{i}", fontsize=8)

        # Hit point.
        ax.scatter([xh], [yh], [zh], color="tab:red", marker="*", s=150, label="x_hit")
        ax.plot([pts_np[-1, 0], xh], [pts_np[-1, 1], yh], [pts_np[-1, 2], zh], "--", color="tab:red", alpha=0.6)

        # Table outline at table surface (x centered, y from 0 to table length).
        half_w = 0.5 * float(self.cfg.table_width_mm)
        y0 = 0.0
        y1 = float(self.cfg.table_length_mm)
        zt = float(self.Z_TABLE_SURFACE)
        tx = np.array([-half_w, half_w, half_w, -half_w, -half_w], dtype=float)
        ty = np.array([y0, y0, y1, y1, y0], dtype=float)
        tz = np.full_like(tx, zt)
        ax.plot(tx, ty, tz, color="saddlebrown", linewidth=1.8, label="Table top")

        # Workspace ellipse at z_min and z_max + a few vertical ribs.
        th = np.linspace(0.0, 2.0 * np.pi, 160)
        ex = float(self.ELLIPSE_A) * np.cos(th)
        ey = float(self.ELLIPSE_B) * np.sin(th)
        ax.plot(ex, ey, np.full_like(ex, float(self.Z_MIN)), color="tab:green", linewidth=1.5, label="Workspace")
        ax.plot(ex, ey, np.full_like(ex, float(self.Z_MAX)), color="tab:green", linewidth=1.5)
        for idx in [0, 40, 80, 120]:
            ax.plot(
                [ex[idx], ex[idx]],
                [ey[idx], ey[idx]],
                [float(self.Z_MIN), float(self.Z_MAX)],
                color="tab:green",
                alpha=0.5,
                linewidth=1.0,
            )

        # Axis limits and aspect from all relevant geometry.
        all_x = np.concatenate([pts_np[:, 0], np.array([xh]), tx, np.array([float(self.ELLIPSE_A), -float(self.ELLIPSE_A)])])
        all_y = np.concatenate([pts_np[:, 1], np.array([yh]), ty, np.array([float(self.ELLIPSE_B), -float(self.ELLIPSE_B)])])
        all_z = np.concatenate([pts_np[:, 2], np.array([zh, float(self.Z_MIN), float(self.Z_MAX), zt])])

        margin = 40.0
        ax.set_xlim(float(np.min(all_x) - margin), float(np.max(all_x) + margin))
        ax.set_ylim(float(np.min(all_y) - margin), float(np.max(all_y) + margin))
        ax.set_zlim(float(np.min(all_z) - margin), float(np.max(all_z) + margin))

        x_range = max(1.0, float(np.ptp(ax.get_xlim3d())))
        y_range = max(1.0, float(np.ptp(ax.get_ylim3d())))
        z_range = max(1.0, float(np.ptp(ax.get_zlim3d())))
        ax.set_box_aspect((x_range, y_range, z_range))

        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_zlabel("Z [mm]")
        sid_f = self._safe_float(row.get("sample_id"))
        sid = int(sid_f) if not np.isnan(sid_f) else -1
        reachable = int(self._safe_float(row.get("is_reachable")) > 0.5)
        t_hit_ms = 1000.0 * self._safe_float(row.get("t_hit"))
        ax.set_title(f"Sample {sid} | reachable={reachable} | t_hit={t_hit_ms:.1f} ms")
        ax.legend(loc="best")
        fig.tight_layout()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path_3d = self.plot_dir / f"sample_{sid}_{ts}_3d.png"
        fig.savefig(out_path_3d, dpi=170)

        # 2D projection figure for easier visual checking.
        fig2, axs = plt.subplots(1, 3, figsize=(15, 4.8))
        ax_xy, ax_yz, ax_xz = axs  # display order: XY, YZ, XZ

        # XY
        ax_xy.plot(pts_np[:, 0], pts_np[:, 1], "-o", color="tab:blue", linewidth=1.8, label="Observed")
        ax_xy.scatter([xh], [yh], color="tab:red", marker="*", s=120, label="x_hit")
        ax_xy.plot(tx, ty, color="saddlebrown", linewidth=1.4, label="Table")
        ax_xy.plot(ex, ey, color="tab:green", linewidth=1.4, label="Workspace")
        ax_xy.set_title("XY Plane")
        ax_xy.set_xlabel("X [mm]")
        ax_xy.set_ylabel("Y [mm]")
        ax_xy.grid(alpha=0.25)
        ax_xy.set_aspect("equal", adjustable="box")

        # XZ
        ax_xz.plot(pts_np[:, 0], pts_np[:, 2], "-o", color="tab:blue", linewidth=1.8, label="Observed")
        ax_xz.scatter([xh], [zh], color="tab:red", marker="*", s=120, label="x_hit")
        ax_xz.plot([-half_w, half_w], [zt, zt], color="saddlebrown", linewidth=1.4, label="Table z")
        xa = float(self.ELLIPSE_A)
        ax_xz.plot([-xa, xa], [float(self.Z_MIN), float(self.Z_MIN)], color="tab:green", linewidth=1.2, label="Workspace proj")
        ax_xz.plot([-xa, xa], [float(self.Z_MAX), float(self.Z_MAX)], color="tab:green", linewidth=1.2)
        ax_xz.plot([-xa, -xa], [float(self.Z_MIN), float(self.Z_MAX)], color="tab:green", linewidth=1.2)
        ax_xz.plot([xa, xa], [float(self.Z_MIN), float(self.Z_MAX)], color="tab:green", linewidth=1.2)
        ax_xz.set_title("XZ Plane")
        ax_xz.set_xlabel("X [mm]")
        ax_xz.set_ylabel("Z [mm]")
        ax_xz.grid(alpha=0.25)

        # YZ
        ax_yz.plot(pts_np[:, 1], pts_np[:, 2], "-o", color="tab:blue", linewidth=1.8, label="Observed")
        ax_yz.scatter([yh], [zh], color="tab:red", marker="*", s=120, label="x_hit")
        ax_yz.plot([y0, y1], [zt, zt], color="saddlebrown", linewidth=1.4, label="Table z")
        yb = float(self.ELLIPSE_B)
        ax_yz.plot([-yb, yb], [float(self.Z_MIN), float(self.Z_MIN)], color="tab:green", linewidth=1.2, label="Workspace proj")
        ax_yz.plot([-yb, yb], [float(self.Z_MAX), float(self.Z_MAX)], color="tab:green", linewidth=1.2)
        ax_yz.plot([-yb, -yb], [float(self.Z_MIN), float(self.Z_MAX)], color="tab:green", linewidth=1.2)
        ax_yz.plot([yb, yb], [float(self.Z_MIN), float(self.Z_MAX)], color="tab:green", linewidth=1.2)
        ax_yz.set_title("YZ Plane")
        ax_yz.set_xlabel("Y [mm]")
        ax_yz.set_ylabel("Z [mm]")
        ax_yz.grid(alpha=0.25)

        handles, labels = ax_xy.get_legend_handles_labels()
        if handles:
            fig2.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)))
        fig2.suptitle(f"Sample {sid} Projections", y=1.02)
        fig2.tight_layout()

        out_path_planes = self.plot_dir / f"sample_{sid}_{ts}_planes.png"
        fig2.savefig(out_path_planes, dpi=170)

        backend = plt.get_backend().lower()
        if "agg" not in backend:
            plt.show(block=False)
            self.status_msg = (
                f"Plotted sample #{sid}; saved: {out_path_3d.name}, {out_path_planes.name}"
            )
        else:
            plt.close(fig)
            plt.close(fig2)
            self.status_msg = (
                f"Saved sample plots (non-interactive backend): {out_path_3d.name}, {out_path_planes.name}"
            )
        return out_path_planes

    # ------------------------------------------------------------------
    # Label selection from observed points
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_velocity(points: List[Tuple[float, float, float, float]], idx: int) -> Tuple[float, float, float]:
        """Estimate velocity at selected index using local linear fit over time."""
        if len(points) < 2:
            return 0.0, 0.0, 0.0

        left = max(0, idx - 2)
        right = min(len(points), idx + 3)
        window = points[left:right]
        if len(window) < 2:
            return 0.0, 0.0, 0.0

        t0 = float(window[0][3])
        ts = np.array([float(p[3]) - t0 for p in window], dtype=float)
        if float(np.max(ts) - np.min(ts)) < 1e-6:
            return 0.0, 0.0, 0.0

        def slope(vals: np.ndarray) -> float:
            A = np.column_stack([ts, np.ones_like(ts)])
            m, _b = np.linalg.lstsq(A, vals, rcond=None)[0]
            return float(m)

        xs = np.array([float(p[0]) for p in window], dtype=float)
        ys = np.array([float(p[1]) for p in window], dtype=float)
        zs = np.array([float(p[2]) for p in window], dtype=float)
        return slope(xs), slope(ys), slope(zs)

    def _count_observed_bounces(self, points: List[Tuple[float, float, float, float]], end_idx: int) -> int:
        """Approximate bounces before selected point from observed Z crossing table."""
        if end_idx < 2:
            return 0

        z_table = self.Z_TABLE_SURFACE
        count = 0
        for i in range(1, end_idx):
            z_prev = float(points[i - 1][2])
            z_now = float(points[i][2])
            if z_prev >= z_table > z_now:
                z_next = float(points[i + 1][2]) if (i + 1) <= end_idx else z_now
                if z_next >= z_now:
                    count += 1
        return count

    def _select_observed_hit(self) -> Optional[Dict[str, float]]:
        if len(self.first_points) < self.cfg.num_input_points or not self.accepted_points:
            return None

        t_ref = float(self.first_points[-1][3])
        cx, cy, cz = self.ROBOT_HOME

        best_ws_idx = None
        best_ws_d2 = float("inf")
        best_any_idx = None
        best_any_d2 = float("inf")

        for i, (x, y, z, t) in enumerate(self.accepted_points):
            if float(t) < t_ref - 1e-6:
                continue
            d2 = (float(x) - cx) ** 2 + (float(y) - cy) ** 2 + (float(z) - cz) ** 2
            if d2 < best_any_d2:
                best_any_d2 = float(d2)
                best_any_idx = i

            if self.in_workspace(float(x), float(y), float(z)) and d2 < best_ws_d2:
                best_ws_d2 = float(d2)
                best_ws_idx = i

        chosen_idx = best_ws_idx if best_ws_idx is not None else best_any_idx
        if chosen_idx is None:
            return None

        xh, yh, zh, th = self.accepted_points[chosen_idx]
        vx, vy, vz = self._estimate_velocity(self.accepted_points, chosen_idx)
        bounces = self._count_observed_bounces(self.accepted_points, chosen_idx)
        is_reachable = 1.0 if self.in_workspace(float(xh), float(yh), float(zh)) else 0.0

        return {
            "x": float(xh),
            "y": float(yh),
            "z": float(zh),
            "vx": float(vx),
            "vy": float(vy),
            "vz": float(vz),
            "t_abs": float(th),
            "is_reachable": float(is_reachable),
            "bounces_before_hit": float(bounces),
        }

    def _finalize_throw_from_observations(self, frame_ts: float, reason: str) -> None:
        if len(self.first_points) < self.cfg.num_input_points:
            self._complete_throw(
                False,
                frame_ts,
                f"{reason} (failed: <{self.cfg.num_input_points} accepted points).",
            )
            return

        observed_hit = self._select_observed_hit()
        if observed_hit is None:
            self._complete_throw(False, frame_ts, f"{reason} (failed: no observed hit candidate).")
            return

        saved_row = self.save_sample(self.first_points, observed_hit)
        self.last_intercept = {
            "x": observed_hit["x"],
            "y": observed_hit["y"],
            "z": observed_hit["z"],
            "vx": observed_hit["vx"],
            "vy": observed_hit["vy"],
            "vz": observed_hit["vz"],
            "time": float(saved_row["t_hit"]),
            "clamped": bool(float(observed_hit["is_reachable"]) < 0.5),
            "bounces": observed_hit["bounces_before_hit"],
        }

        self._complete_throw(True, frame_ts, f"Saved sample #{int(float(saved_row['sample_id']))}.")
        _print(
            f"Saved sample #{self.saved_samples} -> {self.dataset_csv} | "
            f"x_hit={observed_hit['x']:+.1f}, y_hit={observed_hit['y']:+.1f}, "
            f"z_hit={observed_hit['z']:+.1f}, t_hit(p{self.cfg.num_input_points})={saved_row['t_hit']*1000:.1f}ms, "
            f"reachable={int(float(saved_row['is_reachable']))}"
        )

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
                    self._finalize_throw_from_observations(frame_ts, "Throw ended (lost tracking)")
            return

        reproj = float(result.get("reproj_err") or 0.0)
        if reproj > self.cfg.reproj_err_max_px:
            return

        cam_pos = result["position_3d"]
        rx, ry, rz = self.cam_to_robot(self.R, self.t_vec, self.cam_scale, cam_pos[0], cam_pos[1], cam_pos[2])
        self.last_robot_point = (rx, ry, rz)
        self.lost_frames = 0

        # Accept raw stereo point once it passes found_3d + reprojection gate.
        self.accepted_points.append((rx, ry, rz, frame_ts))

        if self.state == self.STATE_ARMED:
            self.state = self.STATE_TRACKING
            self.tracking_start_ts = frame_ts

        if len(self.first_points) < self.cfg.num_input_points:
            self.first_points.append((rx, ry, rz, frame_ts))

        if self.state == self.STATE_TRACKING and self.tracking_start_ts is not None:
            if (frame_ts - self.tracking_start_ts) > self.cfg.max_throw_s:
                self._finalize_throw_from_observations(frame_ts, "Throw ended (timeout)")
                return

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
            f"First accepted points: {len(self.first_points)}/{self.cfg.num_input_points}  Accepted throw points: {len(self.accepted_points)}",
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
                f"Observed hit: X={ip['x']:+.1f} Y={ip['y']:+.1f} Z={ip['z']:+.1f} t={ip['time']*1000:.0f}ms reachable={not ip.get('clamped', False)}",
                (10, 114),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 230, 230),
                1,
            )

        # Extra diagnostics for why far-away tracking may drop.
        tri_rej = result.get("reject_reason")
        if tri_rej:
            cv2.putText(
                left_vis,
                f"Tri reject: {tri_rej}",
                (10, 158),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 170, 255),
                1,
            )
        else:
            lrej = result.get("left_rejected") or []
            rrej = result.get("right_rejected") or []
            if lrej or rrej:
                def top_reason(items: List[Dict[str, object]]) -> str:
                    if not items:
                        return "--"
                    counts: Dict[str, int] = {}
                    for it in items:
                        rs = str(it.get("reason", "?"))
                        counts[rs] = counts.get(rs, 0) + 1
                    return max(counts.items(), key=lambda kv: kv[1])[0]

                cv2.putText(
                    left_vis,
                    f"Det reject L:{top_reason(lrej)}({len(lrej)}) R:{top_reason(rrej)}({len(rrej)})",
                    (10, 158),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 180, 0),
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
            "Keys: s=start  r=stop throw  x/p=pause  d=delete last  v=plot last  b=bg relearn  q=quit",
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
        _print(
            "Tracking thresholds: "
            f"det_min_area={self.triangulator.detector_left.min_area}, "
            f"det_min_circularity={self.triangulator.detector_left.min_circularity}, "
            f"tri_min_disp={self.triangulator.MIN_DISPARITY}, "
            f"tri_max_reproj={self.triangulator.MAX_REPROJ_ERR}, "
            f"tri_max_z={self.triangulator.MAX_Z}"
        )

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
                elif key in (ord("x"), ord("p")):
                    self.disarm()
                    _print("Paused (disarmed).")
                elif key == ord("d"):
                    if self.delete_last_sample():
                        _print(self.status_msg)
                    else:
                        _print(self.status_msg)
                elif key == ord("v"):
                    path = self.plot_last_sample()
                    if path is None:
                        _print(self.status_msg)
                    else:
                        _print(f"{self.status_msg} -> {path}")
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
        default=str(Path(__file__).resolve().parent / "real_data_collected"),
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
    parser.add_argument(
        "--num-points",
        type=int,
        default=6,
        help="Number of initial accepted points to save as model inputs.",
    )
    parser.add_argument("--preview-width", type=int, default=960, help="Display width of visualization window.")
    parser.add_argument(
        "--raw-view",
        action="store_true",
        help="Show raw stereo frames (disable detection/debug overlays).",
    )
    parser.add_argument("--det-min-area", type=float, default=None, help="Override detector min contour area (px).")
    parser.add_argument("--det-max-area", type=float, default=None, help="Override detector max contour area (px).")
    parser.add_argument("--det-min-circularity", type=float, default=None, help="Override detector min circularity.")
    parser.add_argument("--det-search-radius", type=float, default=None, help="Override detector search radius (px).")
    parser.add_argument("--tri-min-disparity", type=float, default=None, help="Override triangulation minimum disparity (px).")
    parser.add_argument("--tri-max-epipolar", type=float, default=None, help="Override triangulation max epipolar error (px).")
    parser.add_argument("--tri-max-reproj", type=float, default=None, help="Override triangulation max reprojection error (px).")
    parser.add_argument("--tri-max-z", type=float, default=None, help="Override triangulation max depth (calibration units).")
    parser.add_argument("--table-length-mm", type=float, default=2740.0, help="Table length along +Y for plotting [mm].")
    parser.add_argument("--table-width-mm", type=float, default=1525.0, help="Table width across X for plotting [mm].")
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
        num_input_points=max(4, int(args.num_points)),
        preview_width=max(320, int(args.preview_width)),
        show_stereo_debug=not bool(args.raw_view),
        det_min_area=args.det_min_area,
        det_max_area=args.det_max_area,
        det_min_circularity=args.det_min_circularity,
        det_search_radius=args.det_search_radius,
        tri_min_disparity=args.tri_min_disparity,
        tri_max_epipolar=args.tri_max_epipolar,
        tri_max_reproj=args.tri_max_reproj,
        tri_max_z=args.tri_max_z,
        table_length_mm=max(100.0, float(args.table_length_mm)),
        table_width_mm=max(100.0, float(args.table_width_mm)),
    )


def main() -> int:
    cfg = parse_args()
    collector = RealTransferCollector(cfg)
    return collector.run()


if __name__ == "__main__":
    raise SystemExit(main())
