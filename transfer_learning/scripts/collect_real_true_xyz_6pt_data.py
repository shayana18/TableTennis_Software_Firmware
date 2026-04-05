#!/usr/bin/env python3
"""
Collect real transfer-learning samples from live stereo triangulation
using true observed hit labels (no robot predictor labels).

Behavior:
- Capture first 6 accepted robot-frame points as inputs.
- Select closest in-workspace point to workspace center (fallback: closest overall).
- Save synthetic-aligned targets: x_hit/y_hit/z_hit, vx_hit/vy_hit/vz_hit, t_hit.
- Keep keyboard controls from collect_real_transfer_data.py:
  s=start, x/p=pause, d=delete-last, r=reset throw, q=quit.

Notes:
- `t_hit` is stored as time from point6 to selected observed hit point [s].
- While filling first 6 inputs, if a time gap is larger than
  `--max-first6-gap-s`, first-point counting restarts from that point.
- Each saved sample also writes a full accepted-point trajectory CSV to
  `output_dir/sample_trajectories/`.
- Adds extra metadata columns for auditing chosen hit:
  hit_obs_index, hit_obs_in_workspace, hit_obs_selected_from_workspace_pool,
  hit_center_dist_mm, has_bounce_before_hit.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

from collect_real_transfer_data import (
    CaptureConfig,
    RealTransferCollector,
    _print,
    parse_args as base_parse_args,
)


class TrueXYZSixPointCollector(RealTransferCollector):
    """Real collector that locks inputs to 6 points and stores hit-selection metadata."""

    def __init__(self, cfg: CaptureConfig) -> None:
        super().__init__(cfg)
        self.trajectory_dir = self.output_dir / "sample_trajectories"
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        self.last_trajectory_csv: Optional[Path] = None
        self.max_first6_gap_s = max(0.0, float(getattr(cfg, "max_first6_gap_s", 0.03)))

    @staticmethod
    def _timestamp_for_filename(captured_at: str) -> str:
        # Keep filename ASCII-safe while preserving timestamp order.
        return (
            str(captured_at)
            .replace("-", "")
            .replace(":", "")
            .replace(".", "")
            .replace("T", "_")
        )

    def _save_trajectory_csv(
        self,
        sample_id: int,
        captured_at: str,
        first_points: List[Tuple[float, float, float, float]],
        observed_hit: Dict[str, float],
    ) -> Optional[Path]:
        if not self.accepted_points:
            return None

        t0 = float(self.accepted_points[0][3])
        t_ref = float(first_points[-1][3])  # point-6 timestamp
        cx, cy, cz = self.ROBOT_HOME

        chosen_idx = int(round(float(observed_hit.get("hit_obs_index", -1.0))))
        if not (0 <= chosen_idx < len(self.accepted_points)):
            chosen_idx = -1

        ts_tag = self._timestamp_for_filename(captured_at)
        out_path = self.trajectory_dir / f"sample_{sample_id:06d}_{ts_tag}_trajectory.csv"
        fieldnames = [
            "sample_id",
            "captured_at",
            "point_index",
            "t_abs",
            "dt_from_prev_s",
            "t_from_first_accepted_s",
            "t_from_p6_s",
            "x",
            "y",
            "z",
            "in_workspace",
            "dist_to_workspace_center_mm",
            "is_first6",
            "is_after_p6",
            "is_selected_hit",
        ]

        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            prev_t: Optional[float] = None
            for idx, (x, y, z, t_abs) in enumerate(self.accepted_points):
                t_abs_f = float(t_abs)
                in_ws = 1.0 if self.in_workspace(float(x), float(y), float(z)) else 0.0
                d_center = math.sqrt((float(x) - cx) ** 2 + (float(y) - cy) ** 2 + (float(z) - cz) ** 2)
                dt_prev = (t_abs_f - prev_t) if prev_t is not None else float("nan")
                prev_t = t_abs_f

                writer.writerow(
                    {
                        "sample_id": int(sample_id),
                        "captured_at": str(captured_at),
                        "point_index": int(idx),
                        "t_abs": t_abs_f,
                        "dt_from_prev_s": float(dt_prev),
                        "t_from_first_accepted_s": float(t_abs_f - t0),
                        "t_from_p6_s": float(t_abs_f - t_ref),
                        "x": float(x),
                        "y": float(y),
                        "z": float(z),
                        "in_workspace": float(in_ws),
                        "dist_to_workspace_center_mm": float(d_center),
                        "is_first6": 1.0 if idx < self.cfg.num_input_points else 0.0,
                        "is_after_p6": 1.0 if t_abs_f >= (t_ref - 1e-6) else 0.0,
                        "is_selected_hit": 1.0 if idx == chosen_idx else 0.0,
                    }
                )

        return out_path

    @staticmethod
    def _build_row(
        sample_id: int,
        first_points: List[Tuple[float, float, float, float]],
        observed_hit: Dict[str, float],
    ) -> Dict[str, float]:
        row = RealTransferCollector._build_row(sample_id, first_points, observed_hit)

        row["has_bounce_before_hit"] = 1.0 if float(row.get("bounces_before_hit", 0.0)) > 0.5 else 0.0
        row["hit_obs_index"] = float(observed_hit.get("hit_obs_index", -1.0))
        row["hit_obs_in_workspace"] = float(observed_hit.get("hit_obs_in_workspace", row.get("is_reachable", 0.0)))
        row["hit_obs_selected_from_workspace_pool"] = float(
            observed_hit.get("hit_obs_selected_from_workspace_pool", 0.0)
        )
        row["hit_center_dist_mm"] = float(observed_hit.get("hit_center_dist_mm", float("nan")))
        return row

    def save_sample(
        self,
        first_points: List[Tuple[float, float, float, float]],
        observed_hit: Dict[str, float],
    ) -> Dict[str, float]:
        sample_id_now = int(self.sample_id)
        row = super().save_sample(first_points, observed_hit)

        try:
            traj_path = self._save_trajectory_csv(
                sample_id=sample_id_now,
                captured_at=str(row.get("captured_at", "")),
                first_points=first_points,
                observed_hit=observed_hit,
            )
            if traj_path is not None:
                self.last_trajectory_csv = traj_path
                _print(f"Saved trajectory CSV for sample_id={sample_id_now}: {traj_path}")
        except Exception as exc:
            _print(f"[WARN] Failed to save trajectory CSV for sample_id={sample_id_now}: {exc}")

        return row

    def _select_observed_hit(self) -> Optional[Dict[str, float]]:
        if len(self.first_points) < self.cfg.num_input_points or not self.accepted_points:
            return None

        # Match synthetic convention: target is selected at/after pK timestamp.
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

        hit_in_workspace = 1.0 if self.in_workspace(float(xh), float(yh), float(zh)) else 0.0
        selected_from_workspace_pool = 1.0 if best_ws_idx is not None else 0.0
        chosen_d2 = best_ws_d2 if best_ws_idx is not None else best_any_d2

        return {
            "x": float(xh),
            "y": float(yh),
            "z": float(zh),
            "vx": float(vx),
            "vy": float(vy),
            "vz": float(vz),
            "t_abs": float(th),
            "is_reachable": float(hit_in_workspace),
            "bounces_before_hit": float(bounces),
            "hit_obs_index": float(chosen_idx),
            "hit_obs_in_workspace": float(hit_in_workspace),
            "hit_obs_selected_from_workspace_pool": float(selected_from_workspace_pool),
            "hit_center_dist_mm": float(math.sqrt(max(0.0, chosen_d2))),
        }

    def _process_capture_frame(self, result: dict, frame_ts: float) -> None:
        """
        Same as base collector, plus a continuity guard for first-6 inputs.
        If a large gap occurs while collecting p1..p6, restart first-point counting.
        """
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

        # Keep the full accepted trajectory for hit-selection + trajectory export.
        self.accepted_points.append((rx, ry, rz, frame_ts))

        if self.state == self.STATE_ARMED:
            self.state = self.STATE_TRACKING
            self.tracking_start_ts = frame_ts

        # Build first N points with continuity check.
        if len(self.first_points) == 0:
            self.first_points.append((rx, ry, rz, frame_ts))
        elif len(self.first_points) < self.cfg.num_input_points:
            prev_t = float(self.first_points[-1][3])
            gap_s = float(frame_ts - prev_t)
            if gap_s > self.max_first6_gap_s:
                self.first_points = [(rx, ry, rz, frame_ts)]
                self.status_msg = (
                    f"Restarted first-{self.cfg.num_input_points} window: "
                    f"gap {gap_s*1000.0:.1f}ms > {self.max_first6_gap_s*1000.0:.1f}ms"
                )
            else:
                self.first_points.append((rx, ry, rz, frame_ts))

        if self.state == self.STATE_TRACKING and self.tracking_start_ts is not None:
            if (frame_ts - self.tracking_start_ts) > self.cfg.max_throw_s:
                self._finalize_throw_from_observations(frame_ts, "Throw ended (timeout)")
                return


def parse_args() -> CaptureConfig:
    # Parse extension-specific args first, then forward remaining args to base parser.
    ext_parser = argparse.ArgumentParser(add_help=False)
    ext_parser.add_argument(
        "--max-first6-gap-s",
        type=float,
        default=0.03,
        help="Max allowed gap between consecutive first-6 points before restart [s].",
    )
    ext_args, remaining = ext_parser.parse_known_args()

    original_argv = list(sys.argv)
    try:
        sys.argv = [sys.argv[0], *remaining]
        cfg = base_parse_args()
    finally:
        sys.argv = original_argv

    # This collector is intentionally fixed to 6-point inputs.
    if int(cfg.num_input_points) != 6:
        _print(f"Overriding --num-points={cfg.num_input_points} -> 6 for this collector.")
    cfg.num_input_points = 6
    cfg.max_first6_gap_s = max(0.0, float(ext_args.max_first6_gap_s))

    # Keep caller override if provided; otherwise use dedicated file name.
    if str(cfg.output_file_name).strip() == "real_transfer_dataset.csv":
        cfg.output_file_name = "real_transfer_true_xyz_6pt.csv"

    return cfg


def main() -> int:
    cfg = parse_args()
    collector = TrueXYZSixPointCollector(cfg)
    return collector.run()


if __name__ == "__main__":
    raise SystemExit(main())
