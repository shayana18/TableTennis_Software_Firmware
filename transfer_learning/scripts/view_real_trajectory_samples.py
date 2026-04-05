#!/usr/bin/env python3
"""
Interactive reviewer for per-sample real trajectory CSV files.

What it does:
- Cycles through `sample_trajectories/*_trajectory.csv` files with keyboard.
- Plots full trajectory in:
  - 3D
  - XY plane
  - XZ plane
  - YZ plane
- Overlays table + workspace geometry.
- Highlights:
  - first 6 points (`is_first6==1`) in orange,
  - selected interception target (`is_selected_hit==1`) as red star.
- Lets you label each sample as:
  - good
  - ok
  - bad
- Writes outputs continuously:
  - labels ledger CSV
  - good-only filtered CSV
  - good+ok filtered CSV

Keyboard controls:
- Right / N / Space: next sample
- Left / P / Backspace: previous sample
- Home: first sample
- End: last sample
- G: label current sample as good
- O: label current sample as ok
- B: label current sample as bad
- U: clear current sample label
- Q / Esc: quit
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABEL_GOOD = "good"
LABEL_OK = "ok"
LABEL_BAD = "bad"
VALID_LABELS = {LABEL_GOOD, LABEL_OK, LABEL_BAD}


@dataclass(frozen=True)
class GeometryConfig:
    table_length_mm: float = 2740.0
    table_width_mm: float = 1525.0
    table_z_mm: float = -1150.74
    workspace_a_mm: float = 720.0
    workspace_b_mm: float = 470.0
    workspace_z_min_mm: float = -1020.0
    workspace_z_max_mm: float = -800.0
    workspace_center_x_mm: float = 0.0
    workspace_center_y_mm: float = 0.0
    workspace_center_z_mm: float = -900.0


@dataclass(frozen=True)
class SampleRef:
    path: Path
    sample_id: int


def _print(*args) -> None:
    print("[traj_reviewer]", *args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive trajectory reviewer with Good/Ok/Bad labeling.")
    parser.add_argument(
        "--trajectory-dir",
        type=str,
        default="",
        help=(
            "Path to folder containing *_trajectory.csv files. "
            "If omitted, auto-picks latest learning_data/real_data/**/sample_trajectories."
        ),
    )
    parser.add_argument("--start-index", type=int, default=0, help="Initial file index (0-based).")
    parser.add_argument(
        "--sort",
        choices=["name", "mtime"],
        default="name",
        help="File ordering for browsing.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_trajectory.csv",
        help="Filename pattern inside trajectory-dir.",
    )

    parser.add_argument(
        "--dataset-csv",
        type=str,
        default="",
        help=(
            "Optional source dataset CSV to filter by sample_id. "
            "If omitted, script auto-finds one in trajectory-dir parent."
        ),
    )
    parser.add_argument(
        "--labels-csv",
        type=str,
        default="",
        help="Output labels CSV path. Default: <trajectory-dir parent>/sample_review_labels.csv",
    )
    parser.add_argument(
        "--good-only-csv",
        type=str,
        default="",
        help="Output CSV containing only samples labeled good. Default: <trajectory-dir parent>/samples_good.csv",
    )
    parser.add_argument(
        "--good-ok-csv",
        type=str,
        default="",
        help="Output CSV containing samples labeled good or ok. Default: <trajectory-dir parent>/samples_good_ok.csv",
    )
    parser.add_argument(
        "--no-auto-next",
        action="store_true",
        help="Do not auto-advance after assigning a label.",
    )

    parser.add_argument("--table-length-mm", type=float, default=2740.0)
    parser.add_argument("--table-width-mm", type=float, default=1525.0)
    parser.add_argument("--table-z-mm", type=float, default=-1150.74)
    parser.add_argument("--workspace-a-mm", type=float, default=720.0)
    parser.add_argument("--workspace-b-mm", type=float, default=470.0)
    parser.add_argument("--workspace-z-min-mm", type=float, default=-1020.0)
    parser.add_argument("--workspace-z-max-mm", type=float, default=-800.0)
    parser.add_argument("--workspace-center-x-mm", type=float, default=0.0)
    parser.add_argument("--workspace-center-y-mm", type=float, default=0.0)
    parser.add_argument("--workspace-center-z-mm", type=float, default=-900.0)

    parser.add_argument("--show-point-index", action="store_true", help="Annotate each trajectory point with point_index.")
    parser.add_argument("--hide-first6", action="store_true", help="Disable first-6 highlighting.")

    parser.add_argument("--dpi", type=int, default=115, help="Figure DPI.")
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(14.0, 9.0),
        metavar=("W", "H"),
        help="Figure size in inches.",
    )
    parser.add_argument("--line-width", type=float, default=2.0, help="Trajectory line width.")
    parser.add_argument("--marker-size", type=float, default=24.0, help="Trajectory point marker size.")
    return parser.parse_args()


def auto_find_trajectory_dir(repo_root: Path) -> Optional[Path]:
    candidates = [p for p in repo_root.glob("learning_data/real_data/**/sample_trajectories") if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_trajectory_files(traj_dir: Path, pattern: str, sort_mode: str) -> List[Path]:
    files = [p for p in traj_dir.glob(pattern) if p.is_file()]
    if sort_mode == "mtime":
        files.sort(key=lambda p: p.stat().st_mtime)
    else:
        files.sort(key=lambda p: p.name)
    return files


def _safe_col(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), float(default), dtype=float)
    vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    vals = np.where(np.isfinite(vals), vals, float(default))
    return vals


def _first_true_index(mask: np.ndarray) -> Optional[int]:
    idx = np.where(mask)[0]
    return int(idx[0]) if idx.size > 0 else None


def _set_equalish_3d(ax, x: np.ndarray, y: np.ndarray, z: np.ndarray, margin: float = 40.0) -> None:
    xmin, xmax = float(np.min(x) - margin), float(np.max(x) + margin)
    ymin, ymax = float(np.min(y) - margin), float(np.max(y) + margin)
    zmin, zmax = float(np.min(z) - margin), float(np.max(z) + margin)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    xr = max(1.0, xmax - xmin)
    yr = max(1.0, ymax - ymin)
    zr = max(1.0, zmax - zmin)
    ax.set_box_aspect((xr, yr, zr))


def sample_id_from_trajectory_csv(path: Path) -> Optional[int]:
    try:
        df = pd.read_csv(path, usecols=["sample_id"], nrows=1)
        if df.empty:
            return None
        sid = pd.to_numeric(df.iloc[0]["sample_id"], errors="coerce")
        if np.isnan(sid):
            return None
        return int(sid)
    except Exception:
        return None


def build_sample_refs(files: List[Path]) -> List[SampleRef]:
    refs: List[SampleRef] = []
    fallback_id = 1
    for p in files:
        sid = sample_id_from_trajectory_csv(p)
        if sid is None:
            sid = fallback_id
            fallback_id += 1
        refs.append(SampleRef(path=p, sample_id=int(sid)))
    return refs


def auto_find_dataset_csv(traj_dir: Path) -> Optional[Path]:
    parent = traj_dir.parent
    candidates = [p for p in parent.glob("*.csv") if p.is_file()]
    if not candidates:
        return None

    scored: List[Tuple[int, int, float, Path]] = []
    for p in candidates:
        name = p.name.lower()
        if "trajectory" in name or "rejected" in name or "qc_metrics" in name:
            continue
        try:
            df_head = pd.read_csv(p, nrows=10)
        except Exception:
            continue
        cols = set(df_head.columns)
        has_sid = 1 if "sample_id" in cols else 0
        looks_main = 1 if {"x1", "y1", "z1", "x_hit", "y_hit", "z_hit"}.issubset(cols) else 0
        mtime = p.stat().st_mtime
        scored.append((looks_main, has_sid, mtime, p))

    if not scored:
        return None
    scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    return scored[0][3]


def normalize_label(value: str) -> Optional[str]:
    v = (value or "").strip().lower()
    return v if v in VALID_LABELS else None


class TrajectoryReviewer:
    def __init__(
        self,
        sample_refs: List[SampleRef],
        geom: GeometryConfig,
        start_index: int,
        show_point_index: bool,
        show_first6: bool,
        line_width: float,
        marker_size: float,
        figsize: Tuple[float, float],
        dpi: int,
        labels_csv: Path,
        good_only_csv: Path,
        good_ok_csv: Path,
        dataset_df: Optional[pd.DataFrame],
        auto_next_after_label: bool,
    ) -> None:
        if not sample_refs:
            raise ValueError("No trajectory files to review.")

        self.sample_refs = sample_refs
        self.geom = geom
        self.idx = int(np.clip(start_index, 0, len(sample_refs) - 1))
        self.show_point_index = bool(show_point_index)
        self.show_first6 = bool(show_first6)
        self.line_width = float(line_width)
        self.marker_size = float(marker_size)
        self.labels_csv = labels_csv
        self.good_only_csv = good_only_csv
        self.good_ok_csv = good_ok_csv
        self.dataset_df = dataset_df
        self.auto_next_after_label = bool(auto_next_after_label)

        # sample_id -> label
        self.labels: Dict[int, str] = self._load_existing_labels()

        self.fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
        gs = self.fig.add_gridspec(2, 2)
        self.ax3d = self.fig.add_subplot(gs[0, 0], projection="3d")
        self.ax_xy = self.fig.add_subplot(gs[0, 1])
        self.ax_xz = self.fig.add_subplot(gs[1, 0])
        self.ax_yz = self.fig.add_subplot(gs[1, 1])
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # Write outputs once on startup so files exist immediately.
        self._write_outputs()

    def _load_existing_labels(self) -> Dict[int, str]:
        if not self.labels_csv.exists():
            return {}
        try:
            df = pd.read_csv(self.labels_csv)
        except Exception:
            return {}
        out: Dict[int, str] = {}
        for _, row in df.iterrows():
            sid = pd.to_numeric(row.get("sample_id"), errors="coerce")
            if np.isnan(sid):
                continue
            label = normalize_label(str(row.get("label", "")))
            if label is None:
                continue
            out[int(sid)] = label
        return out

    def _label_counts(self) -> Tuple[int, int, int, int]:
        vals = list(self.labels.values())
        return (
            sum(1 for v in vals if v == LABEL_GOOD),
            sum(1 for v in vals if v == LABEL_OK),
            sum(1 for v in vals if v == LABEL_BAD),
            len(vals),
        )

    def _labels_table(self) -> pd.DataFrame:
        now = datetime.now().isoformat(timespec="seconds")
        rows = []
        for sref in self.sample_refs:
            sid = int(sref.sample_id)
            lbl = self.labels.get(sid, "")
            if not lbl:
                continue
            rows.append(
                {
                    "sample_id": sid,
                    "label": lbl,
                    "trajectory_file": sref.path.name,
                    "updated_at": now,
                }
            )
        if not rows:
            return pd.DataFrame(columns=["sample_id", "label", "trajectory_file", "updated_at"])
        df = pd.DataFrame(rows).drop_duplicates(subset=["sample_id"], keep="last")
        df = df.sort_values(["sample_id"]).reset_index(drop=True)
        return df

    def _filter_dataset_by_ids(self, ids: set[int]) -> pd.DataFrame:
        if self.dataset_df is None:
            # Fallback: emit review table rows only.
            base = self._labels_table()
            return base[base["sample_id"].isin(list(ids))].copy()

        df = self.dataset_df.copy()
        if "sample_id" not in df.columns:
            # If dataset has no sample_id, cannot align robustly; fallback to label table.
            base = self._labels_table()
            return base[base["sample_id"].isin(list(ids))].copy()

        sid_series = pd.to_numeric(df["sample_id"], errors="coerce").fillna(-1).astype(int)
        mask = sid_series.isin(list(ids))
        out = df.loc[mask].copy()
        return out

    def _write_outputs(self) -> None:
        self.labels_csv.parent.mkdir(parents=True, exist_ok=True)
        self.good_only_csv.parent.mkdir(parents=True, exist_ok=True)
        self.good_ok_csv.parent.mkdir(parents=True, exist_ok=True)

        labels_df = self._labels_table()
        labels_df.to_csv(self.labels_csv, index=False)

        good_ids = {sid for sid, lbl in self.labels.items() if lbl == LABEL_GOOD}
        good_ok_ids = {sid for sid, lbl in self.labels.items() if lbl in {LABEL_GOOD, LABEL_OK}}

        good_df = self._filter_dataset_by_ids(good_ids)
        good_ok_df = self._filter_dataset_by_ids(good_ok_ids)

        good_df.to_csv(self.good_only_csv, index=False)
        good_ok_df.to_csv(self.good_ok_csv, index=False)

    def _apply_label(self, label: str) -> None:
        sref = self.sample_refs[self.idx]
        self.labels[int(sref.sample_id)] = label
        self._write_outputs()
        g, o, b, n = self._label_counts()
        _print(
            f"sample_id={sref.sample_id} -> {label} | labeled={n}/{len(self.sample_refs)} "
            f"(good={g}, ok={o}, bad={b})"
        )
        if self.auto_next_after_label:
            self.idx = (self.idx + 1) % len(self.sample_refs)
        self._draw_sample()

    def _clear_label(self) -> None:
        sref = self.sample_refs[self.idx]
        sid = int(sref.sample_id)
        if sid in self.labels:
            self.labels.pop(sid, None)
            self._write_outputs()
            _print(f"sample_id={sid} label cleared")
        self._draw_sample()

    def _draw_overlays(self, ax3d, ax_xy, ax_xz, ax_yz) -> None:
        g = self.geom
        half_w = 0.5 * g.table_width_mm
        tx = np.array([-half_w, half_w, half_w, -half_w, -half_w], dtype=float)
        ty = np.array([0.0, 0.0, g.table_length_mm, g.table_length_mm, 0.0], dtype=float)
        tz = np.full_like(tx, g.table_z_mm)

        ax3d.plot(tx, ty, tz, color="saddlebrown", linewidth=1.8, label="Table")
        ax_xy.plot(tx, ty, color="saddlebrown", linewidth=1.4, label="Table")
        ax_xz.plot([-half_w, half_w], [g.table_z_mm, g.table_z_mm], color="saddlebrown", linewidth=1.4, label="Table z")
        ax_yz.plot([0.0, g.table_length_mm], [g.table_z_mm, g.table_z_mm], color="saddlebrown", linewidth=1.4, label="Table z")

        th = np.linspace(0.0, 2.0 * np.pi, 220)
        ex = g.workspace_a_mm * np.cos(th) + g.workspace_center_x_mm
        ey = g.workspace_b_mm * np.sin(th) + g.workspace_center_y_mm
        ez_min = np.full_like(ex, g.workspace_z_min_mm)
        ez_max = np.full_like(ex, g.workspace_z_max_mm)

        ax3d.plot(ex, ey, ez_min, color="tab:green", linewidth=1.4, label="Workspace")
        ax3d.plot(ex, ey, ez_max, color="tab:green", linewidth=1.4)
        for k in [0, 55, 110, 165]:
            ax3d.plot(
                [ex[k], ex[k]],
                [ey[k], ey[k]],
                [g.workspace_z_min_mm, g.workspace_z_max_mm],
                color="tab:green",
                linewidth=1.0,
                alpha=0.5,
            )

        ax_xy.plot(ex, ey, color="tab:green", linewidth=1.3, label="Workspace")
        ax_xz.plot(
            [g.workspace_center_x_mm - g.workspace_a_mm, g.workspace_center_x_mm + g.workspace_a_mm],
            [g.workspace_z_min_mm, g.workspace_z_min_mm],
            color="tab:green",
            linewidth=1.2,
            label="Workspace proj",
        )
        ax_xz.plot(
            [g.workspace_center_x_mm - g.workspace_a_mm, g.workspace_center_x_mm + g.workspace_a_mm],
            [g.workspace_z_max_mm, g.workspace_z_max_mm],
            color="tab:green",
            linewidth=1.2,
        )
        ax_yz.plot(
            [g.workspace_center_y_mm - g.workspace_b_mm, g.workspace_center_y_mm + g.workspace_b_mm],
            [g.workspace_z_min_mm, g.workspace_z_min_mm],
            color="tab:green",
            linewidth=1.2,
            label="Workspace proj",
        )
        ax_yz.plot(
            [g.workspace_center_y_mm - g.workspace_b_mm, g.workspace_center_y_mm + g.workspace_b_mm],
            [g.workspace_z_max_mm, g.workspace_z_max_mm],
            color="tab:green",
            linewidth=1.2,
        )

    def _draw_sample(self) -> None:
        for ax in [self.ax3d, self.ax_xy, self.ax_xz, self.ax_yz]:
            ax.clear()

        sref = self.sample_refs[self.idx]
        path = sref.path
        sid = int(sref.sample_id)
        cur_label = self.labels.get(sid, "UNLABELED")

        df = pd.read_csv(path)
        if df.empty:
            self.fig.suptitle(f"[{self.idx + 1}/{len(self.sample_refs)}] sample_id={sid} | {path.name} (empty)")
            self.fig.canvas.draw_idle()
            return

        x = _safe_col(df, "x", default=np.nan)
        y = _safe_col(df, "y", default=np.nan)
        z = _safe_col(df, "z", default=np.nan)
        point_index = _safe_col(df, "point_index", default=np.nan).astype(int)
        first6_mask = _safe_col(df, "is_first6", default=0.0) > 0.5
        hit_mask = _safe_col(df, "is_selected_hit", default=0.0) > 0.5

        self.ax3d.plot(x, y, z, "-o", color="tab:blue", linewidth=self.line_width, markersize=3.0, label="Trajectory")
        self.ax_xy.plot(x, y, "-o", color="tab:blue", linewidth=1.8, markersize=3.0, label="Trajectory")
        self.ax_xz.plot(x, z, "-o", color="tab:blue", linewidth=1.8, markersize=3.0, label="Trajectory")
        self.ax_yz.plot(y, z, "-o", color="tab:blue", linewidth=1.8, markersize=3.0, label="Trajectory")

        if self.show_first6 and np.any(first6_mask):
            xf, yf, zf = x[first6_mask], y[first6_mask], z[first6_mask]
            first6_color = "tab:orange"
            self.ax3d.scatter(xf, yf, zf, s=self.marker_size, color=first6_color, label="First 6")
            self.ax_xy.scatter(xf, yf, s=self.marker_size, color=first6_color, label="First 6")
            self.ax_xz.scatter(xf, zf, s=self.marker_size, color=first6_color, label="First 6")
            self.ax_yz.scatter(yf, zf, s=self.marker_size, color=first6_color, label="First 6")

        hit_idx = _first_true_index(hit_mask)
        if hit_idx is not None:
            hx, hy, hz = float(x[hit_idx]), float(y[hit_idx]), float(z[hit_idx])
            self.ax3d.scatter([hx], [hy], [hz], color="tab:red", marker="*", s=170, label="Selected hit")
            self.ax_xy.scatter([hx], [hy], color="tab:red", marker="*", s=140, label="Selected hit")
            self.ax_xz.scatter([hx], [hz], color="tab:red", marker="*", s=140, label="Selected hit")
            self.ax_yz.scatter([hy], [hz], color="tab:red", marker="*", s=140, label="Selected hit")

        if self.show_point_index:
            for i in range(len(df)):
                lbl = str(point_index[i]) if np.isfinite(point_index[i]) else str(i)
                self.ax3d.text(x[i], y[i], z[i], lbl, fontsize=7)

        self._draw_overlays(self.ax3d, self.ax_xy, self.ax_xz, self.ax_yz)

        all_x = np.concatenate(
            [
                x,
                np.array(
                    [
                        -0.5 * self.geom.table_width_mm,
                        0.5 * self.geom.table_width_mm,
                        self.geom.workspace_center_x_mm - self.geom.workspace_a_mm,
                        self.geom.workspace_center_x_mm + self.geom.workspace_a_mm,
                    ],
                    dtype=float,
                ),
            ]
        )
        all_y = np.concatenate(
            [
                y,
                np.array(
                    [
                        0.0,
                        self.geom.table_length_mm,
                        self.geom.workspace_center_y_mm - self.geom.workspace_b_mm,
                        self.geom.workspace_center_y_mm + self.geom.workspace_b_mm,
                    ],
                    dtype=float,
                ),
            ]
        )
        all_z = np.concatenate(
            [
                z,
                np.array(
                    [
                        self.geom.table_z_mm,
                        self.geom.workspace_z_min_mm,
                        self.geom.workspace_z_max_mm,
                    ],
                    dtype=float,
                ),
            ]
        )
        _set_equalish_3d(self.ax3d, all_x, all_y, all_z, margin=40.0)

        self.ax3d.set_xlabel("X [mm]")
        self.ax3d.set_ylabel("Y [mm]")
        self.ax3d.set_zlabel("Z [mm]")
        self.ax_xy.set_title("XY Plane")
        self.ax_xz.set_title("XZ Plane")
        self.ax_yz.set_title("YZ Plane")
        self.ax_xy.set_xlabel("X [mm]")
        self.ax_xy.set_ylabel("Y [mm]")
        self.ax_xz.set_xlabel("X [mm]")
        self.ax_xz.set_ylabel("Z [mm]")
        self.ax_yz.set_xlabel("Y [mm]")
        self.ax_yz.set_ylabel("Z [mm]")
        self.ax_xy.grid(alpha=0.28)
        self.ax_xz.grid(alpha=0.28)
        self.ax_yz.grid(alpha=0.28)
        self.ax_xy.set_aspect("equal", adjustable="box")

        first_t = float(_safe_col(df, "t_abs", default=np.nan)[0]) if "t_abs" in df.columns else float("nan")
        g, o, b, n = self._label_counts()
        self.fig.suptitle(
            (
                f"[{self.idx + 1}/{len(self.sample_refs)}] sample_id={sid} | label={cur_label} | "
                f"points={len(df)} | t_abs0={first_t:.3f}\n"
                f"Labeled={n}/{len(self.sample_refs)} (good={g}, ok={o}, bad={b}) | "
                "Keys: G/O/B label, U clear, Right/Left nav, Q quit"
            ),
            fontsize=10,
        )

        self.ax3d.legend(loc="best", fontsize=8)
        self.ax_xy.legend(loc="best", fontsize=8)
        self.ax_xz.legend(loc="best", fontsize=8)
        self.ax_yz.legend(loc="best", fontsize=8)
        self.fig.canvas.draw_idle()

    def _on_key(self, event) -> None:
        if event.key is None:
            return
        key = event.key.lower()
        if key in ("right", "n", " "):
            self.idx = (self.idx + 1) % len(self.sample_refs)
            self._draw_sample()
        elif key in ("left", "p", "backspace"):
            self.idx = (self.idx - 1) % len(self.sample_refs)
            self._draw_sample()
        elif key == "home":
            self.idx = 0
            self._draw_sample()
        elif key == "end":
            self.idx = len(self.sample_refs) - 1
            self._draw_sample()
        elif key == "g":
            self._apply_label(LABEL_GOOD)
        elif key == "o":
            self._apply_label(LABEL_OK)
        elif key == "b":
            self._apply_label(LABEL_BAD)
        elif key == "u":
            self._clear_label()
        elif key in ("q", "escape"):
            plt.close(self.fig)

    def run(self) -> None:
        _print("Controls: G/O/B label, U clear, Right/Left navigate, Home/End, Q quit")
        self._draw_sample()
        plt.show()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    if args.trajectory_dir:
        traj_dir = Path(args.trajectory_dir).expanduser().resolve()
    else:
        found = auto_find_trajectory_dir(repo_root)
        if found is None:
            _print("No sample_trajectories folder found. Pass --trajectory-dir.")
            return 1
        traj_dir = found.resolve()

    if not traj_dir.exists() or not traj_dir.is_dir():
        _print(f"Trajectory directory not found: {traj_dir}")
        return 1

    files = load_trajectory_files(traj_dir, pattern=args.pattern, sort_mode=args.sort)
    if not files:
        _print(f"No files matched pattern '{args.pattern}' in: {traj_dir}")
        return 1

    sample_refs = build_sample_refs(files)

    if args.dataset_csv:
        dataset_csv = Path(args.dataset_csv).expanduser().resolve()
    else:
        dataset_csv = auto_find_dataset_csv(traj_dir)

    dataset_df: Optional[pd.DataFrame] = None
    if dataset_csv is not None and dataset_csv.exists():
        try:
            dataset_df = pd.read_csv(dataset_csv)
            _print(f"Using dataset CSV: {dataset_csv}")
            _print(f"Dataset rows: {len(dataset_df)}")
        except Exception as exc:
            _print(f"[WARN] Failed reading dataset CSV {dataset_csv}: {exc}")
            dataset_df = None
    else:
        _print("[WARN] Dataset CSV not found; outputs will be label-table based.")

    out_parent = traj_dir.parent
    labels_csv = Path(args.labels_csv).expanduser().resolve() if args.labels_csv else (out_parent / "sample_review_labels.csv")
    good_only_csv = Path(args.good_only_csv).expanduser().resolve() if args.good_only_csv else (out_parent / "samples_good.csv")
    good_ok_csv = Path(args.good_ok_csv).expanduser().resolve() if args.good_ok_csv else (out_parent / "samples_good_ok.csv")

    geom = GeometryConfig(
        table_length_mm=float(args.table_length_mm),
        table_width_mm=float(args.table_width_mm),
        table_z_mm=float(args.table_z_mm),
        workspace_a_mm=float(args.workspace_a_mm),
        workspace_b_mm=float(args.workspace_b_mm),
        workspace_z_min_mm=float(args.workspace_z_min_mm),
        workspace_z_max_mm=float(args.workspace_z_max_mm),
        workspace_center_x_mm=float(args.workspace_center_x_mm),
        workspace_center_y_mm=float(args.workspace_center_y_mm),
        workspace_center_z_mm=float(args.workspace_center_z_mm),
    )

    _print(f"Trajectory directory: {traj_dir}")
    _print(f"Found {len(sample_refs)} trajectory files")
    _print(f"Labels CSV: {labels_csv}")
    _print(f"Good-only CSV: {good_only_csv}")
    _print(f"Good+Ok CSV: {good_ok_csv}")

    reviewer = TrajectoryReviewer(
        sample_refs=sample_refs,
        geom=geom,
        start_index=int(args.start_index),
        show_point_index=bool(args.show_point_index),
        show_first6=not bool(args.hide_first6),
        line_width=float(args.line_width),
        marker_size=float(args.marker_size),
        figsize=(float(args.figsize[0]), float(args.figsize[1])),
        dpi=int(args.dpi),
        labels_csv=labels_csv,
        good_only_csv=good_only_csv,
        good_ok_csv=good_ok_csv,
        dataset_df=dataset_df,
        auto_next_after_label=not bool(args.no_auto_next),
    )
    reviewer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

