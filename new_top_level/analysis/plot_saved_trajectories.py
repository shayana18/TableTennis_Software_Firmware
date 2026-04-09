"""
Load saved trajectory CSVs and overlay them on one combined 3D plot.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
NEW_TOP_LEVEL_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT_DIR = NEW_TOP_LEVEL_DIR / "test_data" / "triangulation_cvs"


def _collect_input_files(input_path: str, pattern: str) -> list[Path]:
    p = Path(input_path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted([x for x in p.glob(pattern) if x.is_file()])
    return sorted([x for x in Path(".").glob(input_path) if x.is_file()])


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _load_trajectory_csv(csv_path: Path) -> dict[str, object]:
    raw_points: list[tuple[float, float, float]] = []
    predicted_points: list[tuple[float, float, float]] = []
    intercept_point: tuple[float, float, float] | None = None

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            point_type = (row.get("point_type") or "").strip().lower()
            x = _parse_float(row.get("x_mm"))
            y = _parse_float(row.get("y_mm"))
            z = _parse_float(row.get("z_mm"))
            if x is None or y is None or z is None:
                continue

            point = (x, y, z)
            if point_type == "raw":
                raw_points.append(point)
            elif point_type == "predicted":
                predicted_points.append(point)
            elif point_type == "intercept":
                intercept_point = point

    return {
        "name": csv_path.stem,
        "path": csv_path,
        "raw_points": raw_points,
        "predicted_points": predicted_points,
        "intercept_point": intercept_point,
    }


def _set_equal_3d(ax, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> None:
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


def _plot_combined(trajectories: list[dict[str, object]]) -> None:
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    cmap = plt.get_cmap("tab20")

    all_x: list[float] = []
    all_y: list[float] = []
    all_z: list[float] = []

    for i, trajectory in enumerate(trajectories):
        color = cmap(i % cmap.N)
        raw_points = trajectory["raw_points"]
        predicted_points = trajectory["predicted_points"]
        intercept_point = trajectory["intercept_point"]
        label = trajectory["name"]

        if raw_points:
            raw = np.array(raw_points, dtype=float)
            ax.scatter(raw[:, 0], raw[:, 1], raw[:, 2], color=color, s=18, alpha=0.35)
            ax.plot(raw[:, 0], raw[:, 1], raw[:, 2], color=color, alpha=0.25)
            all_x.extend(raw[:, 0].tolist())
            all_y.extend(raw[:, 1].tolist())
            all_z.extend(raw[:, 2].tolist())

        if predicted_points:
            pred = np.array(predicted_points, dtype=float)
            ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], color=color, linewidth=2.2, label=label)
            all_x.extend(pred[:, 0].tolist())
            all_y.extend(pred[:, 1].tolist())
            all_z.extend(pred[:, 2].tolist())
        elif raw_points:
            ax.plot([], [], [], color=color, linewidth=2.2, label=label)

        if intercept_point is not None:
            ax.scatter(
                [intercept_point[0]],
                [intercept_point[1]],
                [intercept_point[2]],
                color=color,
                marker="*",
                s=90,
            )
            all_x.append(intercept_point[0])
            all_y.append(intercept_point[1])
            all_z.append(intercept_point[2])

    if all_x and all_y and all_z:
        _set_equal_3d(
            ax,
            np.array(all_x, dtype=float),
            np.array(all_y, dtype=float),
            np.array(all_z, dtype=float),
        )

    ax.set_title("Combined Saved Trajectories")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay saved trajectory CSV files on one 3D plot."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_DIR),
        help="Input CSV file, directory, or glob pattern.",
    )
    parser.add_argument(
        "--pattern",
        default="throw_*.csv",
        help="Filename pattern when --input is a directory.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on number of files to load (0 = all).",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    input_files = _collect_input_files(args.input, args.pattern)
    if not input_files:
        print(f"No trajectory CSV files found for: {args.input}")
        return 1

    if args.max_files > 0:
        input_files = input_files[: args.max_files]

    trajectories = [_load_trajectory_csv(path) for path in input_files]
    print(f"Loaded {len(trajectories)} trajectory file(s).")
    _plot_combined(trajectories)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
