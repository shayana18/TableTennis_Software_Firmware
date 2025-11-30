#!/usr/bin/env python3
"""Compute baseline from camera extrinsic files.

Reads `camera_parameters/camera0_rot_trans.dat` and
`camera_parameters/camera1_rot_trans.dat`, parses the translation
vectors and computes the baseline as the Euclidean norm of the
translation between the camera centers.

If only one camera file is present, uses that camera's T vector
directly (useful when camera0 is the origin and camera1 stores
the relative translation).
"""
from __future__ import annotations

import argparse
import os
from typing import Iterable, List, Optional

import numpy as np


def parse_T_from_file(path: str) -> Optional[np.ndarray]:
    """Parse the translation vector 'T' from a rot_trans.dat file.

    Returns a (3,) numpy array if successful, otherwise None.
    """
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh]

    # Find the line with 'T:' and then collect numbers from following lines
    t_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("T:"):
            t_idx = i
            break

    if t_idx is None:
        return None

    vals: List[float] = []
    # Read subsequent lines and parse floats until we have 3 values
    for ln in lines[t_idx + 1 :]:
        if not ln:
            continue
        # Stop if a new section starts (e.g., 'R:' or another label)
        if ln.endswith(":") and not any(ch.isdigit() for ch in ln):
            break
        # split tokens and try to parse floats
        for token in ln.split():
            try:
                vals.append(float(token))
            except ValueError:
                # ignore non-float tokens
                continue
        if len(vals) >= 3:
            break

    if len(vals) < 3:
        return None

    return np.array(vals[:3], dtype=float)


def compute_baseline(cam0_path: str, cam1_path: str) -> Optional[float]:
    t0 = parse_T_from_file(cam0_path)
    t1 = parse_T_from_file(cam1_path)

    if t0 is None and t1 is None:
        print(f"No translation vectors found in {cam0_path} or {cam1_path}")
        return None

    if t0 is None:
        T = t1
    elif t1 is None:
        T = t0
    else:
        # baseline is the distance between camera centers
        T = t1 - t0

    baseline = float(np.linalg.norm(T))
    return baseline


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compute stereo baseline from extrinsic files")
    p.add_argument("--cam0", default="camera_parameters/camera0_rot_trans.dat",
                   help="Path to camera0 rot_trans file")
    p.add_argument("--cam1", default="camera_parameters/camera1_rot_trans.dat",
                   help="Path to camera1 rot_trans file")
    args = p.parse_args(argv)

    baseline = compute_baseline(args.cam0, args.cam1)
    if baseline is None:
        return 2

    print(f"Estimated baseline: {baseline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
