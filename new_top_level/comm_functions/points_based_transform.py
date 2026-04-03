"""
Point-pair based camera -> robot rigid transform helpers.

Stored transform equation:
    p_robot = R @ (p_camera * camera_scale_to_robot_units) + t
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


DEFAULT_POINTS_BASED_TRANSFORM_FILE = (
    Path(__file__).resolve().parent.parent / "camera_params" / "transformation_matrix.json"
)


def _as_point_array(points: Iterable[Iterable[float]], name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {arr.shape}")
    return arr


def transform_points(
    points: Iterable[Iterable[float]],
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Apply camera->robot transform to N points."""
    pts = _as_point_array(points, "points")
    R = np.asarray(rotation, dtype=float)
    t = np.asarray(translation, dtype=float).reshape(3)
    if R.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {R.shape}")
    return (R @ pts.T).T + t


def points_based_transform(
    camera_points: Iterable[Iterable[float]],
    robot_points: Iterable[Iterable[float]],
    *,
    return_diagnostics: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, dict]:
    """Estimate rigid transform from matched camera/robot points (Kabsch)."""
    cam = _as_point_array(camera_points, "camera_points")
    rob = _as_point_array(robot_points, "robot_points")

    if cam.shape[0] != rob.shape[0]:
        raise ValueError(
            f"camera_points and robot_points length mismatch: {cam.shape[0]} vs {rob.shape[0]}"
        )
    if cam.shape[0] < 3:
        raise ValueError("At least 3 point pairs are required.")

    cam_centroid = cam.mean(axis=0)
    rob_centroid = rob.mean(axis=0)

    cam_centered = cam - cam_centroid
    rob_centered = rob - rob_centroid

    H = cam_centered.T @ rob_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0.0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T

    t = rob_centroid - (R @ cam_centroid)

    if not return_diagnostics:
        return R, t

    fitted = transform_points(cam, R, t)
    residuals = rob - fitted
    errors = np.linalg.norm(residuals, axis=1)
    diagnostics = {
        "num_points": int(cam.shape[0]),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mean_error": float(np.mean(errors)),
        "max_error": float(np.max(errors)),
        "rotation_det": float(np.linalg.det(R)),
    }
    return R, t, diagnostics


def save_points_based_transform(
    rotation: np.ndarray,
    translation: np.ndarray,
    *,
    output_path: str | Path = DEFAULT_POINTS_BASED_TRANSFORM_FILE,
    camera_scale_to_robot_units: float = 1.0,
) -> str:
    """Save transform JSON."""
    R = np.asarray(rotation, dtype=float)
    t = np.asarray(translation, dtype=float).reshape(3)
    if R.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {R.shape}")

    payload = {
        "camera_scale_to_robot_units": float(camera_scale_to_robot_units),
        "rotation": R.tolist(),
        "translation": t.tolist(),
    }

    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return str(out)


def load_points_based_transform(input_path: str | Path = DEFAULT_POINTS_BASED_TRANSFORM_FILE) -> dict:
    """Load transform JSON with backward-compatible key behavior.

    Returns numeric `rotation`/`translation` arrays plus every original key
    from disk so legacy callers that inspect extra payload fields do not break.
    """
    src = Path(input_path).resolve()
    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "rotation" not in data or "translation" not in data:
        raise ValueError(f"Invalid transform file: {src}")

    R = np.asarray(data["rotation"], dtype=float)
    t = np.asarray(data["translation"], dtype=float).reshape(3)
    if R.shape != (3, 3):
        raise ValueError(f"Invalid rotation shape in {src}: {R.shape}")

    payload = dict(data)
    payload["rotation"] = R
    payload["translation"] = t
    payload["camera_scale_to_robot_units"] = float(data.get("camera_scale_to_robot_units", 1.0))
    if "transformation_matrix" in data:
        payload["transformation_matrix"] = np.asarray(data["transformation_matrix"], dtype=float)
    payload["raw"] = data
    return payload


def cam_to_robot(
    R: np.ndarray,
    t: np.ndarray,
    scale: float,
    cam_x: float,
    cam_y: float,
    cam_z: float,
) -> tuple[float, float, float]:
    """Camera coords -> robot coords."""
    p = R @ (np.array([cam_x, cam_y, cam_z]) * scale) + t
    return float(p[0]), float(p[1]), float(p[2])


def robot_to_cam(
    R: np.ndarray,
    t: np.ndarray,
    scale: float,
    robot_x: float,
    robot_y: float,
    robot_z: float,
) -> tuple[float, float, float]:
    """Robot coords -> camera coords (inverse of cam_to_robot)."""
    p_cam = R.T @ (np.array([robot_x, robot_y, robot_z]) - t) / scale
    return float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
