"""
Point-pair based camera -> robot rigid transform calibration.

Computes R and t such that:
    p_robot = R @ p_camera + t

One-time calibration workflow:
1) Pick 8-15 robot poses that span the reachable workspace.
2) At each pose, place/hold a visible marker at a known robot-frame point.
3) While robot is still, average 10-30 triangulated camera-frame samples.
4) Save matched pairs:
       camera_points[i] <-> robot_points[i]
5) Run `points_based_transform(camera_points, robot_points)`.

Important:
- Use the same units in both arrays before fitting (for example both in mm).
- This solves a rigid transform only (rotation + translation, no scaling).

Under the hood (Kabsch SVD fit):
1) Compute centroids of camera and robot point sets.
2) Subtract centroids to center both sets.
3) Build cross-covariance H = cam_centered.T @ rob_centered.
4) SVD: H = U * S * V^T.
5) Rotation: R = V * U^T.
6) If det(R) < 0, fix reflection by flipping the last V row and recompute R.
7) Translation: t = centroid_robot - R @ centroid_camera.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Tuple

import numpy as np

DEFAULT_POINTS_BASED_TRANSFORM_FILE = os.path.join(
    os.path.dirname(__file__),
    "points_based_transform_result.json",
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
    """
    Apply a camera -> robot rigid transform to N points.

    Args:
        points: Input points, shape (N, 3)
        rotation: 3x3 rotation matrix R
        translation: 3-vector t (same output format as convert_to_robot_plane)
    """
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
    """
    Estimate rigid transform from matched camera/robot points (Kabsch SVD fit).

    Args:
        camera_points: Nx3 points in camera frame
        robot_points: Nx3 matching points in robot frame
        return_diagnostics: if True, also return fit quality metrics

    Returns:
        (R, t) where p_robot = R @ p_camera + t
        R shape: (3, 3), t shape: (3,)

    Solve steps:
    1) mu_c = mean(camera_points), mu_r = mean(robot_points)
    2) Xc = camera_points - mu_c, Xr = robot_points - mu_r
    3) H = Xc.T @ Xr
    4) U, S, Vt = svd(H)
    5) R = Vt.T @ U.T  (with reflection correction if det(R) < 0)
    6) t = mu_r - R @ mu_c

    This gives the least-squares rigid alignment minimizing:
        sum_i || robot_i - (R @ camera_i + t) ||^2
    """
    cam = _as_point_array(camera_points, "camera_points")
    rob = _as_point_array(robot_points, "robot_points")

    if cam.shape[0] != rob.shape[0]:
        raise ValueError(
            f"camera_points and robot_points must have the same length, got {cam.shape[0]} and {rob.shape[0]}"
        )
    if cam.shape[0] < 3:
        raise ValueError("At least 3 point pairs are required; 8-15 spread points are recommended.")

    cam_centroid = cam.mean(axis=0)
    rob_centroid = rob.mean(axis=0)

    cam_centered = cam - cam_centroid
    rob_centered = rob - rob_centroid

    # Solve optimal rotation (least-squares rigid fit, no scaling).
    H = cam_centered.T @ rob_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Enforce a proper rotation (determinant +1), no reflection.
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
    output_path: str = DEFAULT_POINTS_BASED_TRANSFORM_FILE,
    camera_scale_to_robot_units: float = 1.0,
) -> str:
    """
    Save a solved points-based transform to JSON.

    Stored equation:
        p_robot = R @ (p_camera * camera_scale_to_robot_units) + t
    """
    R = np.asarray(rotation, dtype=float)
    t = np.asarray(translation, dtype=float).reshape(3)
    if R.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {R.shape}")

    payload = {
        "camera_scale_to_robot_units": float(camera_scale_to_robot_units),
        "rotation": R.tolist(),
        "translation": t.tolist(),
    }

    out = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return out


def load_points_based_transform(
    input_path: str = DEFAULT_POINTS_BASED_TRANSFORM_FILE,
) -> dict:
    """
    Load a saved points-based transform JSON.

    Returns:
        {
          "rotation": np.ndarray (3,3),
          "translation": np.ndarray (3,),
          "camera_scale_to_robot_units": float,
          "raw": original_json_dict
        }
    """
    src = os.path.abspath(input_path)
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "rotation" not in data or "translation" not in data:
        raise ValueError(f"Invalid transform file: {src}")

    R = np.asarray(data["rotation"], dtype=float)
    t = np.asarray(data["translation"], dtype=float).reshape(3)
    if R.shape != (3, 3):
        raise ValueError(f"Invalid rotation shape in {src}: {R.shape}")

    scale = float(data.get("camera_scale_to_robot_units", 1.0))
    return {
        "rotation": R,
        "translation": t,
        "camera_scale_to_robot_units": scale,
        "raw": data,
    }


def cam_to_robot(
    R: np.ndarray,
    t: np.ndarray,
    scale: float,
    cam_x: float,
    cam_y: float,
    cam_z: float,
) -> tuple[float, float, float]:
    """Camera coords -> robot coords using points-based transform.

    p_robot = R @ (p_camera * scale) + t
    """
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
    """Robot coords -> camera coords (inverse of cam_to_robot).

    p_camera = R^T @ (p_robot - t) / scale
    """
    p_cam = R.T @ (np.array([robot_x, robot_y, robot_z]) - t) / scale
    return float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
