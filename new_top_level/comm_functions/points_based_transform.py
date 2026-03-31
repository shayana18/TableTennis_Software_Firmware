import json
import os

import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DEFAULT_POINTS_BASED_TRANSFORM_FILE = os.path.join(
    PROJECT_DIR, "camera_params", "transformation_matrix.json"
)


def transform_points(points, rotation, translation):
    points_arr = np.asarray(points, dtype=float)
    rotation_arr = np.asarray(rotation, dtype=float)
    translation_arr = np.asarray(translation, dtype=float).reshape(1, 3)
    return (rotation_arr @ points_arr.T).T + translation_arr


def cam_to_robot(rotation, translation, camera_scale_to_robot_units, x, y, z):
    camera_point = np.array(
        [[x, y, z]], dtype=float
    ) * float(camera_scale_to_robot_units)
    robot_point = transform_points(camera_point, rotation, translation)[0]
    return float(robot_point[0]), float(robot_point[1]), float(robot_point[2])


def points_based_transform(camera_points, robot_points, return_diagnostics=False):
    cam = np.asarray(camera_points, dtype=float)
    rob = np.asarray(robot_points, dtype=float)

    if cam.shape != rob.shape or cam.ndim != 2 or cam.shape[1] != 3:
        raise ValueError("camera_points and robot_points must both be Nx3 arrays")
    if cam.shape[0] < 3:
        raise ValueError("At least 3 point correspondences are required")

    cam_centroid = cam.mean(axis=0)
    rob_centroid = rob.mean(axis=0)

    cam_centered = cam - cam_centroid
    rob_centered = rob - rob_centroid

    H = cam_centered.T @ rob_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = rob_centroid - R @ cam_centroid

    if not return_diagnostics:
        return R, t

    fitted = transform_points(cam, R, t)
    residuals = rob - fitted
    per_point_err = np.linalg.norm(residuals, axis=1)
    diagnostics = {
        "num_points": int(cam.shape[0]),
        "rmse": float(np.sqrt(np.mean(np.sum(residuals**2, axis=1)))),
        "mean_error": float(np.mean(per_point_err)),
        "max_error": float(np.max(per_point_err)),
        "rotation_det": float(np.linalg.det(R)),
    }
    return R, t, diagnostics


def load_points_based_transform(input_path=DEFAULT_POINTS_BASED_TRANSFORM_FILE):
    input_path = os.path.abspath(input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    payload["rotation"] = np.asarray(payload["rotation"], dtype=float)
    payload["translation"] = np.asarray(payload["translation"], dtype=float).reshape(3)
    if "transformation_matrix" in payload:
        payload["transformation_matrix"] = np.asarray(
            payload["transformation_matrix"], dtype=float
        )
    return payload


def save_points_based_transform(
    rotation,
    translation,
    output_path=DEFAULT_POINTS_BASED_TRANSFORM_FILE,
    camera_scale_to_robot_units=None,
):
    rotation_arr = np.asarray(rotation, dtype=float)
    translation_arr = np.asarray(translation, dtype=float).reshape(3)

    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    payload = {
        "rotation": rotation_arr.tolist(),
        "translation": translation_arr.tolist(),
    }
    if camera_scale_to_robot_units is not None:
        payload["camera_scale_to_robot_units"] = float(camera_scale_to_robot_units)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return output_path
