# what I need to do here is take a 

import numpy as np
from typing import Tuple

def rotation_translation_from_pose(
    x: float,
    y: float,
    z: float,
    yaw: float,
    pitch: float,
    roll: float = 0.0,
    *,
    degrees: bool = True,
    order: str = "ZYX",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (R_BC, t_BC) such that:
        p_B = R_BC @ p_C + t_BC

    Angles describe the camera orientation relative to the robot base frame.
    Convention:
      - yaw  : rotation about +Z (base)
      - pitch: rotation about +Y (base)
      - roll : rotation about +X (base)
      - Right-hand rule
    'order' controls multiplication order (default ZYX = yaw->pitch->roll).

    NOTE: If your triangulated points are in OpenCV optical axes (x right, y down, z forward),
    you may still need an extra fixed alignment rotation outside this function.
    """
    if degrees:
        yaw = np.deg2rad(yaw)
        pitch = np.deg2rad(pitch)
        roll = np.deg2rad(roll)

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    Rz = np.array([[cy, -sy, 0.0],
                   [sy,  cy, 0.0],
                   [0.0, 0.0, 1.0]])

    Ry = np.array([[ cp, 0.0,  sp],
                   [0.0, 1.0, 0.0],
                   [-sp, 0.0,  cp]])

    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0,  cr, -sr],
                   [0.0,  sr,  cr]])

    # Build rotation in requested order
    mats = {"X": Rx, "Y": Ry, "Z": Rz}
    if any(ch not in mats for ch in order) or len(order) != 3:
        raise ValueError("order must be a 3-character string containing only X, Y, Z (e.g., 'ZYX').")

    R_BC = mats[order[0]] @ mats[order[1]] @ mats[order[2]]
    t_BC = np.array([x, y, z], dtype=float)

    return R_BC, t_BC