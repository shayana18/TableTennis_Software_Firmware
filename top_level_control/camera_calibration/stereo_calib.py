import glob
import os
import sys

import numpy as np
import cv2 as cv

from calib import (
    calibration_settings,
    parse_calibration_settings_file,
    save_frames_two_cams,
    stereo_calibrate,
)


def _clear_frames_pair(folder="frames_pair"):
    """
    Remove existing paired calibration frames so the directory can be repopulated.
    Only deletes files directly inside the folder.
    """
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        return

    for entry in os.listdir(folder):
        path = os.path.join(folder, entry)
        if os.path.isfile(path):
            os.remove(path)

def load_camera_intrinsics(camera_name, folder="camera_parameters"):
    """
    Load intrinsic matrix and distortion coefficients from a *_intrinsics.dat file.
    The file includes an RMSE section that is intentionally ignored.
    """
    path = os.path.join(folder, f"{camera_name}_intrinsics.dat")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Intrinsic file not found: {path}")

    intr_rows = []
    dist_vals = []
    mode = None

    with open(path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.lower().startswith("intrinsic"):
                mode = "intrinsic"
                continue
            if line.lower().startswith("distortion"):
                mode = "distortion"
                continue
            if line.lower().startswith("reprojection error"):
                # RMSE is present in the file but not needed for calibration.
                mode = None
                continue

            if mode == "intrinsic":
                intr_rows.append([float(v) for v in line.split()])
            elif mode == "distortion":
                dist_vals.extend([float(v) for v in line.split()])

    if len(intr_rows) != 3:
        raise ValueError(f"Expected 3 rows for intrinsic matrix in {path}, got {len(intr_rows)}")
    if not dist_vals:
        raise ValueError(f"No distortion coefficients found in {path}")

    cmtx = np.array(intr_rows, dtype=np.float32)
    dist = np.array(dist_vals, dtype=np.float32).reshape(1, -1)
    return cmtx, dist


def main(settings_path):
    parse_calibration_settings_file(settings_path)

    cmtx0, dist0 = load_camera_intrinsics("camera0")
    cmtx1, dist1 = load_camera_intrinsics("camera1")

    print("Loaded intrinsics:")
    print("camera0 matrix:\n", cmtx0)
    print("camera0 distortion:", dist0)
    print("camera1 matrix:\n", cmtx1)
    print("camera1 distortion:", dist1)

    frames_prefix_c0 = os.path.join("frames_pair", "camera0*")
    frames_prefix_c1 = os.path.join("frames_pair", "camera1*")

    print("Clearing existing paired frames (if any) and capturing new stereo calibration frames...")
    _clear_frames_pair("frames_pair")
    save_frames_two_cams("camera0", "camera1")

    R, T = stereo_calibrate(cmtx0, dist0, cmtx1, dist1, frames_prefix_c0, frames_prefix_c1)

    print("Stereo calibration complete.")
    print("Rotation (R):\n", R)
    print("Translation (T):\n", T)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Call with settings filename: "python3 stereo_calib.py calibration_settings.yaml"')
        sys.exit(1)

    main(sys.argv[1])
