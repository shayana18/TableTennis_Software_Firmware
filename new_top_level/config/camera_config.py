from __future__ import annotations

from pathlib import Path

import cv2
import yaml


PROJECT_DIR = Path(__file__).resolve().parent.parent
SETTINGS_FILE = PROJECT_DIR / "camera_params" / "calibration_settings.yaml"
DEFAULT_CAMERA_FPS = 100


def load_camera_settings(settings_path=SETTINGS_FILE):
    settings_path = Path(settings_path)
    with settings_path.open("r", encoding="utf-8") as infile:
        settings = yaml.safe_load(infile)

    required = ("camera0", "camera1", "frame_width", "frame_height")
    missing = [key for key in required if key not in settings]
    if missing:
        raise KeyError(f"Missing keys in {settings_path}: {missing}")

    return settings


_SETTINGS = load_camera_settings()
CAMERA_LEFT_ID = _SETTINGS["camera0"]
CAMERA_RIGHT_ID = _SETTINGS["camera1"]
FRAME_WIDTH = _SETTINGS["frame_width"]
FRAME_HEIGHT = _SETTINGS["frame_height"]
CAMERA_FPS = _SETTINGS.get("camera_fps", DEFAULT_CAMERA_FPS)


def _fourcc_to_str(value):
    value = int(value)
    chars = [(value >> (8 * i)) & 0xFF for i in range(4)]
    return "".join(chr(c) if 32 <= c <= 126 else "." for c in chars)


def configure_camera(cap, width=FRAME_WIDTH, height=FRAME_HEIGHT, fps=CAMERA_FPS):
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    width_rb = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_rb = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_rb = float(cap.get(cv2.CAP_PROP_FPS))
    fourcc_rb = _fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))

    settings_match = (
        width_rb == int(width)
        and height_rb == int(height)
        and fourcc_rb == "MJPG"
    )

    return {
        "width": width_rb,
        "height": height_rb,
        "fps": fps_rb,
        "fourcc": fourcc_rb,
        "settings_match": settings_match,
        "trigger_mode": False,
        "trigger_ok": False,
    }
