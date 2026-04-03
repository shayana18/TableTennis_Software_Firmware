"""
Shared camera configuration for Arducam OV9782 global shutter cameras.

>>> SINGLE SOURCE OF TRUTH <<<
Change camera parameters HERE and they auto-propagate to every script
(calibration, detection, triangulation, trigger verify, etc.)

Trigger mode is set manually via ArduCam AmCap app — NOT through software.
"""

import cv2
import time

# ====================================================================
#  CAMERA SETTINGS
# ====================================================================
CAMERA_LEFT_ID  = 2     # USB device index for left camera  (camersd)
CAMERA_RIGHT_ID = 1       # USB device index for right camera (camera1)
FRAME_WIDTH     = 640     # Horizontal resolution
FRAME_HEIGHT    = 480     # Vertical resolution
FPS             = 100     # Target framerate
FOURCC          = 'MJPG'  # Video codec (MJPG required for full fps)
# ====================================================================


def load_camera_settings():
    """
    Return camera settings as a dict.

    Kept for backward compatibility — all values come from the
    module-level constants above.
    """
    return {
        'camera0': CAMERA_LEFT_ID,
        'camera1': CAMERA_RIGHT_ID,
        'frame_width': FRAME_WIDTH,
        'frame_height': FRAME_HEIGHT,
    }


def configure_camera(cap, width=None, height=None):
    """
    Configure an Arducam OV9782 global shutter camera.

    Sets codec, resolution, and fps only.
    Trigger mode is set manually via ArduCam AmCap app before running.

    Args:
        cap: cv2.VideoCapture object (already opened)
        width:  Frame width  (default: FRAME_WIDTH)
        height: Frame height (default: FRAME_HEIGHT)

    Returns:
        dict with actual accepted camera settings
    """
    if width is None:
        width = FRAME_WIDTH
    if height is None:
        height = FRAME_HEIGHT

    # --- Codec, resolution, framerate ---
    fourcc_mjpg = cv2.VideoWriter_fourcc(*FOURCC)
    cap.set(cv2.CAP_PROP_FOURCC, fourcc_mjpg)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    # --- Read back actual values ---
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    actual_fourcc_str = "".join(
        [chr((actual_fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

    return {
        'width': actual_width,
        'height': actual_height,
        'fps': actual_fps,
        'fourcc': actual_fourcc_str,
        'settings_match': (actual_width == width and actual_height == height),
    }
