"""
Shared camera configuration for Arducam OV9782 global shutter cameras.

>>> SINGLE SOURCE OF TRUTH <<<
Change camera parameters HERE and they auto-propagate to every script
(calibration, detection, triangulation, trigger verify, etc.)
"""

import cv2
import time
     
# ====================================================================
#  CAMERA SETTINGS — MAINNNNN
# ====================================================================
CAMERA_LEFT_ID  = 2      # USB device index for left camera  (camera0)
CAMERA_RIGHT_ID = 1       # USB device index for right camera (camera1)
FRAME_WIDTH     = 640    # Horizontal resolution
FRAME_HEIGHT    = 480  # Vertical resolution
FPS             = 100     # Target framerate
FOURCC          = 'MJPG'  # Video codec (MJPG required for full fps)
TRIGGER_MODE    = True    # Hardware sync via external trigger signal
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
        'trigger_mode': TRIGGER_MODE,
    }


def configure_camera(cap, width=None, height=None, trigger_mode=None):
    """
    Configure an Arducam OV9782 global shutter camera.

    Sets codec, resolution, fps, exposure, and (optionally) trigger mode.
    All defaults come from the module-level constants at the top of this file.

    Args:
        cap: cv2.VideoCapture object (already opened)
        width:  Frame width  (default: FRAME_WIDTH)
        height: Frame height (default: FRAME_HEIGHT)
        trigger_mode: Enable external trigger (default: TRIGGER_MODE)

    Returns:
        dict with actual accepted camera settings
    """
    if width is None:
        width = FRAME_WIDTH
    if height is None:
        height = FRAME_HEIGHT
    if trigger_mode is None:
        trigger_mode = TRIGGER_MODE

    # --- Codec, resolution, framerate ---
    fourcc_mjpg = cv2.VideoWriter_fourcc(*FOURCC)
    cap.set(cv2.CAP_PROP_FOURCC, fourcc_mjpg)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    # --- Exposure control (always manual) ---
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

    # --- External trigger mode ---
    trigger_ok = False
    if trigger_mode:
        # Backlight compensation ON = external trigger mode for ArduCam OV9782
        cap.set(cv2.CAP_PROP_BACKLIGHT, 1)
        # Manual exposure (try DirectShow value first, then V4L2)
        ok_ae = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        if not ok_ae:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        # Exposure value controls trigger frame rate
        cap.set(cv2.CAP_PROP_EXPOSURE, -6)
        trigger_ok = (cap.get(cv2.CAP_PROP_BACKLIGHT) == 1)

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
        'trigger_mode': trigger_mode,
        'trigger_ok': trigger_ok if trigger_mode else None,
    }
