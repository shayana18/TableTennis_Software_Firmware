"""
Stereo Detector for Basler Cameras

Detects ball in both cameras (no 3D triangulation).
Use this for testing detection before triangulation.

CAMERA: Basler acA1920-150uc USB 3.0
        2.3MP, 150fps @ 1920x1200

REQUIREMENTS:
    pip install pypylon
"""

import cv2
import numpy as np

try:
    from pypylon import pylon
    PYPYLON_AVAILABLE = True
except ImportError:
    PYPYLON_AVAILABLE = False

from .ball_tracker import EnhancedBallTracker


class StereoDetector:
    """
    Stereo ball detector for Basler cameras.
    
    Detects ball in both cameras independently.
    Does NOT perform 3D triangulation (use StereoTriangulator for that).
    """

    def __init__(self, left_serial=None, right_serial=None,
                 left_index=0, right_index=1):
        """
        Initialize stereo detector.

        Args:
            left_serial: Left camera serial number (preferred)
            right_serial: Right camera serial number (preferred)
            left_index: Left camera device index (fallback if no serial)
            right_index: Right camera device index (fallback if no serial)
        """
        if not PYPYLON_AVAILABLE:
            raise RuntimeError("pypylon not installed. Run: pip install pypylon")

        self.left_serial = left_serial
        self.right_serial = right_serial
        self.left_index = left_index
        self.right_index = right_index

        # Cameras
        self.cam_left = None
        self.cam_right = None
        self.converter = None

        # Ball trackers (independent thresholds)
        self.tracker_left = EnhancedBallTracker()
        self.tracker_right = EnhancedBallTracker()

    def _get_camera(self, serial=None, device_index=0):
        """Get Basler camera by serial or index."""
        tlFactory = pylon.TlFactory.GetInstance()
        devices = tlFactory.EnumerateDevices()

        if len(devices) == 0:
            raise RuntimeError("No Basler cameras found")

        if serial and serial.strip():
            for device in devices:
                if device.GetSerialNumber() == serial:
                    return pylon.InstantCamera(tlFactory.CreateDevice(device))
            raise RuntimeError(f"Camera with serial '{serial}' not found")

        if device_index >= len(devices):
            raise RuntimeError(f"Device index {device_index} out of range")

        return pylon.InstantCamera(tlFactory.CreateDevice(devices[device_index]))

    def _configure_camera(self, camera, width, height):
        """Configure Basler camera."""
        camera.Open()

        camera.Width.SetValue(width)
        camera.Height.SetValue(height)

        max_w = camera.WidthMax.GetValue()
        max_h = camera.HeightMax.GetValue()
        camera.OffsetX.SetValue((max_w - width) // 2)
        camera.OffsetY.SetValue((max_h - height) // 2)

        camera.ExposureAuto.SetValue("Off")
        camera.ExposureTime.SetValue(3000)
        camera.GainAuto.SetValue("Off")
        camera.Gain.SetValue(0)

    def start_cameras(self, width=1920, height=1200):
        """Open and configure cameras."""
        print("[StereoDetector] Opening cameras...")

        self.cam_left = self._get_camera(
            serial=self.left_serial if self.left_serial else None,
            device_index=self.left_index
        )
        self.cam_right = self._get_camera(
            serial=self.right_serial if self.right_serial else None,
            device_index=self.right_index
        )

        self._configure_camera(self.cam_left, width, height)
        self._configure_camera(self.cam_right, width, height)

        # Image converter
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        print(f"  Left:  {self.cam_left.GetDeviceInfo().GetSerialNumber()}")
        print(f"  Right: {self.cam_right.GetDeviceInfo().GetSerialNumber()}")
        print("[StereoDetector] Cameras ready")

    def stop_cameras(self):
        """Close cameras."""
        if self.cam_left:
            self.cam_left.Close()
        if self.cam_right:
            self.cam_right.Close()
        print("[StereoDetector] Cameras stopped")

    def _grab_frame(self, camera):
        """Grab single frame from camera."""
        if not camera.IsGrabbing():
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grab_result.GrabSucceeded():
            image = self.converter.Convert(grab_result)
            frame = image.GetArray()
            grab_result.Release()
            return frame

        grab_result.Release()
        return None

    def detect(self):
        """
        Capture frames and detect ball in both cameras.

        Returns:
            dict with detection results
        """
        result = {
            'left_frame': None,
            'right_frame': None,
            'left_detection': None,
            'right_detection': None,
            'both_found': False
        }

        # Grab frames
        frame_left = self._grab_frame(self.cam_left)
        frame_right = self._grab_frame(self.cam_right)

        if frame_left is None or frame_right is None:
            return result

        result['left_frame'] = frame_left
        result['right_frame'] = frame_right

        # Detect ball
        result['left_detection'] = self.tracker_left.detect(frame_left)
        result['right_detection'] = self.tracker_right.detect(frame_right)

        result['both_found'] = (
            result['left_detection']['found'] and
            result['right_detection']['found']
        )

        return result

    def load_thresholds(self, filepath):
        """Load ball detection thresholds from JSON file."""
        import json
        try:
            with open(filepath, 'r') as f:
                thresholds = json.load(f)

            if 'left' in thresholds and 'right' in thresholds:
                self.tracker_left.set_hsv_thresholds(
                    thresholds['left']['hsv_lower'], thresholds['left']['hsv_upper'])
                self.tracker_left.set_lab_thresholds(
                    thresholds['left']['lab_lower'], thresholds['left']['lab_upper'])

                self.tracker_right.set_hsv_thresholds(
                    thresholds['right']['hsv_lower'], thresholds['right']['hsv_upper'])
                self.tracker_right.set_lab_thresholds(
                    thresholds['right']['lab_lower'], thresholds['right']['lab_upper'])

                print(f"[StereoDetector] Loaded STEREO thresholds")
            else:
                if 'hsv_lower' in thresholds:
                    self.tracker_left.set_hsv_thresholds(
                        thresholds['hsv_lower'], thresholds['hsv_upper'])
                    self.tracker_right.set_hsv_thresholds(
                        thresholds['hsv_lower'], thresholds['hsv_upper'])

                if 'lab_lower' in thresholds:
                    self.tracker_left.set_lab_thresholds(
                        thresholds['lab_lower'], thresholds['lab_upper'])
                    self.tracker_right.set_lab_thresholds(
                        thresholds['lab_lower'], thresholds['lab_upper'])

                print(f"[StereoDetector] Loaded single thresholds")
        except Exception as e:
            print(f"[StereoDetector] Warning: Could not load thresholds: {e}")

    def set_hsv_thresholds(self, lower, upper):
        """Update HSV thresholds for both trackers."""
        self.tracker_left.set_hsv_thresholds(lower, upper)
        self.tracker_right.set_hsv_thresholds(lower, upper)

    def set_lab_thresholds(self, lower, upper):
        """Update LAB thresholds for both trackers."""
        self.tracker_left.set_lab_thresholds(lower, upper)
        self.tracker_right.set_lab_thresholds(lower, upper)