"""
Basler Stereo Viewer (Standalone)

- Opens 2 Basler pylon cameras (USB3 / GigE supported by pylon)
- Auto-detects cameras and assigns Left/Right by sorted serial number
- Displays frames side-by-side using OpenCV
- No detection, no configs, no extra project imports

Controls:
  q  - quit
"""

import cv2
import numpy as np
from pypylon import pylon


def _make_converter():
    conv = pylon.ImageFormatConverter()
    conv.OutputPixelFormat = pylon.PixelType_BGR8packed
    conv.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    return conv


def _safe_set(cam, attr: str, value):
    """
    Best-effort set of a pylon node if it exists/writable.
    attr is like 'ExposureAuto', 'ExposureTime', 'GainAuto', etc.
    """
    try:
        node = getattr(cam, attr)
        node.SetValue(value)
        return True
    except Exception:
        return False


def _configure_camera(cam, width=1920, height=1200, fps=90, exposure_us=1500.0):
    # Turn off autos where possible for stable viewing (optional)
    _safe_set(cam, "ExposureAuto", "Off")
    _safe_set(cam, "GainAuto", "Off")
    _safe_set(cam, "BalanceWhiteAuto", "Off")

    # Exposure (microseconds)
    _safe_set(cam, "ExposureTime", float(exposure_us))

    # FPS (may fail if camera doesn't allow it in current mode)
    try:
        cam.AcquisitionFrameRateEnable.SetValue(True)
        cam.AcquisitionFrameRate.SetValue(float(fps))
    except Exception:
        pass

    # Set ROI / size (may need to respect increments; if it fails, we just keep defaults)
    try:
        # Some cameras require offsets to be set to 0 before changing width/height
        if hasattr(cam, "OffsetX"):
            cam.OffsetX.SetValue(0)
        if hasattr(cam, "OffsetY"):
            cam.OffsetY.SetValue(0)

        if hasattr(cam, "Width") and hasattr(cam, "Height"):
            cam.Width.SetValue(int(width))
            cam.Height.SetValue(int(height))
    except Exception:
        pass

    # Pixel format: we convert to BGR in software anyway, so leave camera format alone
    # unless you know you want a specific format. Many color Baslers default to Bayer.
    # The ImageFormatConverter handles this conversion to BGR8packed.


def open_two_cameras_by_serial():
    tl = pylon.TlFactory.GetInstance()
    devices = tl.EnumerateDevices()

    if len(devices) < 2:
        raise RuntimeError(f"Need at least 2 Basler cameras. Found {len(devices)}.")

    # Sort devices by serial number for consistent L/R assignment
    dev_infos = []
    for dev in devices:
        try:
            serial = dev.GetSerialNumber()
        except Exception:
            serial = ""
        try:
            model = dev.GetModelName()
        except Exception:
            model = "UnknownModel"
        dev_infos.append((serial, model, dev))

    dev_infos.sort(key=lambda x: x[0])  # sort by serial string

    left_serial, left_model, left_dev = dev_infos[0]
    right_serial, right_model, right_dev = dev_infos[1]

    camL = pylon.InstantCamera(tl.CreateDevice(left_dev))
    camR = pylon.InstantCamera(tl.CreateDevice(right_dev))

    camL.Open()
    camR.Open()

    print("\nDetected Basler cameras (sorted by serial):")
    print(f"  LEFT : {left_model}  Serial={left_serial}")
    print(f"  RIGHT: {right_model}  Serial={right_serial}")

    return camL, camR, (left_serial, right_serial)


def main():
    camL, camR, serials = open_two_cameras_by_serial()

    # Optional tuning for smooth viewing (adjust if you want)
    _configure_camera(camL, width=1920, height=1200, fps=90.0, exposure_us=1500.0)
    _configure_camera(camR, width=1920, height=1200, fps=120.0, exposure_us=1500.0)

    convL = _make_converter()
    convR = _make_converter()

    # Lowest latency strategy: always keep the newest frame
    camL.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    camR.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    cv2.namedWindow("Basler Stereo (L | R)", cv2.WINDOW_NORMAL)

    try:
        while camL.IsGrabbing() and camR.IsGrabbing():
            # Retrieve latest frames
            gL = camL.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)
            gR = camR.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)

            ok = False
            if gL.GrabSucceeded() and gR.GrabSucceeded():
                frameL = convL.Convert(gL).GetArray()
                frameR = convR.Convert(gR).GetArray()
                ok = True

            gL.Release()
            gR.Release()

            if not ok:
                continue

            # Ensure same height for side-by-side
            h = min(frameL.shape[0], frameR.shape[0])
            frameL = frameL[:h, :]
            frameR = frameR[:h, :]

            # Add labels
            cv2.putText(frameL, f"LEFT  Serial: {serials[0]}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frameR, f"RIGHT Serial: {serials[1]}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            vis = np.hstack([frameL, frameR])
            cv2.imshow("Basler Stereo (L | R)", vis)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    finally:
        camL.StopGrabbing()
        camR.StopGrabbing()
        camL.Close()
        camR.Close()
        cv2.destroyAllWindows()
        print("\nStopped.")


if __name__ == "__main__":
    main()
