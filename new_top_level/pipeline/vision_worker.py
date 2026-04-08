from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

from comm_functions.points_based_transform import cam_to_robot

from .messages import FrameObservation, PositionSample


def _push_latest(q: queue.Queue, item) -> None:
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass


@dataclass(frozen=True)
class VisionConfig:
    frame_width: int
    frame_height: int
    reproj_err_max_px: float = 10.0
    warmup_s: float = 2.0


class VisionWorker(threading.Thread):
    """Stereo triangulation producer.

    Owns:
    - Camera start/stop
    - Background warmup
    - Stereo quality filtering
    - Camera->robot transform publishing
    """

    def __init__(
        self,
        *,
        triangulator,
        transform_rotation,
        transform_translation,
        transform_scale: float,
        position_queue: queue.Queue,
        observation_queue: queue.Queue,
        stop_event: threading.Event,
        config: VisionConfig,
        status_printer=print,
    ) -> None:
        super().__init__(name="VisionWorker", daemon=True)
        self._triangulator = triangulator
        self._R = transform_rotation
        self._t = transform_translation
        self._scale = float(transform_scale)
        self._position_queue = position_queue
        self._observation_queue = observation_queue
        self._stop_event = stop_event
        self._cfg = config
        self._print = status_printer
        self._frame_id = 0

    def _warmup_background(self) -> bool:
        # Same operational idea as integration_simple:
        # let MOG2 learn background before enabling live detection.
        if self._cfg.warmup_s <= 0.0:
            return True
        self._print("[vision] Learning background...")
        t0 = time.perf_counter()
        while (time.perf_counter() - t0) < self._cfg.warmup_s and not self._stop_event.is_set():
            if not self._triangulator.cap_left.grab():
                continue
            if not self._triangulator.cap_right.grab():
                continue
            ok_l, frame_l = self._triangulator.cap_left.retrieve()
            ok_r, frame_r = self._triangulator.cap_right.retrieve()
            if not ok_l or not ok_r or frame_l is None or frame_r is None:
                continue
            self._triangulator.build_background(frame_l, frame_r)
        self._print("[vision] Background ready.")
        return not self._stop_event.is_set()

    def run(self) -> None:
        try:
            # Cameras are owned by this worker.
            self._triangulator.start_cameras(self._cfg.frame_width, self._cfg.frame_height)
            if not self._warmup_background():
                return

            while not self._stop_event.is_set():
                # StereoTriangulator internally uses grab/retrieve sync and
                # produces a shared capture timestamp for both cameras.
                result = self._triangulator.update()
                if result.get("left_frame") is None:
                    continue

                self._frame_id += 1
                frame_id = self._frame_id
                capture_time = float(result.get("capture_time", time.perf_counter()))
                found_3d = bool(result.get("found_3d", False))
                reproj = result.get("reproj_err")
                reject_reason = result.get("reject_reason")
                disparity = result.get("disparity")
                cam_pos = result.get("position_3d")

                if found_3d and reproj is not None and reproj > self._cfg.reproj_err_max_px:
                    # Preserve integration_simple tuning: reject if reproj > 10 px.
                    found_3d = False
                    reject_reason = f"reproj({reproj:.1f}px)"

                observation = FrameObservation(
                    frame_id=frame_id,
                    capture_time=capture_time,
                    left_frame=result["left_frame"],
                    found_3d=found_3d,
                    position_3d_cam=tuple(cam_pos) if cam_pos is not None else None,
                    reject_reason=reject_reason,
                    reproj_err_px=float(reproj) if reproj is not None else None,
                    disparity_px=float(disparity) if disparity is not None else None,
                )
                _push_latest(self._observation_queue, observation)

                if not found_3d or cam_pos is None:
                    continue

                # Camera-frame (calibration units) -> robot-frame (mm).
                rx, ry, rz = cam_to_robot(
                    self._R,
                    self._t,
                    self._scale,
                    float(cam_pos[0]),
                    float(cam_pos[1]),
                    float(cam_pos[2]),
                )
                sample = PositionSample(
                    frame_id=frame_id,
                    capture_time=capture_time,
                    x_mm=rx,
                    y_mm=ry,
                    z_mm=rz,
                    reproj_err_px=float(reproj) if reproj is not None else 0.0,
                    disparity_px=float(disparity) if disparity is not None else 0.0,
                )
                _push_latest(self._position_queue, sample)
        except Exception as exc:
            self._print(f"[vision] ERROR: {exc}")
        finally:
            try:
                self._triangulator.stop_cameras()
            except Exception:
                pass
