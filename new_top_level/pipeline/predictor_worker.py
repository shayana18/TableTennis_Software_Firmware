from __future__ import annotations

import queue
import threading

from estimation.robot_predictor import RobotPredictor

from .messages import (
    InterceptPlan,
    PositionSample,
    PredictorCommand,
    PredictorCommandKind,
    PredictorUpdate,
)


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


class PredictorWorker(threading.Thread):
    """Trajectory planner consumer.

    Owns:
    - Temporal acceptance (dt/gap/jump/speed) via RobotPredictor
    - Bounce-aware velocity + intercept planning
    """

    def __init__(
        self,
        *,
        position_queue: queue.Queue,
        predictor_cmd_queue: queue.Queue,
        predictor_update_queue: queue.Queue,
        stop_event: threading.Event,
        status_printer=print,
    ) -> None:
        super().__init__(name="PredictorWorker", daemon=True)
        self._position_queue = position_queue
        self._cmd_queue = predictor_cmd_queue
        self._update_queue = predictor_update_queue
        self._stop_event = stop_event
        self._print = status_printer
        self._predictor = RobotPredictor()
        self._enabled = False

    def _drain_commands(self) -> None:
        # Drain all pending control commands each loop so state changes
        # (enable/disable/reset) are applied promptly and deterministically.
        while True:
            try:
                cmd: PredictorCommand = self._cmd_queue.get_nowait()
            except queue.Empty:
                return

            if cmd.kind == PredictorCommandKind.ENABLE:
                self._enabled = True
            elif cmd.kind == PredictorCommandKind.DISABLE:
                self._enabled = False
            elif cmd.kind == PredictorCommandKind.RESET:
                self._predictor.reset()
            elif cmd.kind == PredictorCommandKind.SHUTDOWN:
                self._stop_event.set()
                return

    def run(self) -> None:
        try:
            while not self._stop_event.is_set():
                self._drain_commands()
                try:
                    sample: PositionSample = self._position_queue.get(timeout=0.02)
                except queue.Empty:
                    continue

                if not self._enabled:
                    # Gate-off mode: consume but ignore queued samples so stale
                    # positions do not replay when gameplay resumes.
                    continue

                # RobotPredictor is unchanged and remains the sole source of
                # trajectory tuning/physics/state acceptance logic.
                accepted = self._predictor.add_position(
                    sample.x_mm, sample.y_mm, sample.z_mm, sample.capture_time
                )

                stats = self._predictor.get_stats()
                vel = self._predictor.velocity
                intercept_plan = None

                if self._predictor.is_ready():
                    # Keep the same prediction trigger conditions used in legacy
                    # integration: only predict once velocity/history is ready.
                    intercept = self._predictor.predict_intercept()
                    if intercept is not None:
                        intercept_plan = InterceptPlan(
                            source_frame_id=sample.frame_id,
                            source_capture_time=sample.capture_time,
                            x_mm=float(intercept["x"]),
                            y_mm=float(intercept["y"]),
                            z_mm=float(intercept["z"]),
                            vx_mm_s=float(intercept.get("vx", 0.0)),
                            vy_mm_s=float(intercept.get("vy", 0.0)),
                            vz_mm_s=float(intercept.get("vz", 0.0)),
                            intercept_time_s=float(intercept["time"]),
                            confidence=float(intercept.get("confidence", 1.0)),
                            clamped=bool(intercept.get("clamped", False)),
                            buffer_points=int(stats.get("buffer", 0)),
                            bounce_count=int(stats.get("bounces", 0)),
                        )

                update = PredictorUpdate(
                    sample=sample,
                    accepted=bool(accepted),
                    reject_reason=self._predictor._last_reject_reason,
                    buffer_points=int(stats.get("buffer", 0)),
                    has_velocity=bool(stats.get("has_vel", False)),
                    ready=bool(self._predictor.is_ready()),
                    bounce_count=int(stats.get("bounces", 0)),
                    velocity=tuple(vel) if vel is not None else None,
                    intercept=intercept_plan,
                )
                _push_latest(self._update_queue, update)
        except Exception as exc:
            self._print(f"[predictor] ERROR: {exc}")
