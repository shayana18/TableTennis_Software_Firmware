from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class UartCommandKind(Enum):
    """Commands produced by the game state machine and consumed by UartWorker."""

    HOME = auto()
    INTERCEPT = auto()
    SHUTDOWN = auto()


class PredictorCommandKind(Enum):
    """Control messages for PredictorWorker lifecycle and gating."""

    ENABLE = auto()
    DISABLE = auto()
    RESET = auto()
    SHUTDOWN = auto()


class UartEventKind(Enum):
    """Parsed robot status and local UART lifecycle events."""

    UART_OPENED = auto()
    UART_CLOSED = auto()
    UART_ERROR = auto()
    RAW_LINE = auto()
    COMPLETED_Q = auto()
    STATE_OFF = auto()
    STATE_IDLE = auto()
    STATE_PLAN = auto()
    STATE_MOVE = auto()
    HOME_DONE = auto()
    INTERCEPT_DONE = auto()
    TX_HOME = auto()
    TX_INTERCEPT = auto()
    TARGET_REJECTED = auto()
    PLAN_FAILED = auto()
    ROBOT_LATE = auto()
    FAULT = auto()


@dataclass(frozen=True)
class PositionSample:
    """A single accepted stereo observation expressed in robot frame (mm)."""

    frame_id: int
    capture_time: float
    x_mm: float
    y_mm: float
    z_mm: float
    reproj_err_px: float
    disparity_px: float


@dataclass(frozen=True)
class FrameObservation:
    """Visualization/debug payload from vision thread."""

    frame_id: int
    capture_time: float
    left_frame: object
    found_3d: bool
    position_3d_cam: Optional[tuple[float, float, float]]
    reject_reason: Optional[str]
    reproj_err_px: Optional[float]
    disparity_px: Optional[float]


@dataclass(frozen=True)
class InterceptPlan:
    """Prediction output that is eligible to be transmitted over UART."""

    source_frame_id: int
    source_capture_time: float
    x_mm: float
    y_mm: float
    z_mm: float
    vx_mm_s: float
    vy_mm_s: float
    vz_mm_s: float
    intercept_time_s: float
    confidence: float
    clamped: bool
    buffer_points: int
    bounce_count: int


@dataclass(frozen=True)
class PredictorUpdate:
    """Per-sample predictor diagnostics for UI/logging/state decisions."""

    sample: PositionSample
    accepted: bool
    reject_reason: Optional[str]
    buffer_points: int
    has_velocity: bool
    ready: bool
    bounce_count: int
    velocity: Optional[tuple[float, float, float]]
    intercept: Optional[InterceptPlan]


@dataclass(frozen=True)
class UartCommand:
    """Command envelope sent to UartWorker."""

    kind: UartCommandKind
    intercept: Optional[InterceptPlan] = None


@dataclass(frozen=True)
class PredictorCommand:
    """Command envelope sent to PredictorWorker."""

    kind: PredictorCommandKind


@dataclass(frozen=True)
class UartEvent:
    """Event emitted by UartWorker to drive game state transitions."""

    kind: UartEventKind
    timestamp: float
    line: Optional[str] = None
    data: dict = field(default_factory=dict)
