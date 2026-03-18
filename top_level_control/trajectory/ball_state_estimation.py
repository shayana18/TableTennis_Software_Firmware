"""
3D robot-frame ball state estimation.

Use:
    - call estimate(px, py, pz, timestamp_s) for each new measurement
    - read position/velocity with get_position() / get_velocity()

Future improvements:
    - adaptive R from reprojection error
    - predict-only coast mode
    - drag-aware EKF/UKF
    - covariance-based readiness checks
"""

from __future__ import annotations

import numpy as np

try:
    from filterpy.kalman import KalmanFilter
except ImportError as exc:  # pragma: no cover - depends on local env
    KalmanFilter = None
    _FILTERPY_IMPORT_ERROR = exc
else:
    _FILTERPY_IMPORT_ERROR = None


class BallStateEstimator3D:
    """Estimate 3D ball position and velocity from timestamped positions."""

    def __init__(
        self,
        gravity_z: float,
        accel_std: float = 4000.0,
        meas_std_xy: float = 15.0,
        meas_std_z: float = 20.0,
        max_gap_s: float = 0.12,
        min_updates: int = 6,
    ) -> None:
        """
        Create the estimator.

        Args:
            gravity_z: Known vertical acceleration in robot frame, mm/s^2.
            accel_std: Process acceleration std, mm/s^2.
            meas_std_xy: Measurement std for x/y, mm.
            meas_std_z: Measurement std for z, mm.
            max_gap_s: Reinitialize after larger time gaps.
            min_updates: Updates required before is_ready() is true.
        """
        if KalmanFilter is None:  # pragma: no cover - depends on local env
            raise ImportError(
                "filterpy is required for BallStateEstimator3D. "
                "Install dependencies from top_level_control/requirements.txt."
            ) from _FILTERPY_IMPORT_ERROR

        self.gravity_z = float(gravity_z)
        self.accel_std = float(accel_std)
        self.meas_std_xy = float(meas_std_xy)
        self.meas_std_z = float(meas_std_z)
        self.max_gap_s = float(max_gap_s)
        self.min_updates = int(min_updates)

        self.kf = KalmanFilter(dim_x=6, dim_z=3, dim_u=1)
        self.kf.x = np.zeros((6, 1), dtype=float)
        self.kf.H = self._build_H()
        # Start conservative; tune R/Q from replayed throws before live hitting.
        self.kf.R = self._build_R(self.meas_std_xy, self.meas_std_z)
        self.kf.P = self._build_initial_P()

        self.initialized = False
        self.last_timestamp_s: float | None = None
        self.update_count = 0

    @staticmethod
    def _compute_dt(current_timestamp_s: float, previous_timestamp_s: float) -> float:
        """Return dt between consecutive timestamped measurements."""
        return float(current_timestamp_s) - float(previous_timestamp_s)

    @staticmethod
    def _build_F(dt: float) -> np.ndarray:
        """Build the state transition matrix."""
        return np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1],
        ], dtype=float)

    @staticmethod
    def _build_B(dt: float) -> np.ndarray:
        """Build the gravity input matrix."""
        return np.array([
            [0.0],
            [0.0],
            [0.5 * dt * dt],
            [0.0],
            [0.0],
            [dt],
        ], dtype=float)

    @staticmethod
    def _build_H() -> np.ndarray:
        """Build the measurement matrix."""
        return np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ], dtype=float)

    @staticmethod
    def _build_R(meas_std_xy: float, meas_std_z: float) -> np.ndarray:
        """Build the measurement covariance."""
        # Larger R trusts triangulation less and smooths harder.
        return np.diag([
            float(meas_std_xy) ** 2,
            float(meas_std_xy) ** 2,
            float(meas_std_z) ** 2,
        ])

    @staticmethod
    def _build_initial_P() -> np.ndarray:
        """Build the initial covariance."""
        return np.diag([
            50.0 ** 2,
            50.0 ** 2,
            50.0 ** 2,
            2000.0 ** 2,
            2000.0 ** 2,
            2000.0 ** 2,
        ])

    @staticmethod
    def _build_Q(dt: float, accel_std: float) -> np.ndarray:
        """Build the process covariance."""
        # Larger Q reacts faster but will chase noisy measurements sooner.
        q = float(accel_std) ** 2
        q_axis = np.array([
            [0.25 * dt**4, 0.5 * dt**3],
            [0.5 * dt**3,       dt**2],
        ], dtype=float) * q

        Q = np.zeros((6, 6), dtype=float)
        Q[np.ix_([0, 3], [0, 3])] = q_axis
        Q[np.ix_([1, 4], [1, 4])] = q_axis
        Q[np.ix_([2, 5], [2, 5])] = q_axis
        return Q

    @staticmethod
    def _predict_step(kf: KalmanFilter, gravity_z: float) -> None:
        """Run the predict step."""
        u = np.array([[float(gravity_z)]], dtype=float)
        kf.predict(u=u)

    @staticmethod
    def _update_step(kf: KalmanFilter, px: float, py: float, pz: float) -> None:
        """Run the measurement update."""
        z = np.array([[px], [py], [pz]], dtype=float)
        kf.update(z)

    @staticmethod
    def _state_tuple(kf: KalmanFilter) -> tuple[float, float, float, float, float, float]:
        """Return the filter state as a flat tuple."""
        s = kf.x.reshape(-1)
        return (
            float(s[0]), float(s[1]), float(s[2]),
            float(s[3]), float(s[4]), float(s[5]),
        )

    def reset(self) -> None:
        """Reset the estimator state."""
        self.kf.x[:] = 0.0
        self.kf.H = self._build_H()
        self.kf.R = self._build_R(self.meas_std_xy, self.meas_std_z)
        self.kf.P = self._build_initial_P()
        self.initialized = False
        self.last_timestamp_s = None
        self.update_count = 0

    def initialize_from_measurement(
        self,
        px: float,
        py: float,
        pz: float,
        timestamp_s: float,
    ) -> tuple[float, float, float, float, float, float]:
        """
        Initialize from the first measurement.

        Args:
            px, py, pz: Robot-frame position in mm.
            timestamp_s: Measurement timestamp in seconds.

        Returns:
            The initialized state tuple.
        """
        self.kf.x = np.array([
            [px],
            [py],
            [pz],
            [0.0],
            [0.0],
            [0.0],
        ], dtype=float)
        self.initialized = True
        self.last_timestamp_s = float(timestamp_s)
        self.update_count = 1
        return self.get_state()

    def estimate(
        self,
        px: float,
        py: float,
        pz: float,
        timestamp_s: float,
    ) -> tuple[bool, tuple[float, float, float, float, float, float] | None]:
        """
        Run one full predict/update cycle.

        Args:
            px, py, pz: Robot-frame position in mm.
            timestamp_s: Timestamp in seconds.

        Returns:
            (accepted, state) after the update.
        """
        if not self.initialized:
            return True, self.initialize_from_measurement(px, py, pz, timestamp_s)

        dt = self._compute_dt(timestamp_s, self.last_timestamp_s)
        if dt <= 0.0:
            return False, self.get_state()

        if dt > self.max_gap_s:
            return True, self.initialize_from_measurement(px, py, pz, timestamp_s)

        self.kf.F = self._build_F(dt)
        self.kf.B = self._build_B(dt)
        self.kf.Q = self._build_Q(dt, self.accel_std)

        self._predict_step(self.kf, self.gravity_z)
        self._update_step(self.kf, px, py, pz)

        self.last_timestamp_s = float(timestamp_s)
        self.update_count += 1
        return True, self.get_state()

    def get_state(self) -> tuple[float, float, float, float, float, float] | None:
        """Return (px, py, pz, vx, vy, vz) or None."""
        if not self.initialized:
            return None
        return self._state_tuple(self.kf)

    def get_position(self) -> tuple[float, float, float] | None:
        """Return (px, py, pz) or None."""
        state = self.get_state()
        if state is None:
            return None
        return state[0], state[1], state[2]

    def get_velocity(self) -> tuple[float, float, float] | None:
        """Return (vx, vy, vz) or None."""
        state = self.get_state()
        if state is None:
            return None
        return state[3], state[4], state[5]

    def get_covariance(self) -> np.ndarray:
        """Return a copy of the covariance matrix."""
        return self.kf.P.copy()

    def is_ready(self) -> bool:
        """Return True when enough updates have been fused."""
        return self.initialized and self.update_count >= self.min_updates
