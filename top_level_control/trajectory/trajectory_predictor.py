"""
Trajectory Predictor - Main Prediction Class

Predicts where and when the ball will arrive at the robot's end of the table.

GEOMETRY:
    Camera is on the SIDE of the table, looking ALONG the table length.

    Camera frame (cm):
        Z = depth = along table length (ball travels on this axis, 274 cm)
        X = lateral = across table width (152.5 cm)
        Y = vertical (down = positive, standard camera convention)

    Robot frame (mm, origin at robot endline center):
        robot_x = across table width (lateral reach, ±680mm)
        robot_y = height above table surface (0 to 500mm)
        robot_z = from endline toward net (0 to 440mm)

    Ball travels along CAMERA Z → intercept at target camera-Z position.

PIPELINE:
  position_buffer.py → velocity_estimator.py → trajectory_predictor.py
                                                        ↓
                                                 physics_model.py
"""

import time
import math
import numpy as np

from .position_buffer import PositionBuffer
from .velocity_estimator import VelocityEstimator
from .physics_model import PhysicsModel


# ================================================================
# TABLE + ROBOT GEOMETRY (mm)
# ================================================================

TABLE_LENGTH_MM    = 2740.0
TABLE_WIDTH_MM     = 1525.0
TABLE_HALF_LEN_MM  = TABLE_LENGTH_MM / 2.0   # 1370 mm
NET_HEIGHT_MM      = 152.5

ROBOT_REACH_X_MM   = 680.0     # semi-axis lateral
ROBOT_REACH_Z_MM   = 440.0     # semi-axis depth from endline
ROBOT_Y_MAX_MM     = 500.0

CM_TO_MM = 10.0


class TrajectoryPredictor:
    """
    Trajectory prediction with Z-axis interception and robot output.

    The ball travels along camera Z (table length).
    Interception target is a camera-Z position near the robot's endline.
    """

    # --- Outlier rejection ---
    MAX_BALL_SPEED    = 1500.0   # cm/s
    MAX_POSITION_JUMP = 40.0     # cm
    MAX_VELOCITY      = 2000.0   # cm/s

    # --- Prediction readiness ---
    MIN_TIME_SPAN     = 0.035    # seconds (~3 frames at 80fps)

    def __init__(self,
                 buffer_size=15,
                 min_points=4,
                 velocity_method='regression',
                 gravity=981.0,
                 y_down=True,
                 enable_drag=True,
                 robot_z_cam=None):
        """
        Args:
            robot_z_cam: Camera-Z coordinate of robot endline (cm).
                         Ball intercept happens at this Z value.
        """
        self.buffer_size = buffer_size
        self.min_points = min_points

        self.robot_z_cam = robot_z_cam   # camera Z at robot endline (cm)

        # Pipeline
        self.position_buffer = PositionBuffer(max_size=buffer_size)
        self.velocity_estimator = VelocityEstimator(method=velocity_method)
        self.physics_model = PhysicsModel(
            gravity=gravity, y_down=y_down, enable_drag=enable_drag)

        self._current_velocity = None
        self._velocity_valid = False
        self._rejected_count = 0
        self._accepted_count = 0
        self._last_command = None
        self._command_history = []

        # Camera-to-robot transform (set via set_table_calibration)
        self._cam_x_center = None   # camera X at table center-width (cm)
        self._cam_y_table = None    # camera Y at table surface (cm)

    # ================================================================
    # CALIBRATION
    # ================================================================

    def set_robot_endline(self, camera_z):
        """Set the camera-Z coordinate of the robot endline."""
        self.robot_z_cam = camera_z
        print(f"[Predictor] Robot endline at camera Z = {camera_z:.1f} cm")

    def set_table_calibration(self, cam_x_center, cam_y_table):
        """
        Set camera coordinates of table reference points.

        Args:
            cam_x_center: Camera X at center of table width (cm)
            cam_y_table: Camera Y at table surface level (cm)
        """
        self._cam_x_center = cam_x_center
        self._cam_y_table = cam_y_table
        print(f"[Predictor] Table: X_center={cam_x_center:.1f}, "
              f"Y_surface={cam_y_table:.1f} cm")

    # ================================================================
    # OUTLIER REJECTION
    # ================================================================

    def _check_outlier(self, x, y, z, timestamp):
        if len(self.position_buffer) == 0:
            return True, None

        last = self.position_buffer.get_latest()
        dt = timestamp - last['t']
        if dt <= 0:
            return False, "zero_dt"

        dx = x - last['x']
        dy = y - last['y']
        dz = z - last['z']
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)

        if dist > self.MAX_POSITION_JUMP:
            return False, f"jump({dist:.0f})"

        speed = dist / dt
        if speed > self.MAX_BALL_SPEED:
            return False, f"speed({speed:.0f})"

        return True, None

    # ================================================================
    # POSITION INPUT
    # ================================================================

    def add_position(self, x, y, z, timestamp=None):
        """Add 3D position with outlier rejection. Returns True if accepted."""
        if timestamp is None:
            timestamp = time.perf_counter()

        ok, reason = self._check_outlier(x, y, z, timestamp)
        if not ok:
            self._rejected_count += 1
            return False

        self._accepted_count += 1
        self.position_buffer.add(x, y, z, timestamp)

        if self.position_buffer.is_ready(self.min_points):
            if self.position_buffer.get_time_span() >= self.MIN_TIME_SPAN:
                self._update_velocity()

        return True

    def _update_velocity(self):
        vel = self.velocity_estimator.estimate_from_buffer(self.position_buffer)
        if not vel['valid']:
            return
        if vel['speed'] > self.MAX_VELOCITY:
            self._velocity_valid = False
            return
        self._current_velocity = vel
        self._velocity_valid = True

    # ================================================================
    # STATE
    # ================================================================

    def get_velocity(self):
        if self._current_velocity is None:
            return {'vx': 0, 'vy': 0, 'vz': 0, 'speed': 0, 'valid': False}
        return self._current_velocity

    def get_current_position(self):
        latest = self.position_buffer.get_latest()
        if latest is None: return None
        return (latest['x'], latest['y'], latest['z'])

    def is_ready(self):
        return (self.position_buffer.is_ready(self.min_points) and
                self._velocity_valid and
                self.position_buffer.get_time_span() >= self.MIN_TIME_SPAN)

    def _get_corrected_velocity(self):
        """Correct Vy for regression midpoint bias."""
        vel = self.get_velocity()
        if not vel['valid']: return None

        _, timestamps = self.position_buffer.get_as_arrays()
        t_latest = timestamps[-1]
        t_mean = timestamps.mean()

        vy_corrected = vel['vy'] + (
            self.physics_model.gravity_sign *
            self.physics_model.gravity *
            (t_latest - t_mean))

        return (vel['vx'], vy_corrected, vel['vz'])

    # ================================================================
    # PREDICTION — intercept along camera Z (table length)
    # ================================================================

    def predict(self, target_z=None):
        """
        Predict where ball will be when it reaches target_z.

        Args:
            target_z: Camera-Z to intercept (cm). None = use robot_z_cam.

        Returns:
            dict with intercept_x/y/z, time_to_intercept, strategy, etc.
        """
        if target_z is None:
            target_z = self.robot_z_cam

        result = {
            'valid': False,
            'intercept_x': None, 'intercept_y': None, 'intercept_z': None,
            'time_to_intercept': None,
            'velocity_at_intercept': None,
            'current_position': None, 'current_velocity': None,
            'strategy': None
        }

        if target_z is None or not self.is_ready():
            return result

        pos = self.get_current_position()
        vel = self._get_corrected_velocity()
        if pos is None or vel is None:
            return result

        result['current_position'] = pos
        result['current_velocity'] = vel

        vz = vel[2]
        vy = vel[1]
        prediction = None
        strategy = None

        # Is ball moving toward robot? (Vz positive = away from camera = deeper)
        # Robot is at robot_z_cam. Ball needs to reach that Z.
        moving_toward = (target_z - pos[2]) * vz > 0

        # Case 1: Ball rising → predict apex (for lobs)
        if vy < 0 and (not moving_toward or abs(vy) > abs(vz) * 0.5):
            prediction = self.physics_model.position_at_apex(
                position=pos, velocity=vel)
            if prediction['valid']:
                strategy = 'apex'

        # Case 2: Z-plane interception (primary)
        if prediction is None or not prediction['valid']:
            prediction = self.physics_model.position_at_z(
                position=pos, velocity=vel, target_z=target_z)
            if prediction['valid']:
                strategy = 'z_plane'

        if not prediction or not prediction['valid']:
            return result

        result['valid'] = True
        result['intercept_x'] = prediction['position'][0]
        result['intercept_y'] = prediction['position'][1]
        result['intercept_z'] = prediction['position'][2]
        result['time_to_intercept'] = prediction['time']
        result['velocity_at_intercept'] = prediction['velocity']
        result['strategy'] = strategy
        return result

    # ================================================================
    # ROBOT COMMAND
    # ================================================================

    def get_robot_command(self, target_z=None):
        """
        Get command for robot controller.

        Returns both camera coords (cm) and robot coords (mm).

        Robot frame (mm):
            robot_x: lateral across table width (from center)
            robot_y: height above table surface
            robot_z: depth from robot endline toward net
        """
        cmd = {
            'valid': False,
            'cam_x': 0.0, 'cam_y': 0.0, 'cam_z': 0.0,
            'robot_x': 0.0, 'robot_y': 0.0, 'robot_z': 0.0,
            'in_workspace': False,
            't': 0.0,
            'strategy': None, 'confidence': 0.0,
            'buffer_points': len(self.position_buffer),
            'time_span': self.position_buffer.get_time_span()
        }

        pred = self.predict(target_z)
        if not pred['valid']:
            return cmd

        cmd['valid'] = True
        cmd['cam_x'] = pred['intercept_x']
        cmd['cam_y'] = pred['intercept_y']
        cmd['cam_z'] = pred['intercept_z']
        cmd['t'] = pred['time_to_intercept']
        cmd['strategy'] = pred['strategy']
        cmd['confidence'] = self._compute_confidence()

        rx, ry, rz = self.cam_to_robot(
            pred['intercept_x'], pred['intercept_y'], pred['intercept_z'])
        cmd['robot_x'] = rx
        cmd['robot_y'] = ry
        cmd['robot_z'] = rz
        cmd['in_workspace'] = self.check_workspace(rx, ry, rz)

        self._last_command = cmd
        self._command_history.append(
            (cmd['cam_x'], cmd['cam_y'], cmd['cam_z'], cmd['t']))
        if len(self._command_history) > 30:
            self._command_history.pop(0)

        return cmd

    # ================================================================
    # COORDINATE TRANSFORM
    # ================================================================

    def cam_to_robot(self, cam_x, cam_y, cam_z):
        """
        Transform camera coords (cm) → robot coords (mm).

        Camera → Robot:
            robot_x = (cam_x - cam_x_center) * 10      [lateral, mm]
            robot_y = -(cam_y - cam_y_table) * 10       [height, mm, flip sign]
            robot_z = |cam_z - robot_z_cam| * 10        [depth from endline, mm]

        Returns (robot_x_mm, robot_y_mm, robot_z_mm).
        """
        x_center = self._cam_x_center if self._cam_x_center is not None else 0.0
        y_table = self._cam_y_table if self._cam_y_table is not None else 0.0
        z_end = self.robot_z_cam if self.robot_z_cam is not None else 0.0

        robot_x = (cam_x - x_center) * CM_TO_MM
        robot_y = -(cam_y - y_table) * CM_TO_MM   # cam Y down → robot Y up
        robot_z = abs(cam_z - z_end) * CM_TO_MM    # distance from endline

        return (robot_x, robot_y, robot_z)

    def check_workspace(self, robot_x, robot_y, robot_z):
        """Check if point is within robot elliptical workspace."""
        nx = robot_x / ROBOT_REACH_X_MM
        nz = robot_z / ROBOT_REACH_Z_MM
        in_xz = (nx*nx + nz*nz) <= 1.0
        in_y = 0 <= robot_y <= ROBOT_Y_MAX_MM
        return in_xz and in_y

    # ================================================================
    # TRAJECTORY FOR VISUALIZATION
    # ================================================================

    def predict_trajectory(self, duration=0.5, dt=0.005):
        if not self.is_ready(): return []
        pos = self.get_current_position()
        vel = self._get_corrected_velocity()
        if pos is None or vel is None: return []
        return self.physics_model.predict_trajectory(
            position=pos, velocity=vel, duration=duration, dt=dt)

    # ================================================================
    # CONFIDENCE / JITTER / RESET / STATS
    # ================================================================

    def _compute_confidence(self):
        n = len(self.position_buffer)
        t_span = self.position_buffer.get_time_span()
        vel = self.get_velocity()
        f_pts = min(n / 10.0, 1.0)
        f_time = min(t_span / 0.15, 1.0)
        total = self._accepted_count + self._rejected_count
        f_qual = self._accepted_count / max(total, 1)
        if vel['valid']:
            s = vel['speed']
            f_vel = min(s/50, 1.0) if s < 50 else (max(0, 1-(s-1200)/800) if s > 1200 else 1.0)
        else:
            f_vel = 0.0
        return round(0.3*f_pts + 0.3*f_time + 0.2*f_qual + 0.2*f_vel, 2)

    def get_command_jitter(self):
        if len(self._command_history) < 3:
            return {'x_std': 0, 'y_std': 0, 'z_std': 0, 'valid': False}
        arr = np.array(self._command_history[-10:])
        return {'x_std': float(np.std(arr[:,0])), 'y_std': float(np.std(arr[:,1])),
                'z_std': float(np.std(arr[:,2])), 'valid': True}

    def reset(self):
        self.position_buffer.clear()
        self._current_velocity = None
        self._velocity_valid = False
        self._rejected_count = 0
        self._accepted_count = 0
        self._last_command = None
        self._command_history = []

    def get_stats(self):
        vel = self.get_velocity()
        return {
            'buffer_size': len(self.position_buffer),
            'is_ready': self.is_ready(),
            'time_span': self.position_buffer.get_time_span(),
            'speed': vel['speed'] if vel['valid'] else 0.0,
            'accepted': self._accepted_count,
            'rejected': self._rejected_count,
            'robot_z_cam': self.robot_z_cam
        }