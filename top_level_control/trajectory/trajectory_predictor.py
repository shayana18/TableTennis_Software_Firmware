"""
DEPRECATED — use trajectory.robot_predictor.RobotPredictor for real-time robot control.

This module is kept for backward compatibility with camera-frame analysis scripts
(test_velocity_validation.py, test_trajectory_prediction.py). For new code, use:
    from trajectory.robot_predictor import RobotPredictor
    from trajectory.workspace import in_workspace, ROBOT_HOME, ...
    from comm_function.points_based_transform import cam_to_robot

Trajectory Predictor - Camera-Frame Prediction Class (DEPRECATED)

Predicts where and when the ball will arrive at the robot's end of the table.

GEOMETRY:
    Camera is on the SIDE of the table, looking ACROSS the table width.
    Cameras at midpoint of table length, ~110cm offset from table edge.

    Camera frame (cm):
        X = along table LENGTH (ball travels player-to-player, 274 cm)
        Y = vertical (down = positive, standard camera convention)
        Z = depth = across table WIDTH (152.5 cm + ~110cm camera offset)

    Robot frame (mm, delta robot from robot.h):
        robot_x = horizontal, across table width
        robot_y = horizontal, along table length
        robot_z = vertical (down = more negative)

    Camera → Robot axis mapping:
        cam_z (table width)   → robot_x (horizontal)
        cam_x (table length)  → robot_y (horizontal)
        cam_y (vertical down) → robot_z (vertical, inverted)

    Ball travels along CAMERA X → intercept at target camera-X position.

PIPELINE:
  position_buffer.py → velocity_estimator.py → trajectory_predictor.py
                                                        ↓
                                                 physics_model.py
"""

import time
import math
import numpy as np
from collections import deque

from .position_buffer import PositionBuffer
from .velocity_estimator import VelocityEstimator
from .physics_model import PhysicsModel


# ================================================================
# TABLE + ROBOT GEOMETRY (mm)
# ================================================================

TABLE_LENGTH_MM    = 2740.0
TABLE_WIDTH_MM     = 1525.0
NET_HEIGHT_MM      = 152.5

# Delta robot workspace (from updated robot.h)
# XY workspace is an ellipse; Z has min/max limits.
ELLIPSE_RADIUS_X = 790.0   # mm
ELLIPSE_RADIUS_Y = 540.0   # mm
LIMIT_POS_Z = -721.0       # mm (upper Z limit)
LIMIT_NEG_Z = -1050.0      # mm (lower Z limit)
NET_Z_TOP = -1000.0        # mm (provided for planner/policy logic)

ROBOT_HOME    = (0.0, 0.0, -900.0) # mm
MAX_CART_VEL  = 4000.0             # mm/s
MAX_CART_ACC  = 20000.0            # mm/s²

CM_TO_MM = 10.0

# ================================================================
# CAMERA POSE IN ROBOT FRAME — MEASURE AND UPDATE THESE
# ================================================================
#
# These define camera 0's position and orientation relative to the
# robot base frame. Used to build the rotation matrix + translation
# that converts triangulated camera coords (cm) → robot coords (mm).
#
# The camera has a 20° downward pitch, so cam_Y and cam_Z are NOT
# aligned with true vertical/horizontal. A simple axis swap would
# give ~100mm errors at typical depths. The rotation matrix fixes this.
#
# ┌─────────────────────────────────────────────────────────────────┐
# │  HOW TO MEASURE                                                │
# │                                                                │
# │  Stand behind the robot, looking toward the net.               │
# │  Robot base plate center = origin (0, 0, 0).                   │
# │                                                                │
# │  Robot frame:                                                  │
# │    +X = to robot's RIGHT (across table width)                  │
# │    +Y = toward the NET (along table length)                    │
# │    +Z = UP (positive = above base plate)                       │
# │    Note: workspace Z is negative because end-effector hangs    │
# │    below the base plate (-1050 to -721 mm).                    │
# │                                                                │
# │  1. POSITION (mm) — tape measure from robot base center        │
# │     to camera 0 lens center:                                   │
# │       CAM_POSE_X_MM : lateral offset                           │
# │         (+) = camera is to robot's right                       │
# │         (-) = camera is to robot's left                        │
# │       CAM_POSE_Y_MM : along-table offset                      │
# │         (+) = camera is toward net from robot                  │
# │         (-) = camera is behind robot                           │
# │       CAM_POSE_Z_MM : vertical offset                         │
# │         (+) = camera is above base plate                       │
# │         (-) = camera is below base plate                       │
# │                                                                │
# │  2. YAW (degrees) — horizontal angle of camera view direction  │
# │     Measured from robot +X axis, looking down from above:      │
# │       0°   = camera looks along robot +X (across table)        │
# │       90°  = camera looks along robot +Y (toward net)          │
# │       180° = camera looks along robot -X (opposite side)       │
# │     Your camera looks across the table → yaw ≈ 0° or 180°     │
# │     depending on which side the camera is on.                  │
# │                                                                │
# │  3. PITCH (degrees) — downward tilt from horizontal            │
# │       0°  = camera is level                                    │
# │       20° = camera tilted 20° DOWNWARD (your fixed stand)      │
# │     Positive = looking down.                                   │
# │                                                                │
# │  4. ROLL (degrees) — tilt around viewing axis                  │
# │       0° = camera is level (horizon is horizontal in image)    │
# │     Positive = clockwise when looking through the camera.      │
# │                                                                │
# │  After measuring, update the values below OR call              │
# │  predictor.set_camera_pose(x, y, z, yaw, pitch, roll).        │
# └─────────────────────────────────────────────────────────────────┘
#
CAM_POSE_X_MM   = 1600.5      # camera is to robot's RIGHT (+X)
CAM_POSE_Y_MM   = 1300     # camera is toward net from origin
CAM_POSE_Z_MM   = -485.4     # camera is below base plate (-Z)
CAM_POSE_YAW    = 180        # camera faces -X (180°) + 5° yaw toward net
CAM_POSE_PITCH  = 20.0       # camera tilted 20° downward (fixed stand)
CAM_POSE_ROLL   = 0.0        # camera is level


class TrajectoryPredictor:
    """
    Trajectory prediction with robot-frame interception output.

    Primary mode scans future trajectory points and chooses the first
    conservative-safe post-bounce workspace point.
    Legacy apex logic is kept as fallback.
    """

    # --- Outlier rejection ---
    MAX_BALL_SPEED    = 1500.0   # cm/s
    MAX_POSITION_JUMP = 40.0     # cm
    MAX_VELOCITY      = 2000.0   # cm/s
    GAP_RESET_TIME    = 0.08     # seconds — reset buffer after this gap
                                 # (~8 frames at 100fps; handles stereo overlap
                                 #  dropouts from camera convergence angle)

    # --- Prediction readiness ---
    MIN_TIME_SPAN     = 0.08     # seconds (~8 frames at 100fps)
    MAX_STALE_SAMPLE_S = 0.08    # do not predict from stale last sample

    # --- Bounce detection ---
    MIN_BOUNCE_FALL   = 10.0     # cm minimum Y descent before accepting bounce
    BOUNCE_RISE_FRAMES = 2       # consecutive rising frames to confirm bounce

    # --- Workspace-first interception policy ---
    # Intuition:
    #   1) Sample future trajectory points in camera frame.
    #   2) Transform ALL sampled points to robot frame.
    #   3) Pick the first post-bounce point that is conservative-safe and
    #      gives enough motion time margin.
    INTERCEPT_SCAN_DURATION_S = 1.0
    INTERCEPT_SCAN_DT_S = 0.01
    POST_BOUNCE_BUFFER_S = 0.03
    MIN_TIME_TO_HIT_S = 0.20
    SAFE_XY_SCALE = 0.85
    SAFE_Z_MARGIN_MM = 20.0

    def __init__(self,
                 buffer_size=15,
                 min_points=6,
                 velocity_method='regression',
                 gravity=981.0,
                 y_down=True,
                 enable_drag=True,
                 camera_pitch_deg=20.0,
                 robot_x_cam=None):
        """
        Args:
            camera_pitch_deg: Camera pitch angle in degrees (20° for our setup).
                              Decomposes gravity into Y and Z components.
            robot_x_cam: Camera-X coordinate of robot endline (cm).
                         Ball intercept happens at this X value.
        """
        self.buffer_size = buffer_size
        self.min_points = min_points

        self.robot_x_cam = robot_x_cam   # camera X at robot endline (cm)

        # Pipeline
        self.position_buffer = PositionBuffer(max_size=buffer_size)
        self.velocity_estimator = VelocityEstimator(method=velocity_method)
        self.physics_model = PhysicsModel(
            gravity=gravity, y_down=y_down, enable_drag=enable_drag,
            camera_pitch_deg=camera_pitch_deg)

        self._current_velocity = None
        self._velocity_valid = False
        self._rejected_count = 0
        self._accepted_count = 0
        self._last_command = None
        self._command_history = []
        self._last_reject_reason = None
        self._bounce_count = 0
        self._y_min_since_reset = None    # track min Y for bounce threshold
        self._rising_count = 0            # consecutive rising frames

        # Z-axis median filter (kills ±3cm stereo depth noise)
        self._z_median_window = deque(maxlen=3)

        # Camera-to-robot rigid transform (rotation + translation)
        # Built from camera pose measurements. Call set_camera_pose() to update.
        self._R_cam_to_robot = None   # 3x3 rotation matrix
        self._t_cam_to_robot = None   # 3-element translation vector (mm)
        self._cam_pose_set = False
        self._build_transform(
            pos_mm=(CAM_POSE_X_MM, CAM_POSE_Y_MM, CAM_POSE_Z_MM),
            yaw=CAM_POSE_YAW, pitch=CAM_POSE_PITCH, roll=CAM_POSE_ROLL)

    # ================================================================
    # CALIBRATION
    # ================================================================

    def set_robot_endline(self, camera_x):
        """Set the camera-X coordinate of the robot endline."""
        self.robot_x_cam = camera_x
        print(f"[Predictor] Robot endline at camera X = {camera_x:.1f} cm")

    def set_camera_pose(self, x_mm, y_mm, z_mm, yaw, pitch, roll):
        """
        Set camera pose in robot frame and rebuild the transform.

        See the HOW TO MEASURE block at the top of this file for details.

        Args:
            x_mm:  lateral offset from robot center (mm, + = robot's right)
            y_mm:  along-table offset from robot center (mm, + = toward net)
            z_mm:  vertical offset from robot base plate (mm, + = above)
            yaw:   horizontal angle from robot +X axis (degrees)
            pitch: downward tilt from horizontal (degrees, + = looking down)
            roll:  rotation around viewing axis (degrees, + = clockwise)
        """
        self._build_transform(pos_mm=(x_mm, y_mm, z_mm),
                              yaw=yaw, pitch=pitch, roll=roll)
        self._cam_pose_set = True
        print(f"[Predictor] Camera pose: pos=({x_mm:.0f}, {y_mm:.0f}, {z_mm:.0f}) mm, "
              f"yaw={yaw:.1f}° pitch={pitch:.1f}° roll={roll:.1f}°")

    def _build_transform(self, pos_mm, yaw, pitch, roll):
        """
        Build rotation matrix R and translation t for camera → robot transform.

            p_robot_mm = R @ (p_cam_cm * CM_TO_MM) + t

        Two stages:
          1. R_optical: converts OpenCV camera axes to a standard frame
             where forward = +X, right = +Y, up = +Z (pure axis swap).
          2. R_euler: applies the camera's physical orientation
             (yaw / pitch / roll) in the robot base frame.

        OpenCV camera axes:
            cam_x = right, cam_y = down, cam_z = forward

        Standard frame (at yaw=pitch=roll=0):
            +X = camera forward (across table width)
            +Y = camera right (along table length)
            +Z = camera up (vertical)
        """
        # Stage 1: OpenCV optical axes → standard frame
        #   cam_z (forward) → +X
        #   cam_x (right)   → -Y  (camera right = robot's -Y)
        #   cam_y (down)    → -Z
        R_optical = np.array([
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ])

        # Stage 2: Euler rotation ZYX (yaw → pitch → roll)
        yaw_r = math.radians(yaw)
        pitch_r = math.radians(pitch)
        roll_r = math.radians(roll)

        cy, sy = math.cos(yaw_r), math.sin(yaw_r)
        cp, sp = math.cos(pitch_r), math.sin(pitch_r)
        cr, sr = math.cos(roll_r), math.sin(roll_r)

        # Rz(yaw): rotation about vertical (+Z)
        Rz = np.array([[cy, -sy, 0.0],
                        [sy,  cy, 0.0],
                        [0.0, 0.0, 1.0]])

        # Ry(pitch): rotation about +Y — positive pitch tilts forward (+X)
        # direction downward (-Z), i.e. camera looks down
        Ry = np.array([[ cp, 0.0, sp],
                        [0.0, 1.0, 0.0],
                        [-sp, 0.0, cp]])

        # Rx(roll): rotation about +X (camera forward after yaw+pitch)
        Rx = np.array([[1.0, 0.0, 0.0],
                        [0.0, cr, -sr],
                        [0.0, sr,  cr]])

        R_euler = Rz @ Ry @ Rx

        # Combined: camera optical frame → robot base frame
        self._R_cam_to_robot = R_euler @ R_optical
        self._t_cam_to_robot = np.array(pos_mm, dtype=float)

    # ================================================================
    # Z-AXIS MEDIAN FILTER
    # ================================================================

    def _median_filter_z(self, z):
        """3-point running median to kill stereo depth noise (±3cm spikes)."""
        self._z_median_window.append(z)
        n = len(self._z_median_window)
        if n == 1:
            return z
        elif n == 2:
            return (self._z_median_window[0] + self._z_median_window[1]) / 2.0
        else:
            return sorted(self._z_median_window)[1]

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

        # After a stereo gap, reset buffer and accept as fresh start.
        # This handles tracking dropouts from limited stereo overlap
        # (camera convergence) or bounces that break continuity.
        if dt > self.GAP_RESET_TIME:
            self.position_buffer.clear()
            self._current_velocity = None
            self._velocity_valid = False
            self._y_min_since_reset = None
            self._rising_count = 0
            self._z_median_window.clear()
            return True, None

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
            self._last_reject_reason = reason
            return False
        self._last_reject_reason = None

        self._accepted_count += 1
        z_filtered = self._median_filter_z(z)
        self.position_buffer.add(x, y, z_filtered, timestamp)

        # Check for bounce — reset buffer to start fresh arc
        if self._detect_bounce():
            self._handle_bounce()
            return True

        if self.position_buffer.is_ready(self.min_points):
            if self.position_buffer.get_time_span() >= self.MIN_TIME_SPAN:
                self._update_velocity()

        return True

    # ================================================================
    # BOUNCE DETECTION
    # ================================================================

    def _detect_bounce(self):
        """
        Detect bounce by Y direction reversal: falling → rising.

        In camera coords Y+ = down, so:
          - Falling = Y increasing (dy > 0)
          - Rising after bounce = Y decreasing (dy < 0)

        Uses noise filtering:
          1. Ball must have fallen at least MIN_BOUNCE_FALL cm since last reset
          2. Must see BOUNCE_RISE_FRAMES consecutive rising frames
        """
        if len(self.position_buffer) < 3:
            return False

        recent = self.position_buffer.get_recent(3)
        y_prev = recent[-2]['y']
        y_curr = recent[-1]['y']
        dy = y_curr - y_prev

        # Track minimum Y (highest point ball has fallen to, since Y+ = down)
        if self._y_min_since_reset is None:
            self._y_min_since_reset = y_curr
        else:
            self._y_min_since_reset = max(self._y_min_since_reset, y_curr)

        # Check if ball is rising (Y decreasing = going up)
        if dy < 0:
            self._rising_count += 1
        else:
            self._rising_count = 0

        # Bounce conditions:
        # 1. Ball has fallen enough (Y increased by MIN_BOUNCE_FALL from start)
        first_y = self.position_buffer.get_oldest()['y']
        fall_amount = self._y_min_since_reset - first_y

        if (fall_amount >= self.MIN_BOUNCE_FALL and
                self._rising_count >= self.BOUNCE_RISE_FRAMES):
            return True

        return False

    def _handle_bounce(self):
        """Reset buffer on bounce, keeping last 2 points as new arc start."""
        last_two = self.position_buffer.get_recent(2)
        self.position_buffer.clear()
        self._current_velocity = None
        self._velocity_valid = False
        self._y_min_since_reset = None
        self._rising_count = 0
        self._bounce_count += 1

        # Re-seed Z median window with the kept points' Z values
        self._z_median_window.clear()
        for pt in last_two:
            self._z_median_window.append(pt['z'])
            self.position_buffer.add(pt['x'], pt['y'], pt['z'], pt['t'])

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
        latest = self.position_buffer.get_latest()
        if latest is None:
            return False

        # Guard against "ghost" predictions when no fresh triangulation arrived.
        sample_age_s = time.perf_counter() - float(latest['t'])
        if sample_age_s > self.MAX_STALE_SAMPLE_S:
            return False

        return (self.position_buffer.is_ready(self.min_points) and
                self._velocity_valid and
                self.position_buffer.get_time_span() >= self.MIN_TIME_SPAN)

    def _get_corrected_velocity(self):
        """Correct Vy and Vz for regression midpoint bias (gravity acts on both)."""
        vel = self.get_velocity()
        if not vel['valid']: return None

        _, timestamps = self.position_buffer.get_as_arrays()
        t_latest = timestamps[-1]
        t_mean = timestamps.mean()
        dt = t_latest - t_mean

        vy_corrected = vel['vy'] + self.physics_model.g_y * dt
        vz_corrected = vel['vz'] + self.physics_model.g_z * dt

        return (vel['vx'], vy_corrected, vz_corrected)

    # ================================================================
    # PREDICTION — workspace-first interception with apex fallback
    # ================================================================

    def predict(self, target_x=None):
        """
        Predict interception command using workspace-first policy.

        Args:
            target_x: Legacy camera-X intercept plane (cm), used only by fallback.

        Returns:
            dict with intercept_x/y/z, time_to_intercept, strategy, etc.
        """
        if target_x is None:
            target_x = self.robot_x_cam

        result = {
            'valid': False,
            'intercept_x': None, 'intercept_y': None, 'intercept_z': None,
            'time_to_intercept': None,
            'velocity_at_intercept': None,
            'current_position': None, 'current_velocity': None,
            'strategy': None
        }

        if not self.is_ready():
            return result

        pos = self.get_current_position()
        vel = self._get_corrected_velocity()
        if pos is None or vel is None:
            return result

        result['current_position'] = pos
        result['current_velocity'] = vel

        prediction = None
        strategy = None

        # Primary policy:
        # scan future points and choose the first conservative-safe workspace
        # point after bounce + time margin.
        future_traj = self.physics_model.predict_trajectory(
            position=pos,
            velocity=vel,
            duration=self.INTERCEPT_SCAN_DURATION_S,
            dt=self.INTERCEPT_SCAN_DT_S,
        )
        future_points_cam = [(p[0], p[1], p[2]) for p in future_traj]
        future_times_s = [p[3] for p in future_traj]

        # Bounce detector resets the buffer when a bounce is observed.
        # If _bounce_count > 0, current arc is post-bounce and t=0 is "now".
        intercept = None
        if self._bounce_count > 0:
            intercept = self.choose_intercept_point(
                points_cam=future_points_cam,
                times_s=future_times_s,
                t_bounce=0.0,
                bounce_buffer=self.POST_BOUNCE_BUFFER_S,
                min_time_to_hit=self.MIN_TIME_TO_HIT_S,
            )

        if intercept is not None:
            t_hit = float(intercept['time_to_hit'])
            prediction = {
                'position': intercept['point_cam'],
                'time': t_hit,
                'velocity': self.physics_model.predict_velocity(vel, t_hit),
                'valid': True,
            }
            strategy = 'workspace_first'

        # Fallback to legacy logic when workspace-first is unavailable:
        # apex-only fallback, as requested.
        if prediction is None or not prediction['valid']:
            vx = vel[0]
            vy = vel[1]
            moving_toward = (
                target_x is not None and (target_x - pos[0]) * vx > 0
            )

            if vy < 0 and (target_x is None or not moving_toward or abs(vy) > abs(vx) * 0.5):
                prediction = self.physics_model.position_at_apex(
                    position=pos, velocity=vel)
                if prediction['valid']:
                    strategy = 'apex'

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

    def choose_intercept_point(
            self,
            points_cam,
            times_s,
            R=None,
            t=None,
            t_bounce=None,
            bounce_buffer=0.03,
            min_time_to_hit=0.20):
        """
        Choose first post-bounce safe interception point from sampled trajectory.

        The important part is frame consistency:
          - Sampled points are generated in camera frame (cm).
          - They are transformed in batch to robot frame (mm).
          - Workspace and reach checks run only in robot frame.

        Note: R/t args are accepted for API similarity with calibration snippets,
        but this predictor uses its internally configured camera→robot transform.
        """
        if points_cam is None or times_s is None:
            return None
        if len(points_cam) == 0 or len(times_s) == 0:
            return None

        n = min(len(points_cam), len(times_s))
        if n <= 0:
            return None

        points_cam_arr = np.asarray(points_cam[:n], dtype=float).reshape(-1, 3)
        points_robot_arr = self.cam_to_robot_batch(points_cam_arr)

        for p_cam, p_robot, dt in zip(points_cam_arr, points_robot_arr, times_s[:n]):
            dt = float(dt)

            if t_bounce is not None and dt < (float(t_bounce) + float(bounce_buffer)):
                continue
            if dt < float(min_time_to_hit):
                continue

            rx, ry, rz = float(p_robot[0]), float(p_robot[1]), float(p_robot[2])
            if not self.check_safe_workspace(rx, ry, rz):
                continue

            return {
                'point_cam': (float(p_cam[0]), float(p_cam[1]), float(p_cam[2])),
                'point_robot': (rx, ry, rz),
                'time_to_hit': dt,
            }

        return None

    # ================================================================
    # ROBOT COMMAND
    # ================================================================

    def get_robot_command(self, target_x=None):
        """
        Get command for robot controller.

        Returns both camera coords (cm) and robot coords (mm).

        Robot frame (mm, delta robot):
            robot_x: horizontal, across table width    (ellipse radius 790)
            robot_y: horizontal, along table length    (ellipse radius 540)
            robot_z: vertical, down = more negative    (-1050 to -721)
        """
        cmd = {
            'valid': False,
            'cam_x': 0.0, 'cam_y': 0.0, 'cam_z': 0.0,
            'robot_x': 0.0, 'robot_y': 0.0, 'robot_z': 0.0,
            'in_workspace': False,
            'reachable': False,
            't': 0.0,
            'strategy': None, 'confidence': 0.0,
            'buffer_points': len(self.position_buffer),
            'time_span': self.position_buffer.get_time_span()
        }

        pred = self.predict(target_x)
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
        cmd['reachable'] = self.check_reachable(rx, ry, rz, cmd['t'])

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
        Transform camera coords (cm) → robot coords (mm) using rigid transform.

            p_robot = R_cam_to_robot @ (p_cam * CM_TO_MM) + t_cam_to_robot

        The rotation matrix R accounts for the camera's orientation (including
        the 20° pitch), so cam_Y and cam_Z are properly decomposed into true
        vertical and horizontal components.

        Returns (robot_x_mm, robot_y_mm, robot_z_mm).
        """
        p_cam_mm = np.array([cam_x, cam_y, cam_z]) * CM_TO_MM
        p_robot = self._R_cam_to_robot @ p_cam_mm + self._t_cam_to_robot
        return (float(p_robot[0]), float(p_robot[1]), float(p_robot[2]))

    def cam_to_robot_batch(self, points_cam):
        """Batch transform Nx3 camera points (cm) → robot points (mm)."""
        pts = np.asarray(points_cam, dtype=float)
        if pts.size == 0:
            return np.empty((0, 3), dtype=float)
        pts = pts.reshape(-1, 3)

        p_cam_mm = pts * CM_TO_MM
        return (self._R_cam_to_robot @ p_cam_mm.T).T + self._t_cam_to_robot

    def robot_to_cam(self, robot_x, robot_y, robot_z):
        """
        Inverse transform: robot coords (mm) → camera coords (cm).

            p_cam = R^T @ (p_robot - t) / CM_TO_MM

        Returns (cam_x, cam_y, cam_z) in cm.
        """
        p_robot = np.array([robot_x, robot_y, robot_z])
        p_cam_mm = self._R_cam_to_robot.T @ (p_robot - self._t_cam_to_robot)
        p_cam_cm = p_cam_mm / CM_TO_MM
        return (float(p_cam_cm[0]), float(p_cam_cm[1]), float(p_cam_cm[2]))

    def check_workspace(self, robot_x, robot_y, robot_z):
        """Check if point is within robot workspace (ellipse in XY + Z bounds)."""
        if not (LIMIT_NEG_Z <= robot_z <= LIMIT_POS_Z):
            return False

        nx = robot_x / ELLIPSE_RADIUS_X
        ny = robot_y / ELLIPSE_RADIUS_Y
        return (nx * nx + ny * ny) <= 1.0

    def check_safe_workspace(self, robot_x, robot_y, robot_z):
        """
        Conservative workspace for selecting interception points.

        We keep margin away from hard bounds so the chosen point is easier
        for timing/model error and path planning.
        """
        z_lo = LIMIT_NEG_Z + self.SAFE_Z_MARGIN_MM
        z_hi = LIMIT_POS_Z - self.SAFE_Z_MARGIN_MM
        if not (z_lo <= robot_z <= z_hi):
            return False

        nx = robot_x / (ELLIPSE_RADIUS_X * self.SAFE_XY_SCALE)
        ny = robot_y / (ELLIPSE_RADIUS_Y * self.SAFE_XY_SCALE)
        return (nx * nx + ny * ny) <= 1.0

    def check_reachable(self, robot_x, robot_y, robot_z, time_available):
        """Check if robot can reach position in time (trapezoidal profile estimate)."""
        home = ROBOT_HOME
        dist = math.sqrt((robot_x - home[0])**2 +
                         (robot_y - home[1])**2 +
                         (robot_z - home[2])**2)
        t_min = dist / MAX_CART_VEL + MAX_CART_VEL / MAX_CART_ACC
        return time_available >= t_min

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
        self._bounce_count = 0
        self._y_min_since_reset = None
        self._rising_count = 0
        self._z_median_window.clear()

    def get_stats(self):
        vel = self.get_velocity()
        return {
            'buffer_size': len(self.position_buffer),
            'is_ready': self.is_ready(),
            'time_span': self.position_buffer.get_time_span(),
            'speed': vel['speed'] if vel['valid'] else 0.0,
            'accepted': self._accepted_count,
            'rejected': self._rejected_count,
            'robot_x_cam': self.robot_x_cam,
            'bounces': self._bounce_count
        }
