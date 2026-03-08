"""
Trajectory Predictor - Main Prediction Class

Predicts where and when the ball will arrive at the robot's end of the table.

GEOMETRY:
    Camera is on the SIDE of the table, looking ACROSS the table width.
    Cameras at midpoint of table length, ~110cm offset from table edge.

    Camera frame (cm):
        X = along table LENGTH (ball travels player-to-player, 274 cm)
        Y = vertical (down = positive, standard camera convention)
        Z = depth = across table WIDTH (152.5 cm + ~110cm camera offset)

    Robot frame (mm, delta robot from robot.h):
        robot_x = horizontal, across table width    (±500mm)
        robot_y = horizontal, along table length    (±350mm)
        robot_z = vertical (down = more negative)   (-1100 to -700mm)

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

from .position_buffer import PositionBuffer
from .velocity_estimator import VelocityEstimator
from .physics_model import PhysicsModel


# ================================================================
# TABLE + ROBOT GEOMETRY (mm)
# ================================================================

TABLE_LENGTH_MM    = 2740.0
TABLE_WIDTH_MM     = 1525.0
NET_HEIGHT_MM      = 152.5

# Delta robot workspace (from robot.h)
ROBOT_LIMIT_X = (-500.0, 500.0)    # horizontal, across table width (mm)
ROBOT_LIMIT_Y = (-350.0, 350.0)    # horizontal, along table length (mm)
ROBOT_LIMIT_Z = (-1100.0, -700.0)  # vertical, down = more negative (mm)
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
# │    below the base plate (-700 to -1100 mm).                    │
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
CAM_POSE_X_MM   = 0.0      # PLACEHOLDER — measure lateral offset from robot center
CAM_POSE_Y_MM   = 0.0      # PLACEHOLDER — measure along-table offset from robot center
CAM_POSE_Z_MM   = 0.0      # PLACEHOLDER — measure vertical offset from robot base plate
CAM_POSE_YAW    = 0.0      # degrees — PLACEHOLDER (0° if camera looks along robot +X)
CAM_POSE_PITCH  = 20.0     # degrees — camera tilted 20° downward (known from fixed stand)
CAM_POSE_ROLL   = 0.0      # degrees — PLACEHOLDER (0° if camera is level)


class TrajectoryPredictor:
    """
    Trajectory prediction with X-axis interception and robot output.

    The ball travels along camera X (table length).
    Interception target is a camera-X position near the robot's endline.
    """

    # --- Outlier rejection ---
    MAX_BALL_SPEED    = 1500.0   # cm/s
    MAX_POSITION_JUMP = 40.0     # cm
    MAX_VELOCITY      = 2000.0   # cm/s
    GAP_RESET_TIME    = 0.08     # seconds — reset buffer after this gap
                                 # (~8 frames at 100fps; handles stereo overlap
                                 #  dropouts from camera convergence angle)

    # --- Prediction readiness ---
    MIN_TIME_SPAN     = 0.035    # seconds (~3 frames at 80fps)

    # --- Bounce detection ---
    MIN_BOUNCE_FALL   = 10.0     # cm minimum Y descent before accepting bounce
    BOUNCE_RISE_FRAMES = 2       # consecutive rising frames to confirm bounce

    def __init__(self,
                 buffer_size=15,
                 min_points=4,
                 velocity_method='regression',
                 gravity=981.0,
                 y_down=True,
                 enable_drag=True,
                 robot_x_cam=None):
        """
        Args:
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
            gravity=gravity, y_down=y_down, enable_drag=enable_drag)

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
        #   cam_x (right)   → +Y
        #   cam_y (down)    → -Z
        R_optical = np.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
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
        self.position_buffer.add(x, y, z, timestamp)

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

        for pt in last_two:
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
    # PREDICTION — intercept along camera X (table length)
    # ================================================================

    def predict(self, target_x=None):
        """
        Predict where ball will be when it reaches target_x.

        Args:
            target_x: Camera-X to intercept (cm). None = use robot_x_cam.

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

        if target_x is None or not self.is_ready():
            return result

        pos = self.get_current_position()
        vel = self._get_corrected_velocity()
        if pos is None or vel is None:
            return result

        result['current_position'] = pos
        result['current_velocity'] = vel

        vx = vel[0]
        vy = vel[1]
        prediction = None
        strategy = None

        # Is ball moving toward robot along X (table length)?
        # Robot is at robot_x_cam. Ball needs to reach that X.
        moving_toward = (target_x - pos[0]) * vx > 0

        # Case 1: Ball rising → predict apex (for lobs)
        if vy < 0 and (not moving_toward or abs(vy) > abs(vx) * 0.5):
            prediction = self.physics_model.position_at_apex(
                position=pos, velocity=vel)
            if prediction['valid']:
                strategy = 'apex'

        # Case 2: X-plane interception (primary)
        if prediction is None or not prediction['valid']:
            prediction = self.physics_model.position_at_x(
                position=pos, velocity=vel, target_x=target_x)
            if prediction['valid']:
                strategy = 'x_plane'

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

    def get_robot_command(self, target_x=None):
        """
        Get command for robot controller.

        Returns both camera coords (cm) and robot coords (mm).

        Robot frame (mm, delta robot):
            robot_x: horizontal, across table width    (±500)
            robot_y: horizontal, along table length    (±350)
            robot_z: vertical, down = more negative    (-1100 to -700)
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
        """Check if point is within robot rectangular workspace (from robot.h)."""
        return (ROBOT_LIMIT_X[0] <= robot_x <= ROBOT_LIMIT_X[1] and
                ROBOT_LIMIT_Y[0] <= robot_y <= ROBOT_LIMIT_Y[1] and
                ROBOT_LIMIT_Z[0] <= robot_z <= ROBOT_LIMIT_Z[1])

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