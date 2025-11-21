"""
Simple Ball Tracking Model

Handles velocity estimation, position prediction, and visualization.
Uses clean linear extrapolation for short-term prediction.



its radius-based depth esntimation for 2d overrhard camera aproach trial -- can further delete it! 
just keep for now for testing purpose. 
"""

import cv2
import numpy as np


class PhysicsModel:
    """Velocity tracking and position prediction for ball tracking."""

    def __init__(self, smoothing=0.7, max_history=32, num_predictions=10,
                 bounce_radius_threshold=4.0, bounce_cooldown=10,
                 stationary_threshold=1.0, stationary_time=3.0, fps=100):
        """
        Initialize the physics model.

        Args:
            smoothing: EMA smoothing factor for velocity (0-1, higher = more smoothing)
            max_history: Maximum position history length for trail
            num_predictions: Number of future positions to predict
            bounce_radius_threshold: Minimum radius increase to detect bounce (pixels)
            bounce_cooldown: Frames to wait before detecting next bounce
            stationary_threshold: Max speed to consider ball stationary (px/frame)
            stationary_time: Seconds ball must be stationary for calibration
            fps: Frame rate for time calculations
        """
        self.smoothing = smoothing
        self.max_history = max_history
        self.num_predictions = num_predictions
        self.bounce_radius_threshold = bounce_radius_threshold
        self.bounce_cooldown_max = bounce_cooldown
        self.stationary_threshold = stationary_threshold
        self.stationary_frames_required = int(stationary_time * fps)

        # State
        self.last_position = None
        self.velocity = np.array([0.0, 0.0])  # (vx, vy) in pixels/frame
        self.position_history = []
        self.predicted_positions = []

        # Bounce detection
        self.radius_history = []  # Track ball size over time
        self.last_radius = None
        self.bounce_detected = False
        self.bounce_cooldown = 0
        self.total_bounces = 0

        # Landing prediction
        self.stationary_frames = 0
        self.calibrated = False
        self.table_radius = None  # Max radius (ball on table)
        self.landing_point = None  # Predicted landing position

    def update(self, detection):
        """
        Update physics model with new detection result.

        Args:
            detection: dict from BallTracker.detect() with 'found', 'center', and 'radius'

        Returns:
            dict with physics info:
                - 'velocity': (vx, vy) array in pixels/frame
                - 'speed': magnitude in pixels/frame
                - 'predicted_positions': list of future (x, y) positions
                - 'bounce_detected': bool, whether bounce was detected this frame
                - 'total_bounces': int, total bounces detected
                - 'landing_point': (x, y) tuple or None, predicted landing position
                - 'calibrated': bool, whether table radius is calibrated
        """
        # Reset bounce flag
        self.bounce_detected = False

        # Decrement cooldown
        if self.bounce_cooldown > 0:
            self.bounce_cooldown -= 1

        result = {
            'velocity': self.velocity.copy(),
            'speed': 0.0,
            'predicted_positions': [],
            'bounce_detected': False,
            'total_bounces': self.total_bounces,
            'landing_point': None,
            'calibrated': self.calibrated
        }

        if detection['found']:
            center = detection['center']
            radius = detection['radius']

            # Update velocity based on position change
            if self.last_position is not None:
                new_velocity = np.array([
                    center[0] - self.last_position[0],
                    center[1] - self.last_position[1]
                ], dtype=np.float64)
                # Smooth velocity with exponential moving average
                self.velocity = self.smoothing * self.velocity + (1 - self.smoothing) * new_velocity

            self.last_position = center

            # Track radius for bounce detection
            self.radius_history.append(radius)
            if len(self.radius_history) > 30:  # Keep last 30 frames (~0.3s at 100fps)
                self.radius_history.pop(0)

            # Calculate current speed
            speed = np.linalg.norm(self.velocity)

            # Auto-calibrate: track maximum radius (ball on table)
            # Also require ball to be relatively stationary for stable calibration
            if speed < self.stationary_threshold:
                self.stationary_frames += 1
                # Update table_radius when stationary and radius is near maximum
                if self.stationary_frames >= self.stationary_frames_required:
                    self.calibrated = True
                    if self.table_radius is None or radius > self.table_radius:
                        self.table_radius = radius
            else:
                self.stationary_frames = 0
                # Also update table_radius if we see a larger radius (handles initial calibration)
                if self.table_radius is None:
                    self.table_radius = radius
                elif radius > self.table_radius * 1.05:  # 5% larger
                    self.table_radius = radius

            # Predict landing point when ball is moving and calibrated
            if speed > 2 and self.calibrated and self.table_radius is not None:
                # Use more robust velocity estimate from position history
                velocity_estimate = self._estimate_velocity_robust()
                self.landing_point = self._predict_landing(center, velocity_estimate, radius)
            else:
                self.landing_point = None

            # Detect bounce: ball suddenly gets larger (closer to overhead camera)
            if self.last_radius is not None and self.bounce_cooldown == 0:
                radius_increase = radius - self.last_radius

                # Bounce detected if radius increased significantly
                if radius_increase > self.bounce_radius_threshold:
                    self.bounce_detected = True
                    self.total_bounces += 1
                    self.bounce_cooldown = self.bounce_cooldown_max

            self.last_radius = radius

            # Update history
            self.position_history.append(center)
            if len(self.position_history) > self.max_history:
                self.position_history.pop(0)

            # Predict future positions using linear extrapolation
            self.predicted_positions = []
            if speed > 2:  # Only predict if ball is moving
                x0, y0 = center
                vx, vy = self.velocity

                for t in range(1, self.num_predictions + 1):
                    x_pred = x0 + vx * t
                    y_pred = y0 + vy * t

                    # Only add if within reasonable bounds
                    if -100 <= x_pred <= 1380 and -100 <= y_pred <= 820:
                        self.predicted_positions.append((int(x_pred), int(y_pred)))

            result['velocity'] = self.velocity.copy()
            result['speed'] = speed
            result['predicted_positions'] = self.predicted_positions
            result['bounce_detected'] = self.bounce_detected
            result['total_bounces'] = self.total_bounces
            result['landing_point'] = self.landing_point
            result['calibrated'] = self.calibrated

        return result

    def _estimate_velocity_robust(self):
        """
        Estimate velocity from recent position history using linear regression.
        More robust than frame-to-frame differences.

        Returns:
            (vx, vy) velocity array in pixels/frame
        """
        if len(self.position_history) < 3:
            return self.velocity  # Fallback to EMA velocity

        # Use last 8 positions for robust estimate
        recent_positions = self.position_history[-8:]
        n = len(recent_positions)

        if n < 3:
            return self.velocity

        # Time steps
        t = np.arange(n)

        # Extract x and y coordinates
        x_coords = np.array([p[0] for p in recent_positions], dtype=np.float64)
        y_coords = np.array([p[1] for p in recent_positions], dtype=np.float64)

        try:
            # Fit linear regression to get velocity (slope)
            vx = np.polyfit(t, x_coords, 1)[0]
            vy = np.polyfit(t, y_coords, 1)[0]
            return np.array([vx, vy], dtype=np.float64)
        except (np.linalg.LinAlgError, ValueError):
            return self.velocity  # Fallback

    def _predict_landing(self, position, velocity, current_radius):
        """
        Predict where ball will land based on current state.

        Uses polynomial extrapolation on radius history to predict when
        ball will reach table (radius = table_radius).

        Args:
            position: (x, y) current position
            velocity: (vx, vy) current velocity
            current_radius: current ball radius in pixels

        Returns:
            (x, y) predicted landing position or None
        """
        if self.table_radius is None or current_radius >= self.table_radius * 0.95:
            return None  # Already on table or very close

        # Need enough history for reliable prediction
        if len(self.radius_history) < 10:
            return None

        # Get recent radius measurements and check if ball is falling
        recent_radii = self.radius_history[-10:]
        radius_change = recent_radii[-1] - recent_radii[0]

        # Only predict if ball is getting closer (radius increasing)
        if radius_change < 0.5:
            return None

        # Fit polynomial to radius history to extrapolate
        # Use last 10 frames for prediction
        time_steps = np.arange(len(recent_radii))

        try:
            # Fit quadratic curve to radius over time
            # Ball falling: radius increases non-linearly as it approaches table
            coeffs = np.polyfit(time_steps, recent_radii, deg=2)
            poly = np.poly1d(coeffs)

            # Find when radius will reach table_radius
            # Solve: poly(t) = table_radius
            # This is a quadratic equation: a*t^2 + b*t + c - table_radius = 0
            a, b, c = coeffs
            c_adjusted = c - self.table_radius

            discriminant = b**2 - 4*a*c_adjusted
            if discriminant < 0:
                return None  # No real solution

            # Get positive root (future time)
            t1 = (-b + np.sqrt(discriminant)) / (2*a)
            t2 = (-b - np.sqrt(discriminant)) / (2*a)

            # Choose the smallest positive time
            frames_to_landing = None
            for t in [t1, t2]:
                if t > 0 and (frames_to_landing is None or t < frames_to_landing):
                    frames_to_landing = t

            if frames_to_landing is None or frames_to_landing > 100:
                return None  # Too far in future or no valid solution

            # Add offset since we measured from start of recent_radii window
            frames_to_landing = frames_to_landing - len(recent_radii) + 1

            # Ensure positive
            if frames_to_landing < 0:
                frames_to_landing = 1

            # Project position linearly (X,Y velocity assumed constant)
            x_land = position[0] + velocity[0] * frames_to_landing
            y_land = position[1] + velocity[1] * frames_to_landing

            # Sanity check: landing point should be within reasonable bounds
            if -200 <= x_land <= 1500 and -200 <= y_land <= 1000:
                return (int(x_land), int(y_land))

        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            # Numerical instability - fallback to simple method
            pass

        # Fallback: simple linear extrapolation
        avg_radius_change = radius_change / len(recent_radii)
        if avg_radius_change > 0.1:
            radius_remaining = self.table_radius - current_radius
            frames_to_landing = radius_remaining / avg_radius_change
            frames_to_landing = max(1, min(frames_to_landing, 100))

            x_land = position[0] + velocity[0] * frames_to_landing
            y_land = position[1] + velocity[1] * frames_to_landing

            if -200 <= x_land <= 1500 and -200 <= y_land <= 1000:
                return (int(x_land), int(y_land))

        return None

    def draw(self, frame, detection, physics_info, show_trail=True,
             show_prediction=True, show_velocity=True, show_mask=False,
             show_landing=True):
        """
        Draw detection and physics visualization on frame.

        Args:
            frame: BGR image to draw on
            detection: dict from BallTracker.detect()
            physics_info: dict from self.update()
            show_trail: draw position history trail
            show_prediction: draw predicted future positions
            show_velocity: draw velocity vector
            show_mask: overlay HSV mask
            show_landing: draw predicted landing point

        Returns:
            Annotated frame
        """
        if show_mask and 'mask' in detection:
            mask_colored = cv2.cvtColor(detection['mask'], cv2.COLOR_GRAY2BGR)
            mask_colored[:, :, 0] = 0
            mask_colored[:, :, 1] = 0
            frame = cv2.addWeighted(frame, 0.7, mask_colored, 0.3, 0)

        # Draw predicted landing point (before everything else so it's in background)
        landing_point = physics_info.get('landing_point')
        if show_landing and landing_point is not None and detection['found']:
            current_pos = detection['center']

            # Draw trajectory curve from current position to landing point
            # Create smooth curve with intermediate points
            num_curve_points = 15
            for i in range(num_curve_points):
                t = i / (num_curve_points - 1)
                # Simple linear interpolation for now
                x = int(current_pos[0] + t * (landing_point[0] - current_pos[0]))
                y = int(current_pos[1] + t * (landing_point[1] - current_pos[1]))

                if i > 0:
                    # Draw line segment
                    prev_x = int(current_pos[0] + (t - 1/(num_curve_points-1)) * (landing_point[0] - current_pos[0]))
                    prev_y = int(current_pos[1] + (t - 1/(num_curve_points-1)) * (landing_point[1] - current_pos[1]))
                    # Gradient color from cyan to yellow
                    color = (int(255 * (1 - t)), int(255), int(255 * t))
                    thickness = max(1, int(4 * (1 - t * 0.5)))
                    cv2.line(frame, (prev_x, prev_y), (x, y), color, thickness)

            # Draw prominent landing target
            # Pulsing effect could be added by varying sizes over time
            cv2.circle(frame, landing_point, 40, (0, 255, 255), 4)  # Yellow outer ring
            cv2.circle(frame, landing_point, 28, (0, 255, 255), 3)
            cv2.circle(frame, landing_point, 16, (0, 255, 255), 2)
            cv2.circle(frame, landing_point, 6, (0, 255, 255), -1)  # Filled center

            # Draw extended crosshair
            cv2.line(frame, (landing_point[0] - 50, landing_point[1]),
                    (landing_point[0] + 50, landing_point[1]), (0, 255, 255), 3)
            cv2.line(frame, (landing_point[0], landing_point[1] - 50),
                    (landing_point[0], landing_point[1] + 50), (0, 255, 255), 3)

            # Label with shadow for visibility
            text_pos = (landing_point[0] - 50, landing_point[1] - 60)
            # Shadow
            cv2.putText(frame, "LANDING", (text_pos[0] + 2, text_pos[1] + 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
            # Text
            cv2.putText(frame, "LANDING", text_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Draw trail (historical path)
        if show_trail and len(self.position_history) > 1:
            for i in range(1, len(self.position_history)):
                thickness = int(np.sqrt(self.max_history / float(i + 1)) * 2.5)
                alpha = i / len(self.position_history)
                color = (0, int(255 * alpha), int(255 * (1 - alpha)))
                cv2.line(frame, self.position_history[i - 1],
                        self.position_history[i], color, thickness)

        # Draw predicted future positions
        if show_prediction and len(physics_info['predicted_positions']) > 0:
            predictions = physics_info['predicted_positions']

            # Draw connecting line
            if len(predictions) > 1:
                for i in range(len(predictions) - 1):
                    alpha = i / len(predictions)
                    # Gradient from cyan to magenta
                    color = (int(255 * (1 - alpha)), int(200 - 100 * alpha), int(100 + 155 * alpha))
                    cv2.line(frame, predictions[i], predictions[i + 1], color, 2)

            # Draw circles at each predicted position
            for i, pred_pos in enumerate(predictions):
                alpha = i / len(predictions)
                # Size increases slightly with distance
                radius = int(6 + alpha * 6)
                # Color gradient cyan to magenta
                color = (int(255 * (1 - alpha)), int(200 - 100 * alpha), int(100 + 155 * alpha))
                cv2.circle(frame, pred_pos, radius, color, 2)
                # Fill the furthest one
                if i == len(predictions) - 1:
                    cv2.circle(frame, pred_pos, radius - 2, color, -1)

        if detection['found']:
            center = detection['center']
            radius = int(detection['radius'])

            # Draw circle around ball (current position)
            ball_color = (0, 255, 0)  # Green by default

            # Flash orange if bounce just detected
            if physics_info.get('bounce_detected', False):
                ball_color = (0, 165, 255)  # Orange flash
                # Draw larger impact circle
                cv2.circle(frame, center, radius + 15, (0, 165, 255), 5)
                cv2.circle(frame, center, radius + 25, (0, 100, 255), 2)

            cv2.circle(frame, center, radius, ball_color, 3)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            # Draw velocity vector (thick and prominent)
            if show_velocity and physics_info['speed'] > 2:
                vel = physics_info['velocity']
                scale = 5  # Scale up the arrow
                end_point = (int(center[0] + vel[0] * scale), int(center[1] + vel[1] * scale))
                cv2.arrowedLine(frame, center, end_point, (0, 255, 255), 3, tipLength=0.3)

            # Info text
            speed = physics_info['speed']
            text = f"Ball: ({center[0]}, {center[1]}) | Speed: {speed:.1f} px/f"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Velocity components
            vx, vy = physics_info['velocity']
            vel_text = f"Velocity: ({vx:.1f}, {vy:.1f})"
            cv2.putText(frame, vel_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Radius info
            radius_text = f"Radius: {radius:.1f}px"
            cv2.putText(frame, radius_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

        else:
            cv2.putText(frame, "Ball: Not detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame

    def reset(self):
        """Reset all physics state."""
        self.last_position = None
        self.velocity = np.array([0.0, 0.0])
        self.position_history = []
        self.predicted_positions = []
        self.radius_history = []
        self.last_radius = None
        self.bounce_detected = False
        self.bounce_cooldown = 0
        self.total_bounces = 0
        self.stationary_frames = 0
        self.calibrated = False
        self.table_radius = None
        self.landing_point = None
