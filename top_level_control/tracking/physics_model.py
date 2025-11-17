"""
Physics-based Ball Tracking Model

Handles velocity estimation, position prediction, and visualization.
Separate from detection logic for clean separation of concerns.
"""

import cv2
import numpy as np


class PhysicsModel:
    """Velocity tracking and position prediction for ball tracking."""

    def __init__(self, smoothing=0.7, max_history=32):
        """
        Initialize the physics model.

        Args:
            smoothing: EMA smoothing factor for velocity (0-1, higher = more smoothing)
            max_history: Maximum position history length for trail
        """
        self.smoothing = smoothing
        self.max_history = max_history

        # State
        self.last_position = None
        self.velocity = np.array([0.0, 0.0])  # (vx, vy) in pixels/frame
        self.predicted_position = None
        self.position_history = []

    def update(self, detection):
        """
        Update physics model with new detection result.

        Args:
            detection: dict from BallTracker.detect() with 'found' and 'center'

        Returns:
            dict with physics info:
                - 'velocity': (vx, vy) array in pixels/frame
                - 'speed': magnitude in pixels/frame
                - 'predicted_position': (x, y) or None
        """
        result = {
            'velocity': self.velocity.copy(),
            'speed': 0.0,
            'predicted_position': None
        }

        if detection['found']:
            center = detection['center']

            # Update velocity based on position change
            if self.last_position is not None:
                new_velocity = np.array([
                    center[0] - self.last_position[0],
                    center[1] - self.last_position[1]
                ], dtype=np.float64)
                # Smooth velocity with exponential moving average
                self.velocity = self.smoothing * self.velocity + (1 - self.smoothing) * new_velocity

            self.last_position = center

            # Update history
            self.position_history.append(center)
            if len(self.position_history) > self.max_history:
                self.position_history.pop(0)

            # Predict next position
            self.predicted_position = (
                int(center[0] + self.velocity[0]),
                int(center[1] + self.velocity[1])
            )

            result['velocity'] = self.velocity.copy()
            result['speed'] = np.linalg.norm(self.velocity)
            result['predicted_position'] = self.predicted_position

        return result

    def draw(self, frame, detection, physics_info, show_trail=True,
             show_prediction=True, show_velocity=True, show_mask=False):
        """
        Draw detection and physics visualization on frame.

        Args:
            frame: BGR image to draw on
            detection: dict from BallTracker.detect()
            physics_info: dict from self.update()
            show_trail: draw position history trail
            show_prediction: draw predicted next position
            show_velocity: draw velocity vector
            show_mask: overlay HSV mask

        Returns:
            Annotated frame
        """
        if show_mask and 'mask' in detection:
            mask_colored = cv2.cvtColor(detection['mask'], cv2.COLOR_GRAY2BGR)
            mask_colored[:, :, 0] = 0
            mask_colored[:, :, 1] = 0
            frame = cv2.addWeighted(frame, 0.7, mask_colored, 0.3, 0)

        if detection['found']:
            center = detection['center']
            radius = int(detection['radius'])

            # Draw circle around ball
            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            cv2.circle(frame, center, 3, (0, 0, 255), -1)

            # Draw velocity vector
            if show_velocity and physics_info['speed'] > 1:
                vel = physics_info['velocity']
                end_point = (int(center[0] + vel[0] * 3), int(center[1] + vel[1] * 3))
                cv2.arrowedLine(frame, center, end_point, (255, 255, 0), 2, tipLength=0.3)

            # Draw predicted position
            if show_prediction and physics_info['predicted_position']:
                pred = physics_info['predicted_position']
                cv2.circle(frame, pred, 8, (255, 0, 255), 2)
                cv2.line(frame, center, pred, (255, 0, 255), 1)

            # Info text
            speed = physics_info['speed']
            text = f"Ball: ({center[0]}, {center[1]}) | Speed: {speed:.1f} px/f"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Ball: Not detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw trail
        if show_trail and len(self.position_history) > 1:
            for i in range(1, len(self.position_history)):
                thickness = int(np.sqrt(self.max_history / float(i + 1)) * 2.5)
                alpha = i / len(self.position_history)
                color = (0, int(255 * alpha), int(255 * (1 - alpha)))
                cv2.line(frame, self.position_history[i - 1],
                        self.position_history[i], color, thickness)

        return frame

    def reset(self):
        """Reset all physics state."""
        self.last_position = None
        self.velocity = np.array([0.0, 0.0])
        self.predicted_position = None
        self.position_history = []
