import cv2
import numpy as np

from circle_detection import CircleDetection
import color_tracking as ct


class BallTracker:
    """
    Multi-stage ball detector:
      1) Motion gate (background subtraction mask)
      2) Color gate (HSV+LAB fused mask)
      3) Shape gate (Hough circle)
    """

    def __init__(
        self,
        circle_detector=None,
        color_tracker=None,
        back_sub_history=30,
        back_sub_var_threshold=150,
        back_sub_detect_shadows=False,
        back_sub_learning_rate=0.3,
        motion_blur_ksize=3,
        color_min_pixels=30,
        gate_motion=True,
        gate_color=True,
        gate_shape=True,
    ):
        """
        Args:
            circle_detector: Optional CircleDetection instance.
            color_tracker: Optional EnhancedBallTracker instance.
            back_sub_history: Background subtractor history size.
            back_sub_var_threshold: Threshold on squared Mahalanobis distance.
            back_sub_detect_shadows: Enable shadow detection in MOG2.
            back_sub_learning_rate: Learning rate for background update.
            motion_blur_ksize: Median blur kernel size for motion mask cleanup.
            color_min_pixels: Minimum matching pixels required to accept color.
            gate_motion: Enable motion gate by default.
            gate_color: Enable color gate by default.
            gate_shape: Enable shape gate by default.
        """
        self.circle_detector = circle_detector or CircleDetection()
        self.color_tracker = color_tracker or ct.EnhancedBallTracker()
        self.back_sub = cv2.createBackgroundSubtractorMOG2(
            history=back_sub_history,
            varThreshold=back_sub_var_threshold,
            detectShadows=back_sub_detect_shadows,
        )
        self.back_sub_learning_rate = back_sub_learning_rate
        self.motion_blur_ksize = motion_blur_ksize
        self.color_min_pixels = color_min_pixels

        self.motion_gate = gate_motion
        self.color_gate = gate_color
        self.shape_gate = gate_shape

    def set_gates(self, motion=None, color=None, shape=None):
        """
        Update gate states. Any argument left as None is unchanged.
        """
        if motion is not None:
            self.motion_gate = bool(motion)
        if color is not None:
            self.color_gate = bool(color)
        if shape is not None:
            self.shape_gate = bool(shape)

    def toggle_gate(self, gate_name):
        """
        Toggle a gate by name: "motion", "color", or "shape".
        """
        if gate_name == "motion":
            self.motion_gate = not self.motion_gate
        elif gate_name == "color":
            self.color_gate = not self.color_gate
        elif gate_name == "shape":
            self.shape_gate = not self.shape_gate
        else:
            raise ValueError(f"Unknown gate: {gate_name}")

    def _gate_state(self):
        return self.motion_gate, self.color_gate, self.shape_gate

    def get_gates(self):
        """
        Return current gate states as a dict.
        """
        gate_motion, gate_color, gate_shape = self._gate_state()
        return {"motion": gate_motion, "color": gate_color, "shape": gate_shape}

    def _binary_mask(self, mask):
        if mask is None:
            return None
        _, bin_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        return bin_mask

    def _motion_mask(self, frame):
        """
        Compute a cleaned motion mask using background subtraction.
        """
        mask = self.back_sub.apply(frame, learningRate=self.back_sub_learning_rate)
        if self.motion_blur_ksize > 1:
            mask = cv2.medianBlur(mask, self.motion_blur_ksize)
        return mask

    def _color_mask(self, frame, motion_mask=None):
        """
        Compute HSV+LAB fused color mask, optionally restricted by motion.
        """
        color_mask = self.color_tracker.build_fused_mask(frame)
        if motion_mask is not None:
            motion_bin = self._binary_mask(motion_mask)
            if motion_bin is not None:
                color_mask = cv2.bitwise_and(color_mask, color_mask, mask=motion_bin)
        return color_mask

    def _color_gate(self, color_mask):
        """
        Check if the color mask contains enough matching pixels.
        """
        if color_mask is None:
            return False
        return cv2.countNonZero(color_mask) >= self.color_min_pixels

    def _shape_gate(self, frame, gate_mask):
        """
        Run circle detection using the provided gate mask.
        """
        return self.circle_detector._circle_detection(frame, gate_mask)

    def _annotate_status(self, frame, motion_found, color_found, shape_found, gate_state, found=None):
        """
        Overlay detection booleans and gate state on the frame.
        """
        gate_motion, gate_color, gate_shape = gate_state
        lines = [
            f"motion: {motion_found} (gate {'ON' if gate_motion else 'OFF'})",
            f"color: {color_found} (gate {'ON' if gate_color else 'OFF'})",
            f"shape: {shape_found} (gate {'ON' if gate_shape else 'OFF'})",
        ]
        if found is not None:
            lines.append(f"final: {found}")
        y = 24
        for line in lines:
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 22

    def detect(self, frame):
        """
        Full detection pipeline: motion gate -> color gate -> shape gate.

        Args:
            frame: BGR frame from the camera.

        Returns:
            dict with keys:
                found: True if enabled gates pass
                motion_found: True if motion is detected
                color_found: True if color mask passes threshold
                shape_found: True if a circle was detected
                center: (x, y) circle center or None
                radius: Circle radius or None
                bbox: (x, y, w, h) bounding box or None
                motion_frame: motion mask for display
                color_frame: fused color mask for display
                shape_frame: circle search image for display
                full_frame: annotated display frame
        """
        gate_state = self._gate_state()
        gate_motion, gate_color, gate_shape = gate_state

        motion_mask = None
        motion_bin = None
        motion_found = False
        if gate_motion:
            motion_mask = self._motion_mask(frame)
            motion_bin = self._binary_mask(motion_mask)
            motion_found = motion_bin is not None and cv2.countNonZero(motion_bin) > 0

        color_mask = None
        color_found = False
        if gate_color:
            color_mask = self._color_mask(frame, motion_mask if gate_motion else None)
            color_found = self._color_gate(color_mask)

        gate_mask = None
        if gate_motion and motion_bin is not None:
            gate_mask = motion_bin
        if gate_color and color_mask is not None:
            gate_mask = color_mask if gate_mask is None else cv2.bitwise_and(gate_mask, color_mask)

        shape_result = {"found": False, "center": None, "radius": None, "bbox": None, "image": None}
        shape_found = False
        center = None
        radius = None
        bbox = None
        if gate_shape:
            shape_result = self._shape_gate(frame, gate_mask)
            shape_found = bool(shape_result.get("found"))
            center = shape_result.get("center")
            radius = shape_result.get("radius")
            bbox = shape_result.get("bbox")

        found = (
            (not gate_motion or motion_found)
            and (not gate_color or color_found)
            and (not gate_shape or shape_found)
        )

        full_frame = frame.copy()
        if gate_motion and motion_found and motion_bin is not None and gate_color and color_found:
            contours_data = cv2.findContours(motion_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours_data[0] if len(contours_data) == 2 else contours_data[1]
            if contours:
                biggest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(biggest)
                cv2.rectangle(full_frame, (x, y), (x + w, y + h), (2, 0, 0), 2)


        self._annotate_status(full_frame, motion_found, color_found, shape_found, gate_state, found)

        blank = np.zeros(frame.shape[:2], dtype=np.uint8)
        return {
            "found": found,
            "motion_found": motion_found,
            "color_found": color_found,
            "shape_found": shape_found,
            "center": center,
            "radius": radius,
            "bbox": bbox,
            "motion_frame": motion_mask if gate_motion and motion_mask is not None else blank,
            "color_frame": color_mask if gate_color and color_mask is not None else blank,
            "shape_frame": shape_result.get("image") if gate_shape and shape_result.get("image") is not None else blank,
            "full_frame": full_frame,
        }

    def draw_detection(self, frame, color=(0, 255, 0), thickness=2):
        """
        Draw the final (motion + shape + color) detection on the frame.

        Args:
            frame: BGR frame to draw on (modified in-place).
            color: BGR color tuple for drawing.
            thickness: Line thickness for circle and box.

        Returns:
            The frame with annotations (if any).
        """
        detection = self.detect(frame)
        draw_frame = frame.copy()
        
        self._annotate_status(
            draw_frame,
            detection.get("motion_found"),
            detection.get("color_found"),
            detection.get("shape_found"),
            self._gate_state(),
            detection.get("found"),
        )
        return draw_frame, detection["shape_frame"]
