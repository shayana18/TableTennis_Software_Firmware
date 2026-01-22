import cv2
import numpy as np


class CircleDetection:
    """
    Motion-gated circle detection using HoughCircles.

    Designed to work on top of a background-subtraction mask.
    """

    def __init__(
        self,
        min_radius=2,
        max_radius=50,
        min_contour_area=10,
        roi_padding=8,
        hough_dp=1.2,
        hough_min_dist=100,
        hough_param1=120,
        hough_param2=16,
        blur_ksize=5,
        mask_blur_ksize=3,
    ):
        """
        Args:
            min_radius: Smallest circle radius to accept (pixels).
            max_radius: Largest circle radius to accept (pixels).
            min_contour_area: Rejects tiny motion blobs in the mask (pixels^2).
            roi_padding: Expands the motion ROI to avoid cutting off the ball.
            hough_dp: Inverse ratio of accumulator resolution for HoughCircles.
            hough_min_dist: Minimum distance between circle centers (pixels).
            hough_param1: Canny high threshold used by HoughCircles.
            hough_param2: Accumulator threshold; higher means fewer detections.
            blur_ksize: Gaussian blur size for the frame before HoughCircles.
            mask_blur_ksize: Median blur size for the motion mask cleanup.
        """
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_contour_area = min_contour_area
        self.roi_padding = roi_padding
        self.hough_dp = hough_dp
        self.hough_min_dist = hough_min_dist
        self.hough_param1 = hough_param1
        self.hough_param2 = hough_param2
        self.blur_ksize = blur_ksize
        self.mask_blur_ksize = mask_blur_ksize

        # Elliptical kernel for morphological cleanup of the motion mask.
        self.mask_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def _normalize_mask(self, mask):
        """
        Normalize an input mask into a binary uint8 image.

        Args:
            mask: Binary-like mask (any dtype or 1/3 channel).

        Returns:
            Binary mask (uint8) with values 0 or 255, or None if input is None.
        """
        if mask is None:
            return None
        if mask.ndim == 3:
            # Ensure a single-channel mask for contour detection.
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        # Convert any non-zero pixels to 255 so the mask is truly binary.
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        return mask

    def _extract_contours(self, mask):
        """
        Extract external contours from the motion mask.

        Args:
            mask: Binary motion mask (uint8).

        Returns:
            List of contours filtered by minimum area.
        """
        # Find external contours of moving regions in the mask.
        contours_data = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # in opencv4 the return of find contours is (contours, hierarchy) but in opencv3 its (image, contours, hierarchy)
        # this will pick the right one based on length of returned tuple 
        contours = contours_data[0] if len(contours_data) == 2 else contours_data[1]
        filtered = [c for c in contours if cv2.contourArea(c) >= self.min_contour_area]
        return filtered

    def _build_roi(self, contours, frame_shape):
        """
        Build a tight Region of Interest (ROI) around motion contours with padding.

        Args:
            contours: List of contours from the motion mask.
            frame_shape: Shape of the full frame (H, W, C).

        Returns:
            ROI tuple (x, y, w, h) or None if no valid ROI.
        """
        if not contours:
            return None
        x_min = y_min = float("inf")
        x_max = y_max = -float("inf")
        for contour in contours:
            # Bounding rectangles give a quick ROI around motion blobs.
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        h_img, w_img = frame_shape[:2]
        x_min = max(0, int(x_min) - self.roi_padding)
        y_min = max(0, int(y_min) - self.roi_padding)
        x_max = min(w_img, int(x_max) + self.roi_padding)
        y_max = min(h_img, int(y_max) + self.roi_padding)
        if x_max <= x_min or y_max <= y_min:
            return None
        return (x_min, y_min, x_max - x_min, y_max - y_min)

    def _clamp_bbox(self, bbox, frame_shape):
        """
        Clamp a bounding box to the frame limits.

        Args:
            bbox: Proposed bounding box (x, y, w, h).
            frame_shape: Shape of the full frame (H, W, C).

        Returns:
            Clamped bounding box (x, y, w, h) or None if invalid.
        """
        if bbox is None:
            return None
        x, y, w, h = bbox
        h_img, w_img = frame_shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        if w <= 0 or h <= 0:
            return None
        return (x, y, w, h)

    def _circle_detection(self, frame, motion_mask=None):
        """
        Run motion-gated Hough circle detection.

        Args:
            frame: BGR frame
            motion_mask: binary mask from background subtraction

        Returns:
            dict with keys:
                found: bool detection flag
                center: (x, y) center or None
                radius: int radius or None
                bbox: (x, y, w, h) bounding box or None
                roi: (x, y, w, h) ROI used for search or None
                contours: motion contours used to build ROI
                mask: normalized motion mask (uint8) or None
        """
        mask = self._normalize_mask(motion_mask)
        contours = []
        roi = None
        if mask is not None:
            if self.mask_blur_ksize > 1:
                # Median blur removes isolated mask speckles without smearing edges.
                mask = cv2.medianBlur(mask, self.mask_blur_ksize)
            # Opening removes tiny blobs while keeping larger motion regions.
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.mask_kernel, iterations=1)
            contours = self._extract_contours(mask)
            roi = self._build_roi(contours, frame.shape)

        result = {
            "found": False,
            "center": None,
            "radius": None,
            "bbox": None,
            "roi": roi,
            "contours": contours,
            "mask": mask,
            "image": None,
        }

        if mask is not None and not contours:
            return result

        # HoughCircles expects a single-channel, slightly smoothed image.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.blur_ksize > 1:
            gray = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)

        search = gray
        offset_x = 0
        offset_y = 0
        if roi is not None:
            x, y, w, h = roi
            search = gray[y:y + h, x:x + w]
            offset_x, offset_y = x, y
            if mask is not None:
                # Mask the search region to restrict HoughCircles to motion areas.
                mask_roi = mask[y:y + h, x:x + w]
                search = cv2.bitwise_and(search, search, mask=mask_roi)
        result["image"] = search

        # HoughCircles uses a gradient-based circle accumulator.
        circles = cv2.HoughCircles(
            search,
            cv2.HOUGH_GRADIENT,
            dp=self.hough_dp,
            minDist=self.hough_min_dist,
            param1=self.hough_param1,
            param2=self.hough_param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )

        if circles is None:
            return result

        circles = np.round(circles[0, :]).astype("int")
        # Prefer the largest detected circle (likely the ball).
        best_circle = max(circles, key=lambda c: c[2])
        x, y, r = best_circle
        center = (int(x + offset_x), int(y + offset_y))
        bbox = self._clamp_bbox((center[0] - r, center[1] - r, 2 * r, 2 * r), frame.shape)

        result.update(
            {
                "found": True,
                "center": center,
                "radius": int(r),
                "bbox": bbox,
                "image": search
            }
        )
        return result

    def draw_detection(self, frame, mask,  color=(0, 255, 0), thickness=2):
        """
        Draw the detected circle and bounding box onto the frame.

        Args:
            frame: BGR frame to draw on (modified in-place).
            mask: Motion mask used to gate detection.
            color: BGR color tuple for drawing.
            thickness: Line thickness for circle and box.

        Returns:
            The same frame with annotations applied.
        """
        detection = self._circle_detection(frame, mask)

        if not detection or not detection.get("found"):
            return frame
        center = detection["center"]
        radius = detection["radius"]
        bbox = detection["bbox"]
        cv2.circle(frame, center, radius, color, thickness)
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        return (frame, detection["image"])
