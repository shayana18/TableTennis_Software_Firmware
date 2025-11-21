"""
Stereo Triangulation

Converts 2D points from two cameras into 3D world coordinates.
Uses the Direct Linear Transform (DLT) method.
"""

import cv2
import numpy as np


def triangulate_point(pt_left, pt_right, K_left, K_right, R, T):
    """
    Calculate 3D position from corresponding 2D points in stereo cameras.

    How it works:
    1. Each 2D point defines a ray from that camera into 3D space
    2. With two cameras, we have two rays
    3. The 3D point is where these rays intersect (or closest point)

    Args:
        pt_left: (x, y) pixel coordinates in left camera
        pt_right: (x, y) pixel coordinates in right camera
        K_left: 3x3 intrinsic matrix of left camera
        K_right: 3x3 intrinsic matrix of right camera
        R: 3x3 rotation matrix (right camera relative to left)
        T: 3x1 translation vector (right camera relative to left) in mm

    Returns:
        (X, Y, Z) position in 3D space (mm), or None if failed
    """
    # Build projection matrices
    # Left camera is at origin: P1 = K1 * [I | 0]
    # Right camera is offset: P2 = K2 * [R | T]
    P1 = K_left @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K_right @ np.hstack([R, T.reshape(3, 1)])

    # Convert points to the format OpenCV expects
    pts_left = np.array([[pt_left[0], pt_left[1]]], dtype=np.float64)
    pts_right = np.array([[pt_right[0], pt_right[1]]], dtype=np.float64)

    # Triangulate using OpenCV's DLT method
    # Returns 4D homogeneous coordinates
    points_4d = cv2.triangulatePoints(P1, P2, pts_left.T, pts_right.T)

    # Convert to 3D by dividing by w
    if points_4d[3, 0] == 0:
        return None

    X = points_4d[0, 0] / points_4d[3, 0]
    Y = points_4d[1, 0] / points_4d[3, 0]
    Z = points_4d[2, 0] / points_4d[3, 0]

    return (X, Y, Z)


def compute_disparity(pt_left, pt_right):
    """
    Calculate horizontal pixel difference between matched points.

    Disparity is inversely proportional to depth:
    - Large disparity = close object
    - Small disparity = far object

    Args:
        pt_left: (x, y) in left camera
        pt_right: (x, y) in right camera

    Returns:
        Disparity in pixels (always positive)
    """
    return abs(pt_left[0] - pt_right[0])


def estimate_depth_from_disparity(disparity, focal_length, baseline_mm):
    """
    Quick depth estimate using disparity.

    Formula: Z = (f * B) / d
    where f = focal length (pixels), B = baseline (mm), d = disparity (pixels)

    Args:
        disparity: Pixel difference between left and right points
        focal_length: Camera focal length in pixels
        baseline_mm: Distance between cameras in mm

    Returns:
        Estimated depth in mm, or None if disparity is too small
    """
    if disparity < 1:
        return None

    return (focal_length * baseline_mm) / disparity


def validate_stereo_match(pt_left, pt_right, max_y_diff=30):
    """
    Check if two detections are likely the same ball.

    For rectified stereo cameras, matching points should have similar Y coordinates.
    Even for non-rectified cameras, Y difference shouldn't be huge.

    Args:
        pt_left: (x, y) in left camera
        pt_right: (x, y) in right camera
        max_y_diff: Maximum allowed Y difference in pixels

    Returns:
        True if points are likely a valid match
    """
    y_diff = abs(pt_left[1] - pt_right[1])
    return y_diff <= max_y_diff


def undistort_point(point, K, dist_coeffs):
    """
    Remove lens distortion from a single point.

    Args:
        point: (x, y) distorted pixel coordinates
        K: 3x3 camera matrix
        dist_coeffs: Distortion coefficients

    Returns:
        (x, y) undistorted coordinates
    """
    pts = np.array([[point]], dtype=np.float64)
    undistorted = cv2.undistortPoints(pts, K, dist_coeffs, P=K)
    return (undistorted[0, 0, 0], undistorted[0, 0, 1])
