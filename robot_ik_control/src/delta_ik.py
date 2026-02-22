# delta_ik.py
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Tuple


# ---- Constants
SIN120 = math.sqrt(3.0) / 2.0
COS120 = -0.5
PI = math.pi


# Delta Robot Object
@dataclass
class DeltaRobot:
    """
    All lengths are in mm.
    base_radius : distance from origin to upper arm joint
    ee_radius   : distance from end-effector origin to lower arm joint
    rf : upper arm length (shoulder to elbow)
    re : lower arm length (elbow to wrist)
    """
    base_radius: float
    ee_radius: float
    upper_arm_length: float
    lower_arm_length: float

    # Optional joint angle limits (degrees), used for sanity checks
    theta_min_deg: float = -360.0
    theta_max_deg: float = 360.0


class IKError(ValueError):
    pass


def _delta_calcAngleYZ(robot: DeltaRobot, x0: float, y0: float, z0: float) -> Tuple[int, float]:
    """
    Helper: calculates theta for one arm in the YZ plane.

    Parameters:
        geom : Delta Robot Objects
        x0, y0, z0 : target point coordinates

    Returns:
      (status, theta_deg)
        status is whether a solution exists: 0 yes, -1 non-existing position
    """
    base_radius = robot.base_radius
    ee_radius = robot.ee_radius
    upper_arm_length = robot.upper_arm_length
    lower_arm_length = robot.lower_arm_length

    # Avoid divide-by-zero if z0 is 0
    if abs(z0) < 1:     # if less than 1 mm, treat as not possible to reach
        return -1, float("nan")

    # Upper and Lower Joint Locations
    y1 = -base_radius
    y0 = y0 - ee_radius

    # Combined sphere and circle intersection expressions
    a = (x0 * x0 + y0 * y0 + z0 * z0 + upper_arm_length * upper_arm_length - lower_arm_length * lower_arm_length - y1 * y1) / (2.0 * z0)    
    b = (y1 - y0) / z0

    # Discriminant: determine how many solutions exist
    d = -(a + b * y1) ** 2 + upper_arm_length * (b * b * upper_arm_length + upper_arm_length)

    # No solution exists
    if d < 0:
        return -1, float("nan")

    # Choose "elbow-down" configuration (common choice)
    yj = (y1 - a * b - math.sqrt(d)) / (b * b + 1.0)
    zj = a + b * yj

    # Compute angle
    theta = -PI - math.atan2(zj, (yj - y1))  # atan2 works in all quadrants

    theta_deg = math.degrees(theta)
    return 0, theta_deg


def inverse_kinematics(
    robot: DeltaRobot,
    x0: float,
    y0: float,
    z0: float,
    *,
    check_limits: bool = True,
) -> Tuple[float, float, float]:
    """
    Inverse kinematics:
      (x0, y0, z0) -> (theta1, theta2, theta3) in DEGREES

    Coordinate convention:
      - Origin at base center
      - x,y in base plane
      - z is vertical (typically negative for "down")

    Raises IKError if point is unreachable (or violates limits if enabled).
    """
    # Arm 1 (no rotation)
    s1, t1 = _delta_calcAngleYZ(robot, x0, y0, z0)
    if s1 != 0:
        raise IKError("Unreachable (arm 1)")

    # Arm 2: rotate coords +120 deg
    x1 = x0 * COS120 - y0 * SIN120
    y1 = y0 * COS120 + x0 * SIN120
    s2, t2 = _delta_calcAngleYZ(robot, x1, y1, z0)
    if s2 != 0:
        raise IKError("Unreachable (arm 2)")

    # Arm 3: rotate coords -120 deg
    x2 = x0 * COS120 - y0 * -SIN120
    y2 = y0 * COS120 + x0 * -SIN120
    s3, t3 = _delta_calcAngleYZ(robot, x2, y2, z0)
    if s3 != 0:
        raise IKError("Unreachable (arm 3)")

    if check_limits:
        lo, hi = robot.theta_min_deg, robot.theta_max_deg
        for i, th in enumerate((t1, t2, t3), start=1):
            if not (lo <= th <= hi):
                raise IKError(f"Joint {i} angle {th:.2f}° out of limits [{lo},{hi}]")

    return t1, t2, t3


if __name__ == "__main__":
    geom = DeltaRobot(base_radius=165.0, ee_radius=50.0, upper_arm_length=350.0, lower_arm_length=1000.0)
    test_pt = (0.0, 0.0, -1000.740)
    print("Geometry:", geom)
    print("Test point:", test_pt)
    print("Thetas (deg):", inverse_kinematics(geom, *test_pt))
