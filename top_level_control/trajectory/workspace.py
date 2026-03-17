"""
Workspace constants and boundary checks for the delta robot.

All values in robot frame (mm). The workspace is an elliptical cylinder:
  XY: ellipse with semi-axes ELLIPSE_A x ELLIPSE_B
  Z:  bounded by Z_MIN .. Z_MAX (end-effector hangs below base plate)

These constants match the firmware (robot.h) with safety margins applied.
"""

import math

# Workspace -- firmware ellipse with safety margin to avoid IK rejections
ELLIPSE_A      = 790.0 * 0.9   # mm X semi-axis (711.0 after 10% margin)
ELLIPSE_B      = 540.0 * 0.9   # mm Y semi-axis (486.0 after 10% margin)
Z_MIN          = -1050.0        # mm (-1025, 25mm margin from robot.h limit)
Z_MAX          = -720.0         # mm (-731, 10mm margin)
MAX_CLAMP_DIST = 350.0          # mm -- max distance to clamp to workspace

ROBOT_HOME     = (0.0, 0.0, -900.0)
MAX_CART_VEL   = 4000.0         # mm/s
MAX_CART_ACC   = 20000.0        # mm/s^2

CM_TO_MM       = 10.0

GRAVITY_Z      = -9810.0        # mm/s^2, robot Z is vertical, negative = down

# Air drag for ping pong ball (mass=2.7g, diameter=40mm, Cd=0.40)
# k = 0.5 * Cd * rho_air * A / m  (in mm^-1)
#   = 0.5 * 0.40 * 1.2e-9 * pi*20^2 / 0.0027 = 1.12e-4
DRAG_K         = 0.000112       # mm^-1 -- drag deceleration: a = -DRAG_K * |v| * v


def in_workspace(x, y, z):
    """Firmware-matching ellipse check."""
    return (Z_MIN <= z <= Z_MAX and
            (x / ELLIPSE_A) ** 2 + (y / ELLIPSE_B) ** 2 <= 1.0)


def clamp_to_workspace(x, y, z):
    """Clamp a point to the nearest workspace boundary. Returns (x, y, z, dist)."""
    # Clamp Z
    z_c = max(Z_MIN, min(Z_MAX, z))
    # Clamp XY to ellipse
    r = math.sqrt((x / ELLIPSE_A) ** 2 + (y / ELLIPSE_B) ** 2)
    if r > 1.0:
        x_c = x / r
        y_c = y / r
    else:
        x_c, y_c = x, y
    dist = math.sqrt((x - x_c) ** 2 + (y - y_c) ** 2 + (z - z_c) ** 2)
    return x_c, y_c, z_c, dist
