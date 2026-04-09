"""
Workspace constants and boundary checks for the delta robot.

All values in robot frame (mm). The workspace is an elliptical cylinder:
  XY: ellipse with semi-axes ELLIPSE_A x ELLIPSE_B
  Z:  bounded by Z_MIN .. Z_MAX (end-effector hangs below base plate)

These constants match the firmware (robot.h) with safety margins applied.
"""

import math

# Workspace -- firmware ellipse with safety margin to avoid IK rejections
ELLIPSE_A      = 720    # mm X semi-axis (711.0 after 10% margin)
ELLIPSE_B      = 470.0    # mm Y semi-axis (486.0 after 10% margin)
Z_MIN          = -1020.0        # mm (-1025, 25mm margin from robot.h limit)
Z_MAX          = -800.0         # mm (-731, 10mm margin)
MAX_CLAMP_DIST = 100         # mm -- max distance to clamp to workspace

ROBOT_HOME     = (0.0, 0.0, -900.0)
MAX_CART_VEL   = 4000.0         # mm/s
MAX_CART_ACC   = 20000.0        # mm/s^2

CM_TO_MM       = 10.0

GRAVITY_Z      = -9810.0        # mm/s^2, robot Z is vertical, negative = down

# Air drag for ping pong ball (mass=2.7g, diameter=40mm, Cd=0.445)
# k = 0.5 * Cd * rho_air * A / m  (in mm^-1)
#   = 0.5 * 0.445 * 1.2e-9 * pi*20^2 / 0.0027 = 1.243e-4
DRAG_K         = 0.000180       # mm^-1 -- drag deceleration: a = -DRAG_K * |v| * v

Z_TABLE_SURFACE   = -1120.0    # mm, table top in robot frame (measured from rolling ball test)
RESTITUTION_COEFF = 0.99       # vz damping on bounce (ping pong on hard table)
FRICTION_COEFF    = 0.99       # vx/vy damping on bounce (tangential friction)
MAX_BOUNCES       = 2          # max bounces in prediction scan


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
