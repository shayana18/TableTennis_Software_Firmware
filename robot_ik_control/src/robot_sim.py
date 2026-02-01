"""
Delta Robot Simulation 
IGEN 430
Rocky Cao, Jan 26, 2026

This program visualizes/simulates the movement of the delta robot.
"""
from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from delta_ik import DeltaRobot, inverse_kinematics, IKError

SQRT3 = math.sqrt(3.0)
SIN120 = SQRT3 / 2.0
COS120 = -0.5


def rotz(v: np.ndarray, deg: float) -> np.ndarray:
    """Rotate a 3D vector about Z by deg."""
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    x, y, z = v
    return np.array([c * x - s * y, s * x + c * y, z], dtype=float) 


def rotz_inv(v: np.ndarray, deg: float) -> np.ndarray:
    """Inverse rotation about Z by deg."""
    return rotz(v, -deg)


def base_joints(robot: DeltaRobot) -> np.ndarray:
    """Base shoulder joint centers (3x3)."""
    y1 = robot.base_radius
    b1 = np.array([0.0, y1, 0.0])
    b2 = rotz(b1, 120.0)
    b3 = rotz(b1, -120.0)
    return np.vstack([b1, b2, b3])


def effector_joints(robot: DeltaRobot, target: np.ndarray) -> np.ndarray:
    """End-effector attachment points (3x3) in global coordinates."""
    y0 = robot.ee_radius
    e1 = target + np.array([0.0, y0, 0.0])
    e2 = target + rotz(np.array([0.0, y0, 0.0]), 120.0)
    e3 = target + rotz(np.array([0.0, y0, 0.0]), -120.0)
    return np.vstack([e1, e2, e3])


def elbow_points(robot: DeltaRobot, thetas_deg: np.ndarray) -> np.ndarray:
    """
    Compute elbow points for each arm (3x3).
    Angle convention: 0° = arm horizontal (pointing in +y direction).
    Negative angle = arm rotates downward (more negative z).
    """
    base_rad = robot.base_radius
    upper_arm_l = robot.upper_arm_length

    elbows = []
    for arm_idx, th_deg in enumerate(thetas_deg):
        th = math.radians(float(th_deg))
        # Horizontal is 0°; negative angle rotates downward
        yL = base_rad + upper_arm_l * math.cos(th)
        zL = upper_arm_l * math.sin(th)
        eL = np.array([0.0, yL, zL], dtype=float)

        if arm_idx == 0:
            elbows.append(eL)
        elif arm_idx == 1:
            elbows.append(rotz_inv(eL, -120.0))
        else:
            elbows.append(rotz_inv(eL, 120.0))

    return np.vstack(elbows)


def build_square_path(center, half, z, steps_per_edge):
    """Build a square path for the end-effector to follow."""
    cx, cy = center
    corners = [
        (cx - half, cy - half, z),
        (cx + half, cy - half, z),
        (cx + half, cy + half, z),
        (cx - half, cy + half, z),
    ]
    path = []
    for i in range(len(corners)):
        p0 = np.array(corners[i], dtype=float)
        p1 = np.array(corners[(i + 1) % len(corners)], dtype=float)
        for a in np.linspace(0.0, 1.0, steps_per_edge, endpoint=False):
            path.append((1 - a) * p0 + a * p1)
    path.append(np.array(corners[0], dtype=float))
    return np.array(path)


def set_line(line, p0, p1):
    """Update a 3D line from p0 to p1."""
    line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
    line.set_3d_properties([p0[2], p1[2]])


def main():
    robot = DeltaRobot(
        base_radius=165.0,
        ee_radius=50.0,
        upper_arm_length=350.0,
        lower_arm_length=1000.0,
        theta_min_deg=-360.0,
        theta_max_deg=360.0,
    )

    path = build_square_path(center=(0, 0), half=100.0, z=-1100.0, steps_per_edge=35)

    # Compute IK for all path points
    thetas = []
    reachable = []
    for p in path:
        try:
            t = inverse_kinematics(robot, float(p[0]), float(p[1]), float(p[2]))
            thetas.append(t)
            reachable.append(True)
        except IKError as e:
            thetas.append((math.nan, math.nan, math.nan))
            reachable.append(False)

    thetas = np.array(thetas, dtype=float)
    reachable = np.array(reachable, dtype=bool)
    
    # Debug: print reachability
    print(f"Total points: {len(path)}, Reachable: {np.sum(reachable)}, Unreachable: {len(path) - np.sum(reachable)}")
    if np.sum(reachable) > 0:
        print(f"Sample angles (first reachable): {thetas[np.where(reachable)[0][0]]}")

    # 3D animation
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Delta Robot Animation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Static base
    B = base_joints(robot)
    base_loop = np.vstack([B, B[0]])
    ax.plot(base_loop[:, 0], base_loop[:, 1], base_loop[:, 2], 'k-', linewidth=2, label="Base")

    # Arm lines
    upper_lines = [ax.plot([], [], [], 'b-', linewidth=2)[0] for _ in range(3)]
    lower_lines = [ax.plot([], [], [], 'r-', linewidth=2)[0] for _ in range(3)]
    ee_tri, = ax.plot([], [], [], 'g-', linewidth=2, label="End-Effector")

    # Set axes bounds
    pad = 800
    ax.set_xlim(np.min(path[:, 0]) - pad, np.max(path[:, 0]) + pad)
    ax.set_ylim(np.min(path[:, 1]) - pad, np.max(path[:, 1]) + pad)
    ax.set_zlim(np.min(path[:, 2]) - pad, 0 + pad)
    ax.legend()

    def update(i):
        if not reachable[i]:
            return upper_lines + lower_lines + [ee_tri]

        p = path[i]
        th = thetas[i]
        B = base_joints(robot)
        E = elbow_points(robot, th)
        P = effector_joints(robot, p)
        #print("Elbow points:", E)

        for k in range(3):
            set_line(upper_lines[k], B[k], E[k])
            set_line(lower_lines[k], E[k], P[k])

        loop = np.vstack([P, P[0]])
        ee_tri.set_data(loop[:, 0], loop[:, 1])
        ee_tri.set_3d_properties(loop[:, 2])

        return upper_lines + lower_lines + [ee_tri]

    anim = FuncAnimation(fig, update, frames=range(0, len(path), 2), interval=5, blit=False, repeat=True)
    plt.show()


if __name__ == "__main__":
    main()
