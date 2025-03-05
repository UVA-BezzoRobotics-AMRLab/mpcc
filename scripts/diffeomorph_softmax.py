#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


def soft_argmin_boundary(angles, ranges, query_angle, gamma=50):
    """
    Computes a smooth boundary radius using a soft-argmin weighted average.
    This avoids sharp discontinuities in the lookup table.
    """
    weights = np.exp(-gamma * (angles - query_angle) ** 2)
    weights /= np.sum(weights)
    return np.sum(weights * ranges)


def compute_boundary_lookup(points):
    """
    Precomputes angle-to-range mappings from point cloud.
    Uses soft interpolation for smooth lookup.
    """
    angles = np.arctan2(points[:, 1], points[:, 0])
    ranges = np.linalg.norm(points, axis=1)

    sorted_indices = np.argsort(angles)
    return angles[sorted_indices], ranges[sorted_indices]


def find_smooth_boundary_radius(q, angles, ranges):
    """
    Finds a differentiable approximation of R(q) using soft-argmin interpolation.
    """
    theta = np.arctan2(q[1], q[0])
    return soft_argmin_boundary(angles, ranges, theta)


def beta_function(q, angles, ranges):
    """
    Computes β(q), where:
      - β(q) < 0 inside the star set
      - β(q) = 0 on the boundary
      - β(q) > 0 outside
    """
    R_q = find_smooth_boundary_radius(q, angles, ranges)
    return np.linalg.norm(q) / R_q - 1


def convert_pcld_diffeomorphism(points, angles, ranges, is_odom=False):
    """
    Apply diffeomorphic transformation using smooth R(q).
    """
    transformed_points = []
    for q in points:
        beta = beta_function(q, angles, ranges)
        scale = np.sqrt(1 + beta) if is_odom else 1
        transformed_q = scale * q / np.linalg.norm(q)
        transformed_points.append(transformed_q)
    return np.array(transformed_points)


def main():
    NUM_SAMPLES = 729
    MAX_RANGE = 2

    # Generate LIDAR-like points
    thetas = np.linspace(0, 2 * np.pi, NUM_SAMPLES)
    ranges = np.random.rand(NUM_SAMPLES) * MAX_RANGE + 0.5  # Simulated LiDAR data
    points = np.column_stack((ranges * np.cos(thetas), ranges * np.sin(thetas)))

    # Compute smooth boundary lookup
    angles, smooth_ranges = compute_boundary_lookup(points)

    # Apply transformation
    transformed_points = convert_pcld_diffeomorphism(points, angles, smooth_ranges)

    # Generate random odometry point
    odom_dist = np.random.rand() * 2
    odom_angle = thetas[np.random.randint(NUM_SAMPLES)]
    odom = np.array([[odom_dist * np.cos(odom_angle), odom_dist * np.sin(odom_angle)]])

    odom_color = (
        "g"
        if find_smooth_boundary_radius(odom[0], angles, smooth_ranges) >= odom_dist
        else "r"
    )
    transformed_odom = convert_pcld_diffeomorphism(
        odom, angles, smooth_ranges, is_odom=True
    )

    # Plot original and transformed points
    fig, ax = plt.subplots()
    # ax.plot(
    #     points[:, 0], points[:, 1], c="gray", alpha=0.3, label="Star-Shaped Boundary"
    # )

    # Smoothed boundary plot
    smoothed_boundary_x = smooth_ranges * np.cos(angles)
    smoothed_boundary_y = smooth_ranges * np.sin(angles)
    ax.plot(
        smoothed_boundary_x,
        smoothed_boundary_y,
        "b",
        alpha=0.5,
        label="Smooth Boundary",
    )

    # Odometry visualization
    ax.scatter(odom[:, 0], odom[:, 1], c=odom_color, marker="x", label="Odom Position")
    ax.scatter(
        transformed_odom[:, 0],
        transformed_odom[:, 1],
        c=odom_color,
        label="Mapped Odom",
    )

    # Plot unit circle
    circle = plt.Circle((0, 0), 1, color="b", fill=False, alpha=0.5, linestyle="dashed")
    ax.add_patch(circle)

    ax.set_aspect("equal")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
