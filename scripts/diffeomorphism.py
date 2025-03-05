# #!/usr/bin/env python3
#
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def convert_pcld_diffeomorphism(points, is_odom=False):
#     """
#     Apply the transformation to warp star-shaped free space into a 2D ball.
#     """
#
#     # Compute distance from robot to each point in free space
#     dists = np.linalg.norm(points, axis=1)
#
#     ret = []
#     for i, dist in enumerate(dists):
#         val = points[i] / dist
#         if is_odom:
#             d = dist - 1
#             val *= np.sqrt(1 - d)
#
#         ret.append(val)
#
#     ret = np.array(ret)
#
#     return ret
#
#
# def main():
#     NUM_SAMPLES = 10
#     MAX_RANGE = 5
#
#     thetas = np.linspace(0, 2 * np.pi, NUM_SAMPLES)
#
#     ranges = np.random.rand(NUM_SAMPLES) * MAX_RANGE
#
#     points = np.array([ranges * np.cos(thetas), ranges * np.sin(thetas)]).T
#
#     transformed_points = convert_pcld_diffeomorphism(points)
#
#     points = np.concatenate([points, [points[0]]], axis=0)
#
#     odom = np.expand_dims(np.random.rand(2) * 1.5, axis=0)
#     transformed_odom = convert_pcld_diffeomorphism(odom, is_odom=True)
#
#     fig, ax = plt.subplots()
#     ax.plot(points[:, 0], points[:, 1], c="b")
#     ax.scatter(transformed_points[:, 0], transformed_points[:, 1], c="r")
#
#     ax.scatter(odom[:, 0], odom[:, 1], c="g", marker="x")
#     ax.scatter(transformed_odom[:, 0], transformed_odom[:, 1], c="g")
#
#     # plot circle
#     circle = plt.Circle((0, 0), 1, color="b", fill=False, alpha=0.5)
#
#     ax.add_patch(circle)
#
#     ax.set_aspect("equal")
#     plt.show()
#
#
# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


def compute_boundary_lookup(points):
    """
    Builds a lookup table for R(q) by storing the maximum boundary distance at each angle.
    """
    angles = np.arctan2(points[:, 1], points[:, 0])  # Compute angles
    ranges = np.linalg.norm(points, axis=1)  # Compute distances

    # Create a dictionary to store the max range per unique angle
    angle_to_range = {}
    for ang, r in zip(angles, ranges):
        if ang not in angle_to_range or r > angle_to_range[ang]:
            angle_to_range[ang] = r

    return angle_to_range


def find_nearest_boundary_radius(q, angle_to_range):
    """
    Given a point q, find the closest boundary radius R(q).
    """
    theta = np.arctan2(q[1], q[0])

    # Find the closest angle in the lookup table
    closest_theta = min(angle_to_range.keys(), key=lambda a: abs(a - theta))

    return angle_to_range[closest_theta]  # Return the corresponding R(q)


def beta_function(q, angle_to_range):
    """
    Computes β(q), where:
      - β(q) < 0 inside the star set
      - β(q) = 0 on the boundary
      - β(q) > 0 outside
    """
    R_q = find_nearest_boundary_radius(q, angle_to_range)
    return np.linalg.norm(q) / R_q - 1  # Normalized implicit function


def convert_pcld_diffeomorphism(points, angle_to_range, is_odom=False):
    """
    Apply diffeomorphic transformation using dynamic R(q).
    """
    transformed_points = []

    for q in points:
        beta = beta_function(q, angle_to_range)

        scale = 1
        if is_odom:
            d = beta
            scale = np.sqrt(1 + d)  # Smooth transition

        transformed_q = scale * q / np.linalg.norm(q)
        transformed_points.append(transformed_q)

    return np.array(transformed_points)


def main():
    NUM_SAMPLES = 729
    MAX_RANGE = 2

    # Generate LIDAR-like points
    thetas = np.linspace(0, 2 * np.pi, NUM_SAMPLES)
    ranges = (
        np.random.rand(NUM_SAMPLES) * MAX_RANGE + 0.1
    )  # Simulating varying lidar returns
    points = np.array([ranges * np.cos(thetas), ranges * np.sin(thetas)]).T

    # Compute boundary lookup
    angle_to_range = compute_boundary_lookup(points)

    # Apply transformation
    transformed_points = convert_pcld_diffeomorphism(points, angle_to_range)

    # Generate odometry point inside the star set
    # pick random range between 0 and 2
    odom_dist = np.random.rand() * 2
    # pick random angle in thetas
    ind = np.random.randint(NUM_SAMPLES)
    odom_angle = thetas[ind]
    odom = np.array(
        [odom_dist * np.cos(odom_angle), odom_dist * np.sin(odom_angle)]
    ).reshape(1, 2)

    odom_color = "g"
    if ranges[ind] < odom_dist:
        odom_color = "r"

    transformed_odom = convert_pcld_diffeomorphism(odom, angle_to_range, is_odom=True)

    # Plot original and transformed points
    fig, ax = plt.subplots()
    ax.plot(
        points[:, 0], points[:, 1], c="gray", alpha=0.3, label="Star-Shaped Boundary"
    )
    # ax.scatter(
    #     transformed_points[:, 0], transformed_points[:, 1], c="r", label="Mapped Points",
    # )

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
