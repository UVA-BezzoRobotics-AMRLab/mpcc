#!/usr/bin/env python3

import rospy
import numpy as np
import matplotlib.pyplot as plt

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


class DiffeoNode:
    def __init__(self):

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
        self.ax1.set_xlim(-5, 5)
        self.ax1.set_ylim(-5, 5)
        self.ax1.set_aspect("equal")

        circle = plt.Circle(
            (0, 0), 1, color="b", fill=False, alpha=0.5, linestyle="dashed"
        )
        self.ax1.add_patch(circle)

        self.odom_plt = self.ax2.scatter(
            [], [], c="g", marker="x", label="Odom Position"
        )

        self.transformed_odom_plt = self.ax1.scatter(
            [], [], c="g", marker="x", label="Odom Position"
        )

        # (self.scan_data,) = self.ax.plot([], [], "b-", label="Star-Shaped Boundary")

        self.got_first_scan = False

        self.angles = None
        self.ranges = None

        self.odom = None
        self.center_star = None

        rospy.Subscriber("/front/scan", LaserScan, self.laser_cb)
        rospy.Subscriber("/gmapping/odometry", Odometry, self.odom_cb)
        rospy.Timer(rospy.Duration(0.1), self.timer_cb)

    def spin(self):
        plt.ion()
        plt.show(block=True)

    def timer_cb(self, event):
        if self.angles is None or self.ranges is None or self.odom is None:
            print("waiting for data")
            return

        # print(self.odom - self.center_star)
        transformed_odom = self.convert_pcld_diffeomorphism(
            np.array([self.odom - self.center_star]),
            is_odom=True,
        )

        self.transformed_odom_plt.set_offsets(transformed_odom)
        self.odom_plt.set_offsets(self.odom)

        self.ax2.set_xlim(self.odom[0] - 5, self.odom[0] + 5)
        self.ax2.set_ylim(self.odom[1] - 5, self.odom[1] + 5)
        self.ax2.set_aspect("equal")

        plt.draw()

    def odom_cb(self, msg):
        self.odom = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        # convert quaternion to euler angles
        q = msg.pose.pose.orientation
        (_, _, self.yaw) = self.euler_from_quaternion(q)

        if self.center_star is None:
            self.center_star = self.odom

    def laser_cb(self, msg):

        if self.got_first_scan or self.odom is None:
            return

        self.got_first_scan = True

        ranges = np.array(msg.ranges)

        # cap ranges at max range
        ranges[ranges > msg.range_max] = msg.range_max

        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        # sorted_indices = np.argsort(angles)
        # angles = angles[sorted_indices]
        # ranges = ranges[sorted_indices]

        self.angles = angles
        self.ranges = ranges

        points = np.column_stack(
            (
                self.ranges * np.cos(self.angles + self.yaw) + self.odom[0],
                self.ranges * np.sin(self.angles + self.yaw) + self.odom[1],
            )
        )

        self.ax2.plot(
            points[:, 0],
            points[:, 1],
            c="gray",
            alpha=0.3,
            label="Star-Shaped Boundary",
        )

    def euler_from_quaternion(self, quaternion):
        """
        Converts quaternion (w in last place) to euler roll, pitch, yaw
        quaternion = [x, y, z, w]
        Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
        """
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def beta_function(self, q, threshold=1.0):
        """
        Computes β(θ) based on the proximity of lidar measurements to obstacles.
        """

        # add tiny noise to avoid domain error if q = 0 vector
        q = q + (1e-6 * np.array([1, 0]))
        theta = np.arctan2(q[1], q[0])
        # find the closest angle in the lookup table
        theta_ind = np.argmin(np.abs(np.array(self.angles) - theta + self.yaw))
        print("theta_ind", theta_ind)
        print("theta:", theta, "closest:", self.angles[theta_ind] - theta + self.yaw)
        R_theta = self.ranges[theta_ind]

        print("q:", q)
        print("R_theta is:", R_theta)
        return np.linalg.norm(q) / R_theta - threshold

    def convert_pcld_diffeomorphism(self, points, is_odom=False):
        """
        Apply diffeomorphic transformation using smooth R(q).
        """
        transformed_points = []
        for q in points:
            # if np.linalg.norm(q) != 0:
            beta = self.beta_function(q)
            print("beta is", beta)
            scale = np.sqrt(1 + beta) if is_odom else 1
            transformed_q = scale * q / (np.linalg.norm(q) + 1e-6)
            # else:
            #     transformed_q = [0, 0]
            transformed_points.append(transformed_q)

        return np.array(transformed_points)


if __name__ == "__main__":
    rospy.init_node("diffeo_node", anonymous=True)

    node = DiffeoNode()
    node.spin()
