#!/usr/bin/env python3

import rospy
import numpy as np
import cvxpy as cp
import casadi as ca
from scipy.interpolate import CubicSpline, make_interp_spline

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker


class DiffeoCBF:
    def __init__(self):

        self.ranges = None
        self.angles = None

        self.odom = None
        self.range_coeffs = None

        rospy.Subscriber("/front/scan", LaserScan, self.laser_cb)
        rospy.Subscriber("/gmapping/odometry", Odometry, self.odom_cb)
        rospy.Timer(rospy.Duration(0.1), self.timer_cb)

        self.pub_marker = rospy.Publisher("/smoothed_lidar", Marker, queue_size=1)

    def timer_cb(self, event):
        if self.angles is None or self.ranges is None or self.odom is None:
            print("waiting for data")
            return

        self.interpolate_ranges()
        self.compute_lie_derivatives()

        opti = ca.Opti()
        u = opti.variable(2)

        u_des = [0.7, 0.1]
        obj = ca.mtimes((u - u_des).T, u - u_des)

        cons = self.Lfh_fn(self.odom) + self.Lgh_fn(self.odom) @ u >= -5 * self.h_fn(
            self.odom
        )

        opti.minimize(obj)
        opti.subject_to(u[0] >= -1)  # v >= -1
        opti.subject_to(u[0] <= 1)  # v <= 1
        opti.subject_to(u[1] >= 0.5)  # omega >= -1
        opti.subject_to(u[1] <= 0.5)  # omega <= 1
        opti.subject_to(cons)
        opti.solver("ipopt")

        opti.solve()

        print(opti.value(u))

    def publish_smoothed_lidar(self):
        """
        Takes a LaserScan message, smooths it, and publishes the interpolated points as a Marker in RViz.
        """

        # Interpolate R(theta) using cubic B-spline
        smoothed_ranges = self.interpolate_ranges()

        if self.range_coeffs is None:
            return

        # Create Marker for visualization
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "smoothed_lidar"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.05  # Point size
        marker.scale.y = 0.05
        marker.color.a = 1.0
        marker.color.r = 1.0  # Red points for interpolated values

        # Convert polar (angle, radius) -> Cartesian (x, y)
        for theta in self.angles:
            r = smoothed_ranges(theta, self.range_coeffs)
            x, y = r * np.cos(theta), r * np.sin(theta)
            marker.points.append(Point(x, y, 0))

        # Publish Marker
        self.pub_marker.publish(marker)

    def laser_cb(self, msg):

        self.laser_msg = msg
        self.ranges = np.array(msg.ranges)
        self.ranges[self.ranges > msg.range_max] = msg.range_max

        self.angles = np.linspace(msg.angle_min, msg.angle_max, len(self.ranges))

        if self.odom is not None:
            self.publish_smoothed_lidar()

    def odom_cb(self, msg):

        def euler_from_quaterion(quaternion):
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

        self.odom = np.array(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                euler_from_quaterion(msg.pose.pose.orientation)[2],
            ]
        )

    def interpolate_ranges(self):
        """
        Interpolates the laser ranges using B-splines in CasADi to ensure smoothness and differentiability.
        """
        spline = make_interp_spline(self.angles + self.odom[2], self.ranges)

        # Step 2: Extract the knots and coefficients
        knots = spline.t
        self.range_coeffs = spline.c

        # Step 3: Define symbolic variables for spline coefficients using CasADi
        self.r_coeffs = ca.MX.sym("range_coeffs", self.range_coeffs.shape[0])

        # Step 4: Create B-spline representation using CasADi
        self.v = ca.MX.sym("v")
        self.laser_spline_mx = ca.bspline(
            self.v, self.r_coeffs, [list(knots)], [3], 1, {}
        )

        self.laser_spline = ca.Function(
            "yr", [self.v, self.r_coeffs], [self.laser_spline_mx], {}
        )

        # Step 5: Return the spline
        return self.laser_spline

    def compute_lie_derivatives(self):
        """
        Computes the Lie derivatives Lfh and Lgh for the CBF formulation.
        """
        # Define the symbolic variables for the control inputs and state
        self.robot_dynamics()

        self.vx = self.x1 + ca.cos(self.theta1) * 0.1
        self.vy = self.y1 + ca.sin(self.theta1) * 0.1

        self.xy_atan = ca.atan2(self.vx, self.vy)
        self.beta = 1e-6 / self.laser_spline(self.xy_atan, self.range_coeffs) - 1
        self.diffeo_tran_x = ca.sqrt(1 + self.beta) * self.vx / 1e-6
        self.diffeo_tran_y = ca.sqrt(1 + self.beta) * self.vy / 1e-6

        self.h = 1 - ca.sqrt(self.diffeo_tran_x**2 + self.diffeo_tran_y**2)

        # Compute Lfh (Lie derivative of h with respect to f, robot dynamics)
        self.Lfh = ca.jacobian(self.h, self.x) @ self.f

        # Compute Lgh (Lie derivative of h with respect to g, control input)
        self.Lgh = ca.jacobian(self.h, self.x) @ self.g

        self.h_fn = ca.Function("h", [self.x], [self.h], {})
        self.Lfh_fn = ca.Function("Lfh", [self.x], [self.Lfh], {})
        self.Lgh_fn = ca.Function("Lgh", [self.x], [self.Lgh], {})

    def robot_dynamics(self):

        self.x1 = ca.MX.sym("x1")
        self.y1 = ca.MX.sym("y1")
        self.theta1 = ca.MX.sym("theta1")

        self.x = ca.vertcat(self.x1, self.y1, self.theta1)

        self.v1 = ca.MX.sym("v1")
        self.w1 = ca.MX.sym("w1")

        self.u = ca.vertcat(self.v1, self.w1)

        self.f = ca.vertcat(0, 0, 0)
        self.g = ca.vertcat(
            ca.horzcat(ca.cos(self.theta1), 0),
            ca.horzcat(ca.sin(self.theta1), 0),
            ca.horzcat(0, 1),
        )


if __name__ == "__main__":
    rospy.init_node("diffeo_cbf_node")
    diff = DiffeoCBF()
    rospy.spin()
