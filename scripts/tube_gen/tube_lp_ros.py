#!/usr/bin/env python

import rospy
import numpy as np
import cvxpy as cp
from cvxpygen import cpg

from scipy import interpolate
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from trajectory_msgs.msg import JointTrajectory
from geometry_msgs.msg import Point, Pose, PoseArray
from tf.transformations import euler_from_quaternion, quaternion_from_euler


# cpg.generate_code(problem, code_dir="nonneg_LS", solver="SCS")


class Solver:

    def __init__(self):
        traj_sub = rospy.Subscriber(
            "/reference_trajectory", JointTrajectory, self.traj_sub
        )
        laser_sub = rospy.Subscriber("/front/scan", LaserScan, self.laser_sub)
        odom_sub = rospy.Subscriber("/gmapping/odometry", Odometry, self.odom_sub)

        timer = rospy.Timer(
            rospy.Duration(1), self.gen_tube
        )  # Call callback every 1 second

        self.marker_pub = rospy.Publisher("/python_tube", Marker, queue_size=1)
        self.poses_pub = rospy.Publisher("/norm_poses", PoseArray, queue_size=1)

        self.odom = None
        self.traj = None
        self.lidar = None

        self.traj_len = None

        # setup LP
        # number of points in lidar
        self.m = 720
        # degree of polynomial + 1
        self.n = 7
        self.N = 100

        # arc length domain
        self.domain = cp.Parameter(2, name="Domain")

        self.x = cp.Variable(self.n)

        # cost is maximizing area of the curve on domain
        self.coeffs = 1 / np.arange(1, self.n + 1)
        self.cost = self.coeffs @ self.x * (self.domain[1] - self.domain[0])

        self.A = cp.Parameter((2 * self.N + self.m, self.n), name="A_mat")
        self.b = cp.Parameter(2 * self.N + self.m, name="b_vec")

        self.problem = cp.Problem(cp.Minimize(-self.cost), [self.A @ self.x <= self.b])

    def traj_sub(self, msg):

        N = len(msg.points)
        ss = [0] * N
        xs = [0] * N
        ys = [0] * N

        for i in range(N):
            ss[i] = msg.points[i].time_from_start.to_sec()
            xs[i] = msg.points[i].positions[0]
            ys[i] = msg.points[i].positions[1]

        self.traj = [
            interpolate.make_interp_spline(ss, xs),
            interpolate.make_interp_spline(ss, ys),
        ]

        self.traj_len = ss[-1]

    def odom_sub(self, msg):
        self.odom = msg

    def laser_sub(self, msg):
        self.lidar = msg

    def gen_tube(self, msg):

        if self.traj is None or self.lidar is None or self.odom is None:
            return

        print("extracting pointcloud data")
        # map point cloud vals to traj
        obs = self.map_pcld_to_traj()

        min_dist = np.min(obs[:, 1]) / 1.5
        # obs_cast = np.array([val if val > min_dist else min_dist for val in obs[:, 1]])

        print("setting up params")
        self.A.value = np.zeros((2 * self.N + self.m, self.n))
        self.b.value = np.zeros((2 * self.N + self.m))

        x_is = np.linspace(0, self.traj_len, self.N)
        self.A.value[: self.N, :] = np.array(
            [[-(x**k) for k in range(self.n)] for x in x_is]
        )
        self.A.value[self.N : 2 * self.N, :] = np.array(
            [[(x**k) for k in range(self.n)] for x in x_is]
        )
        self.A.value[2 * self.N :, :] = np.array(
            [[x**k for k in range(self.n)] for x in obs[:, 0]]
        )

        self.b.value[self.N : 2 * self.N] = min_dist
        # self.b.value[2 * self.N :] = obs[:, 1]
        self.b.value[2 * self.N :] = obs[:, 1]
        self.domain.value = np.array([0, self.traj_len])

        print("params setup, solving")

        self.problem.solve(solver=cp.CLARABEL, verbose=True)
        print("Solve done: ", self.problem.solver_stats.solve_time)

        if self.x.value is not None:
            res = [a for a in self.x.value]
            self.visualize_tubes(res)

    def map_pcld_to_traj(self):

        ds = np.linspace(0, self.traj_len, self.N)
        sz = len(self.lidar.ranges)

        x = self.odom.pose.pose.position.x
        y = self.odom.pose.pose.position.y
        q = [
            self.odom.pose.pose.orientation.x,
            self.odom.pose.pose.orientation.y,
            self.odom.pose.pose.orientation.z,
            self.odom.pose.pose.orientation.w,
        ]
        yaw = euler_from_quaternion(q)[2]

        ret = np.zeros((self.m, 2))
        print(self.m)
        print(ret.shape)

        for i in range(sz):

            angle = self.lidar.angle_min + i * self.lidar.angle_increment
            lidar_x = x + self.lidar.ranges[i] * np.cos(angle + yaw)
            lidar_y = y + self.lidar.ranges[i] * np.sin(angle + yaw)
            lidar_pt = np.array([lidar_x, lidar_y])

            closest_traj_pt = None
            closest_s = None
            closest_dist = 10000

            for s in ds:
                pt = np.array([self.traj[0](s), self.traj[1](s)])

                d = np.linalg.norm(lidar_pt - pt)
                if d < closest_dist:
                    closest_dist = d
                    closest_traj_pt = pt
                    closest_s = s

            d_traj = [self.traj[0].derivative(1), self.traj[1].derivative(1)]
            normal = np.array([-d_traj[1](closest_s), d_traj[0](closest_s)])
            normal /= np.linalg.norm(normal)

            # if not on same side of normal, disregard point
            if np.dot(lidar_pt - closest_traj_pt, normal) < 0:
                ret[i, 0] = 0  # closest_s
                ret[i, 1] = 100
            else:
                ret[i, 0] = closest_s
                # ret[i, 1] = np.dot((lidar_pt - closest_traj_pt), normal)
                ret[i, 1] = closest_dist  # np.linalg.norm(lidar_pt - closest_traj_pt)

        return ret

    def visualize_tubes(self, coeffs):
        msg = Marker()
        msg.header.frame_id = "map"
        msg.header.stamp = rospy.Time.now()
        msg.ns = "tube_poly_abv"
        msg.id = 62
        msg.action = Marker.ADD
        msg.type = Marker.LINE_STRIP
        msg.scale.x = 0.05
        msg.color.a = 1.0
        msg.pose.orientation.w = 1

        pose_arr = PoseArray()
        pose_arr.header.frame_id = "map"
        pose_arr.header.stamp = rospy.Time.now()

        d_traj = [self.traj[0].derivative(1), self.traj[1].derivative(1)]
        ds = np.linspace(0, self.traj_len, self.N)

        for s in ds:
            traj_pt = np.array([self.traj[0](s), self.traj[1](s)])
            traj_norm = np.array([-d_traj[1](s), d_traj[0](s)])
            traj_norm /= np.linalg.norm(traj_norm)

            norm_theta = np.arctan2(traj_norm[1], traj_norm[0])
            q = quaternion_from_euler(0, 0, norm_theta)

            norm_pose = Pose()
            norm_pose.position.x = traj_pt[0]
            norm_pose.position.y = traj_pt[1]
            norm_pose.position.z = 0.1
            norm_pose.orientation.x = q[0]
            norm_pose.orientation.y = q[1]
            norm_pose.orientation.z = q[2]
            norm_pose.orientation.w = q[3]

            pose_arr.poses.append(norm_pose)

            poly_val = self.eval_poly(s, coeffs)
            if poly_val < 0:
                print("AHHHHHH", poly_val)

            tube_pt = traj_pt + poly_val * traj_norm

            pt = Point()
            pt.x = tube_pt[0]
            pt.y = tube_pt[1]
            msg.points.append(pt)

        self.marker_pub.publish(msg)
        self.poses_pub.publish(pose_arr)

    def eval_poly(self, x, coeffs):

        ret = 0
        for k, c in enumerate(coeffs):
            ret += c * (x**k)

        return ret

    def spin(self):
        rospy.spin()


def main():
    rospy.init_node("tube_gen", anonymous=True)

    solver = Solver()
    solver.spin()


if __name__ == "__main__":
    main()
