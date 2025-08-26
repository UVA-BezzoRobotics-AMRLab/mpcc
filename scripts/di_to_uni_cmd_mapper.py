#!/usr/bin/env python3

import rospy
import numpy as np

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan


class CmdMapper:
    def __init__(self):

        rospy.Subscriber("/gmapping/odometry", Odometry, self.odom_cb)
        rospy.Subscriber("/mpc_vel", Twist, self.mpc_cb)

        self.vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        self.odom = None

    def mpc_cb(self, msg):
        if self.odom is None:
            return

        theta_v = np.arctan2(msg.linear.y, msg.linear.x)
        error = theta_v - self.odom[2]

        # bound to -pi and pi
        error = np.arctan2(np.sin(error), np.cos(error))

        kp = 5.0
        vel_msg = Twist()

        # if error is too high, turn in place (20 degrees threshold)
        v = np.linalg.norm([msg.linear.x, msg.linear.y])
        if np.abs(error) > np.pi / 9:
            vel_msg.linear.x = 0
        else:
            vel_msg.linear.x = v

        if v > 1e-2:
            vel_msg.angular.z = kp * error

        self.vel_pub.publish(vel_msg)

    def odom_cb(self, msg):
        q = msg.pose.pose.orientation
        (_, _, yaw) = self.euler_from_quaternion(q)
        self.odom = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, yaw])

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


if __name__ == "__main__":
    rospy.init_node("cmd_mapper", anonymous=True)

    node = CmdMapper()
    rospy.spin()
