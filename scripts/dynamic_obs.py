#!/usr/bin/env python3

import os
import time
import rospy
import numpy as np

from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker


def main():
    rospy.init_node("dyna_obs_node")

    viz_pub = rospy.Publisher("obs_viz", Marker, queue_size=1)
    odom_pub = rospy.Publisher("obs_odom", Odometry, queue_size=1)

    freq = 50.0
    dt = 1.0 / freq
    rate = rospy.Rate(freq)

    viz_msg = Marker()
    viz_msg.header.frame_id = "map"
    viz_msg.type = Marker.SPHERE
    viz_msg.ns = "obs"
    viz_msg.id = 9081
    viz_msg.action = Marker.ADD
    viz_msg.scale.x = 0.4
    viz_msg.scale.y = 0.4
    viz_msg.scale.z = 0.4
    viz_msg.color.a = 1.0
    viz_msg.color.g = 1.0

    # pos = np.array([-3.3, 4.0, 0.0])
    # vel = np.array([0.3, -0.1, 0.0])
    pos = np.array([-2.05, 5.6, 0.0])
    vel = np.array([0.0, -0.2, 0.0])

    while not rospy.is_shutdown():
        # Update the position of the sphere
        pos += vel * dt

        viz_msg.header.stamp = rospy.Time.now()
        viz_msg.pose.position.x = pos[0]
        viz_msg.pose.position.y = pos[1]
        viz_msg.pose.position.z = pos[2]

        # Create a new Odometry message
        odom_msg = Odometry()
        odom_msg.header.frame_id = "map"
        odom_msg.header.stamp = rospy.Time.now()
        odom_msg.pose.pose.position.x = pos[0]
        odom_msg.pose.pose.position.y = pos[1]
        odom_msg.pose.pose.position.z = pos[2]
        odom_msg.twist.twist.linear.x = vel[0]
        odom_msg.twist.twist.linear.y = vel[1]
        odom_msg.twist.twist.linear.z = vel[2]

        # Publish the messages
        viz_pub.publish(viz_msg)
        odom_pub.publish(odom_msg)

        rate.sleep()


if __name__ == "__main__":
    main()
