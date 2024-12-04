#! /usr/bin/env python3

import rospy
import shapely

from visualization_msgs.msg import Marker


class TubeGen:
    def __init__(self):

        self.corridor_sub = rospy.Subscriber(
            "/visualizer/edge", Marker, self.corridor_cb
        )

        self.corridor = None

    def corridor_cb(self, msg):

        self.corridor = msg.points
        print(self.corridor)

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    rospy.init_node("tube_generator")

    tg = TubeGen()
    tg.spin()
