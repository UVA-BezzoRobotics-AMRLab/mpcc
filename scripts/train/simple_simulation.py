import rospy
import numpy as np

from std_srvs.srv import Empty
from cbf_tracking.srv import GetState
from geometry_msgs.msg import Quaternion
from std_msgs.msg import Bool


class SimpleSimulation:

    def __init__(self, collision_topic="/collision"):
        self._model_state = rospy.ServiceProxy("/get_state", GetState)

        self.collision_count = 0
        self._collision_sub = rospy.Subscriber(
            collision_topic, Bool, self.collision_monitor
        )

    def collision_monitor(self, msg):
        if msg.data:
            self.collision_count += 1

    def get_hard_collision(self):
        # hard collision count since last call
        collided = self.collision_count > 0
        self.collision_count = 0
        return collided

    def get_model_state(self):
        rospy.wait_for_service("/get_state")
        try:
            return self._model_state()
        except rospy.ServiceException:
            rospy.logwarn("/gazebo/get_model_state service call failed")
