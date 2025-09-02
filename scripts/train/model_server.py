#!/usr/bin/env python3

import os
import time
import torch
import rospy
import numpy as np
import rlkit.torch.pytorch_util as ptu

from mpcc.srv import QuerySAC, QuerySACResponse

from std_msgs.msg import Float32
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic

policy = None


class ModelServer:
    def __init__(self):

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = int(rospy.get_param("/train/hidden_dims", 256))
        self.action_dim = int(rospy.get_param("/train/action_dim", 2))
        self.state_dim = int(rospy.get_param("/train/state_dim", 12))

        self.is_eval = rospy.get_param("/train/is_eval", True)

        self.model_file = rospy.get_param("/train/model_file", "./sac_policy.pth")

        self.alpha_dot_abv_pub = rospy.Publisher("alpha_dot_abv", Float32, queue_size=1)
        self.alpha_dot_blw_pub = rospy.Publisher("alpha_dot_blw", Float32, queue_size=1)

        self.stoch_policy = TanhGaussianPolicy(
            obs_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_sizes=[self.hidden_dim, self.hidden_dim],
        ).to(ptu.device)
        self.qf1 = ConcatMlp(
            input_size=self.state_dim + self.action_dim,
            output_size=1,
            hidden_sizes=[self.hidden_dim, self.hidden_dim],
        ).to(ptu.device)
        self.qf2 = ConcatMlp(
            input_size=self.state_dim + self.action_dim,
            output_size=1,
            hidden_sizes=[self.hidden_dim, self.hidden_dim],
        ).to(ptu.device)

        rospy.logerr("*******************************")
        rospy.logerr("Model Server Initialized")
        rospy.logerr("*******************************")

        # if file exists, load the model. Otherwise use the default model
        model_dir = os.path.dirname(self.model_file)
        fname = self.model_file
        if os.path.exists(fname):
            rospy.logerr("Loading model from " + fname)
            self.stoch_policy.load_state_dict(torch.load(fname))
            # self.qf1.load_state_dict(torch.load(os.path.join(model_dir, "qf1.pth")))
            # self.qf2.load_state_dict(torch.load(os.path.join(model_dir, "qf2.pth")))
        else:
            print("Model file not found at", fname, " using default model")

        self.policy = self.stoch_policy

        if self.is_eval:
            self.policy = MakeDeterministic(self.stoch_policy).to(ptu.device)
            rospy.logerr("*******************************")
            rospy.logerr("Policy is in deterministic mode")
            rospy.logerr("*******************************")
        else:
            rospy.logerr("****************************")
            rospy.logerr("Policy is in stochastic mode")
            rospy.logerr("****************************")

        self.policy.eval().to(ptu.device)
        # self.qf1.eval().to(ptu.device)
        # self.qf2.eval().to(ptu.device)

        rospy.Service("query_sac", QuerySAC, self.query_sac)

    def spin(self):
        rospy.spin()

    def scale_action(self, action, low, high):
        return low + (high - low) * (action + 1) / 2

    def query_sac(self, req):
        req_state = list(req.state.state)
        req_state.append(1.0 if req.state.solver_status == True else 0.0)
        obs = np.array(req_state)

        obs = torch.FloatTensor(obs).to(ptu.device)

        start = time.time()
        action_np = self.policy.get_action(obs)

        dist = self.stoch_policy(obs)
        end = time.time()

        print("inference time is:", end - start)

        action_abv = action_np[0][0]
        action_blw = action_np[0][1]

        resp = QuerySACResponse()
        resp.alpha_dot = [action_abv, action_blw]
        resp.success = True

        alpha_dot_msg = Float32()
        alpha_dot_msg.data = resp.alpha_dot[0]
        self.alpha_dot_abv_pub.publish(alpha_dot_msg)

        alpha_dot_msg.data = resp.alpha_dot[1]
        self.alpha_dot_blw_pub.publish(alpha_dot_msg)

        return resp


def main():
    rospy.init_node("model_server")
    ptu.set_gpu_mode(True)
    model_server = ModelServer()
    model_server.spin()


if __name__ == "__main__":
    main()
