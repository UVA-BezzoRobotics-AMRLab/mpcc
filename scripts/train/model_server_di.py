#!/usr/bin/env python3

import os
import time
import torch
import rospy
import numpy as np
import rlkit.torch.pytorch_util as ptu

from std_msgs.msg import Float32
from rlkit.torch.networks import ConcatMlp
from mpcc.srv import QuerySACDI, QuerySACDIResponse
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic

policy = None


class ModelServer:
    def __init__(self):

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = int(rospy.get_param("/train/hidden_dims", 256))
        self.action_dim = int(rospy.get_param("/train/action_dim", 2))
        self.state_dim = int(rospy.get_param("/train/state_dim", 12))

        self.min_alpha_dot = float(rospy.get_param("/train/min_alpha_dot", -2.0))
        self.max_alpha_dot = float(rospy.get_param("/train/max_alpha_dot", 2.0))

        self.is_eval = rospy.get_param("/train/is_eval", True)

        self.model_file = rospy.get_param("/train/model_file", "./sac_policy.pth")
        self.dynamic_model = rospy.get_param("/train/dynamic_model", "unicycle")

        self.vx_min = float(rospy.get_param("/train/vx_min", -2.0))
        self.vx_max = float(rospy.get_param("/train/vx_max", 2.0))
        self.vy_min = float(rospy.get_param("/train/vy_min", -2.0))
        self.vy_max = float(rospy.get_param("/train/vy_max", 2.0))
        self.ax_min = float(rospy.get_param("/train/ax_min", -5.0))
        self.ax_max = float(rospy.get_param("/train/ax_max", 5.0))
        self.ay_min = float(rospy.get_param("/train/ay_min", -5.0))
        self.ay_max = float(rospy.get_param("/train/ay_max", 5.0))

        self.dist_to_obs_min = float(rospy.get_param("/train/dist_to_obs_min", -0.2))
        self.dist_to_obs_max = float(rospy.get_param("/train/dist_to_obs_max", 100.0))
        self.head_to_obs_min = float(rospy.get_param("/train/head_to_obs_min", -np.pi))
        self.head_to_obs_max = float(rospy.get_param("/train/head_to_obs_max", np.pi))
        self.progress_min = float(rospy.get_param("/train/progress_min", 0.0))
        self.progress_max = float(rospy.get_param("/train/progress_max", 1.0))
        self.h_value_min = float(rospy.get_param("/train/h_value_min", -100.0))
        self.h_value_max = float(rospy.get_param("/train/h_value_max", 100.0))
        self.min_alpha = float(rospy.get_param("/train/min_alpha", 0.1))
        self.max_alpha = float(rospy.get_param("/train/max_alpha", 10.0))

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

        rospy.Service("query_sac", QuerySACDI, self.query_sac)

    def spin(self):
        rospy.spin()

    def scale_action(self, action, low, high):
        return low + (high - low) * (action + 1) / 2

    def query_sac(self, req):
        obs = None
        obs = np.array(
            [
                self.normalize(req.vx, self.vx_min, self.vx_max),
                self.normalize(req.vy, self.vy_min, self.vy_max),
                self.normalize(req.ax, self.ax_min, self.ax_max),
                self.normalize(req.ay, self.ay_min, self.ay_max),
                self.normalize(
                    req.obs_dist_abv, self.dist_to_obs_min, self.dist_to_obs_max
                ),
                self.normalize(
                    req.obs_dist_blw, self.dist_to_obs_min, self.dist_to_obs_max
                ),
                self.normalize(
                    req.heading_dist, self.head_to_obs_min, self.head_to_obs_max
                ),
                self.normalize(req.progress, self.progress_min, self.progress_max),
                req.h_val_abv,
                req.h_val_blw,
                self.normalize(req.alpha_abv, self.min_alpha, self.max_alpha),
                self.normalize(req.alpha_blw, self.min_alpha, self.max_alpha),
                1.0 if req.solver_status == "true" else 0.0,
            ]
        )

        obs = torch.FloatTensor(obs).to(ptu.device)

        start = time.time()
        action_np = self.policy.get_action(obs)

        dist = self.stoch_policy(obs)
        end = time.time()

        print("inference time is:", end - start)

        action_abv = action_np[0][0]
        action_blw = action_np[0][1]

        action1 = self.scale_action(action_abv, self.min_alpha_dot, self.max_alpha_dot)
        action2 = self.scale_action(action_blw, self.min_alpha_dot, self.max_alpha_dot)

        resp = QuerySACDIResponse()
        resp.alpha_dot = [action1, action2]
        resp.success = True

        alpha_dot_msg = Float32()
        alpha_dot_msg.data = resp.alpha_dot[0]
        self.alpha_dot_abv_pub.publish(alpha_dot_msg)

        alpha_dot_msg.data = resp.alpha_dot[1]
        self.alpha_dot_blw_pub.publish(alpha_dot_msg)

        return resp

    def normalize(self, value, min_val, max_val):
        print("NORMALIZE", value, min_val, max_val)
        return (value - min_val) / (max_val - min_val)


def main():
    rospy.init_node("model_server")
    ptu.set_gpu_mode(True)
    model_server = ModelServer()
    model_server.spin()


if __name__ == "__main__":
    main()
