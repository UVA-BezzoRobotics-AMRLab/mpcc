from logging import exception
import os
import sys
import json
import torch
import yaml
import random
import sqlite3
import torch.nn as nn
import torch.optim as optim
import rlkit.torch.pytorch_util as ptu
import numpy as np
import gymnasium as gym
import rospkg
import rlkit.torch.pytorch_util as ptu

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.policies import TanhGaussianPolicy


class CustomEnvDI(gym.Env):
    def __init__(self):
        super(CustomEnvDI, self).__init__()

        self.state_dim = 11
        self.vx_min = -2.0
        self.vx_max = 2.0
        self.vy_min = -2.0
        self.vy_max = 2.0
        self.ax_min = -5
        self.ax_max = 5
        self.ay_min = -5
        self.ay_max = 5
        self.distance_to_obstacle_min = 0
        self.distance_to_obstacle_max = 100
        self.heading_to_obstacle_min = -np.pi
        self.heading_to_obstacle_max = np.pi
        self.progress_min = 0.0
        self.progress_max = 1.0
        self.h_value_min = -1e5
        self.h_value_max = 1e5
        self.alpha_min = 0.1
        self.alpha_max = 8

        self.action_dim = 2
        # self.alpha_min = .1
        # self.alpha_max = 8
        self.alpha_dot_min = -2
        self.alpha_dot_max = 2

    def set_obs_space(self):

        low = np.array(
            [
                self.vx_min,
                self.vy_min,
                self.ax_min,
                self.ay_min,
                self.distance_to_obstacle_min,
                self.distance_to_obstacle_min,
                self.heading_to_obstacle_min,
                self.progress_min,
                self.h_value_min,
                self.h_value_min,
                self.alpha_min,
                self.alpha_min,
                0,  # for solver status
            ],
            dtype=np.float64,
        )
        high = np.array(
            [
                self.vx_max,
                self.vy_max,
                self.ax_max,
                self.ay_max,
                self.distance_to_obstacle_max,
                self.distance_to_obstacle_max,
                self.heading_to_obstacle_max,
                self.progress_max,
                self.h_value_max,
                self.h_value_max,
                self.alpha_max,
                self.alpha_max,
                1,  # for solver status
            ],
            dtype=np.float64,
        )

        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float64)

        self.state = np.zeros(self.state_dim, dtype=np.float64)

    def set_action_space(self):
        # define action space
        self.action_space = gym.spaces.Box(
            low=np.array([self.alpha_dot_min]),
            high=np.array([self.alpha_dot_max]),
            dtype=np.float64,
        )

        ptu.set_gpu_mode(True)


class TrainManagerDI:
    def __init__(self, buffer_fname, log_fname, batch_size, env):

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter(log_fname)
        self.global_step = 0

        rospack = rospkg.RosPack()
        base_path = rospack.get_path("mpcc")
        param_file = os.path.join(base_path, "params", "train.yaml")

        with open(param_file, "r") as f:
            yaml_data = yaml.safe_load(f)

        self.batch_size = batch_size
        self.buffer_fname = buffer_fname

        self.obs_dim = env.state_dim
        self.action_dim = env.action_dim
        self.hidden_dim = yaml_data["hidden_dims"]

        self.buffer_size = yaml_data["buffer_size"]

        self.min_alpha_dot = float(yaml_data["min_alpha_dot"])
        self.max_alpha_dot = float(yaml_data["max_alpha_dot"])

        # Define networks
        self.qf1 = ConcatMlp(
            input_size=self.obs_dim + self.action_dim,
            output_size=1,
            hidden_sizes=[self.hidden_dim, self.hidden_dim],
        ).to(ptu.device)
        self.qf2 = ConcatMlp(
            input_size=self.obs_dim + self.action_dim,
            output_size=1,
            hidden_sizes=[self.hidden_dim, self.hidden_dim],
        ).to(ptu.device)
        self.target_qf1 = ConcatMlp(
            input_size=self.obs_dim + self.action_dim,
            output_size=1,
            hidden_sizes=[self.hidden_dim, self.hidden_dim],
        ).to(ptu.device)
        self.target_qf2 = ConcatMlp(
            input_size=self.obs_dim + self.action_dim,
            output_size=1,
            hidden_sizes=[self.hidden_dim, self.hidden_dim],
        ).to(ptu.device)
        self.policy = TanhGaussianPolicy(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=[self.hidden_dim, self.hidden_dim],
        ).to(ptu.device)

        self.env = env

        # Define SAC trainer
        self.trainer = SACTrainer(
            env=self.env,
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            discount=0.99,
            reward_scale=1.0,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=1e-5,
            qf_lr=1e-5,
            use_automatic_entropy_tuning=True,
        )

        # try to connect to the database
        self.conn = None
        try:
            self.conn = sqlite3.connect(buffer_fname)
        except Exception as e:
            print("Error connecting to database: ", str(e))
            sys.exit(-1)

    # set action between -1 and 1
    def unscale_action(self, action, low, high):
        return 2 * (action - low) / (high - low) - 1

    def normalize(self, value, min_value, max_value):
        return (value - min_value) / (max_value - min_value)

    def load_from_db(self):

        cur = self.conn.cursor()
        random_sample = None
        try:
            # query_random_from_recent = f"""
            # WITH recent_entries AS (
            #     SELECT * FROM replay_buffer
            #     ORDER BY id DESC
            #     LIMIT {self.buffer_size}
            # )
            # SELECT * FROM recent_entries
            # ORDER BY RANDOM()
            # LIMIT {self.batch_size}
            # """
            # cur.execute(query_random_from_recent)
            # Retrieve the newest 50,000 entries

            query_limit_recent = f"""
            SELECT * FROM replay_buffer
            WHERE id > (SELECT MAX(id) - {self.buffer_size} FROM replay_buffer)
            ORDER BY id DESC
            """
            cur.execute(query_limit_recent)
            recent_entries = cur.fetchall()

            # Randomly sample from the limited data set
            if len(recent_entries) < self.batch_size:
                random_sample = recent_entries
            else:
                random_sample = random.sample(recent_entries, self.batch_size)

        except Exception as e:
            print("Failed to get entries from replay buffer: ", str(e))
            return [], [], [], [], []

        labels = [description[0] for description in cur.description]

        # make mapping of label to index
        label_to_index = {label: i for i, label in enumerate(labels)}
        # rows = cur.fetchall()
        rows = random_sample

        # return states, actions, rewards, next_states, dones
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for row in rows:
            state = [
                self.normalize(
                    float(row[label_to_index["prev_vx"]]),
                    self.env.vx_min,
                    self.env.vx_max,
                ),
                self.normalize(
                    float(row[label_to_index["prev_vy"]]),
                    self.env.vy_min,
                    self.env.vy_max,
                ),
                self.normalize(
                    float(row[label_to_index["prev_ax"]]),
                    self.env.ax_min,
                    self.env.ax_max,
                ),
                self.normalize(
                    float(row[label_to_index["prev_ay"]]),
                    self.env.ay_min,
                    self.env.ay_max,
                ),
                self.normalize(
                    float(row[label_to_index["prev_obs_dist_abv"]]),
                    self.env.distance_to_obstacle_min,
                    self.env.distance_to_obstacle_max,
                ),
                self.normalize(
                    float(row[label_to_index["prev_obs_dist_blw"]]),
                    self.env.distance_to_obstacle_min,
                    self.env.distance_to_obstacle_max,
                ),
                self.normalize(
                    float(row[label_to_index["prev_obs_heading"]]),
                    self.env.heading_to_obstacle_min,
                    self.env.heading_to_obstacle_max,
                ),
                self.normalize(
                    float(row[label_to_index["prev_progress"]]),
                    self.env.progress_min,
                    self.env.progress_max,
                ),
                float(row[label_to_index["prev_h_abv"]]),
                float(row[label_to_index["prev_h_blw"]]),
                self.normalize(
                    float(row[label_to_index["prev_alpha_abv"]]),
                    self.env.alpha_min,
                    self.env.alpha_max,
                ),
                self.normalize(
                    float(row[label_to_index["prev_alpha_blw"]]),
                    self.env.alpha_min,
                    self.env.alpha_max,
                ),
                1.0 if row[label_to_index["prev_solver_status"]] == "true" else 0.0,
            ]
            action = [
                self.unscale_action(
                    float(row[label_to_index["alpha_dot_abv"]]),
                    self.min_alpha_dot,
                    self.max_alpha_dot,
                ),
                self.unscale_action(
                    float(row[label_to_index["alpha_dot_blw"]]),
                    self.min_alpha_dot,
                    self.max_alpha_dot,
                ),
            ]
            reward = float(row[label_to_index["reward"]])
            next_state = [
                self.normalize(
                    float(row[label_to_index["curr_vx"]]),
                    self.env.vx_min,
                    self.env.vx_max,
                ),
                self.normalize(
                    float(row[label_to_index["curr_vy"]]),
                    self.env.vy_min,
                    self.env.vy_max,
                ),
                self.normalize(
                    float(row[label_to_index["curr_ax"]]),
                    self.env.ax_min,
                    self.env.ax_max,
                ),
                self.normalize(
                    float(row[label_to_index["curr_ay"]]),
                    self.env.ay_min,
                    self.env.ay_max,
                ),
                self.normalize(
                    float(row[label_to_index["curr_obs_dist_abv"]]),
                    self.env.distance_to_obstacle_min,
                    self.env.distance_to_obstacle_max,
                ),
                self.normalize(
                    float(row[label_to_index["curr_obs_dist_blw"]]),
                    self.env.distance_to_obstacle_min,
                    self.env.distance_to_obstacle_max,
                ),
                self.normalize(
                    float(row[label_to_index["curr_obs_heading"]]),
                    self.env.heading_to_obstacle_min,
                    self.env.heading_to_obstacle_max,
                ),
                self.normalize(
                    float(row[label_to_index["curr_progress"]]),
                    self.env.progress_min,
                    self.env.progress_max,
                ),
                float(row[label_to_index["curr_h_abv"]]),
                float(row[label_to_index["curr_h_blw"]]),
                self.normalize(
                    float(row[label_to_index["curr_alpha_abv"]]),
                    self.env.alpha_min,
                    self.env.alpha_max,
                ),
                self.normalize(
                    float(row[label_to_index["curr_alpha_blw"]]),
                    self.env.alpha_min,
                    self.env.alpha_max,
                ),
                1.0 if row[label_to_index["curr_solver_status"]] == "true" else 0.0,
            ]

            done = True if row[label_to_index["is_done"]] == "true" else False

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        return states, actions, rewards, next_states, dones

    def train_sac(self):
        states, actions, rewards, next_states, dones = self.load_from_db()

        states = torch.FloatTensor(states).to(ptu.device)
        # actions = torch.FloatTensor(actions).unsqueeze(1).to(ptu.device)
        actions = torch.FloatTensor(actions).to(ptu.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(ptu.device)
        next_states = torch.FloatTensor(next_states).to(ptu.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(ptu.device)

        self.trainer.train_from_torch(
            batch={
                "observations": states,
                "actions": actions,
                "rewards": rewards,
                "next_observations": next_states,
                "terminals": dones,
            }
        )

        # Logging metrics
        eval_stats = self.trainer.get_diagnostics()

        # average reward for the episode
        episode_reward = rewards.mean().item()
        q_values_1 = self.trainer.qf1(states, actions).detach().cpu().numpy()
        q_values_2 = self.trainer.qf2(states, actions).detach().cpu().numpy()
        average_q_value_1 = q_values_1.mean()
        average_q_value_2 = q_values_2.mean()
        min_average_q_value = min(average_q_value_1, average_q_value_2)

        self.writer.add_scalar("Total Episode Reward", episode_reward, self.global_step)
        self.writer.add_scalar(
            "Average Q-Value QF1", average_q_value_1, self.global_step
        )
        self.writer.add_scalar(
            "Average Q-Value QF2", average_q_value_2, self.global_step
        )
        self.writer.add_scalar(
            "Min Average Q-Value", min_average_q_value, self.global_step
        )

        if "Policy Loss" in eval_stats:
            self.writer.add_scalar(
                "Policy Loss", eval_stats["Policy Loss"], self.global_step
            )
        if "QF1 Loss" in eval_stats:
            self.writer.add_scalar(
                "Q-Function Loss QF1", eval_stats["QF1 Loss"], self.global_step
            )
        if "QF2 Loss" in eval_stats:
            self.writer.add_scalar(
                "Q-Function Loss QF2", eval_stats["QF2 Loss"], self.global_step
            )
        if "Log Pis" in eval_stats:
            self.writer.add_scalar("Entropy", eval_stats["Log Pis"], self.global_step)
        if "Alpha" in eval_stats:
            self.writer.add_scalar("Alpha", eval_stats["Alpha"], self.global_step)
        if "Alpha Loss" in eval_stats:
            self.writer.add_scalar(
                "Alpha Loss", eval_stats["Alpha Loss"], self.global_step
            )

        self.global_step += 1

    def close(self):
        self.writer.close()
