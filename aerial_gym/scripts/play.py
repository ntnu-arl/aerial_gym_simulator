# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from aerial_gym import AERIAL_GYM_ROOT_DIR
import os

import isaacgym
from aerial_gym.envs import *
from aerial_gym.utils import  get_args, task_registry, Logger

import numpy as np
import torch

import time


def play(args):
    env_cfg = task_registry.get_cfgs(name=args.task)
    
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.control.controller = "lee_position_control"

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    stop_state_log = 800 # number of steps before plotting states
    counter = 0

    env.reset()
    for i in range(10*int(env.max_episode_length)):
        if counter == 0:
            start_time = time.time()
        counter += 1
        actions = torch.zeros(env.num_envs, 4, device=env.device)
        actions[:, 0] = 1
        actions[:, 1] = 1.0
        actions[:, 2] = 1.0
        actions[:, 3] = -0.8
        obs, priviliged_obs, rewards, resets, extras = env.step(actions.detach())
        if counter % 800 == 0:
            env.reset()
            end_time = time.time()
            print(f"FPS: {env_cfg.env.num_envs * 100 / (end_time - start_time)}")
            start_time = time.time()

        if i < stop_state_log:
            abs_vel = torch.norm(env.root_states[:, 7:10], dim=1)
            logger.log_states(
                {
                    'command_action_x_vel': actions[robot_index, 0].item(),
                    'command_action_y_vel': actions[robot_index, 1].item(),
                    'command_action_z_vel': actions[robot_index, 2].item(),
                    'command_action_yaw_vel': actions[robot_index, 3].item(),
                    'reward': rewards[robot_index].item(),
                    'pos_x' : obs[robot_index, 0].item(),
                    'pos_y' : obs[robot_index, 1].item(),
                    'pos_z' : obs[robot_index, 2].item(),
                    'linvel_x': obs[robot_index, 7].item(),
                    'linvel_y': obs[robot_index, 8].item(),
                    'linvel_z': obs[robot_index, 9].item(),
                    'angvel_x': obs[robot_index, 10].item(),
                    'angvel_y': obs[robot_index, 11].item(),
                    'angvel_z': obs[robot_index, 12].item(),
                    'abs_linvel': abs_vel[robot_index].item()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
