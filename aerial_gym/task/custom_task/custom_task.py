from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
import torch
import numpy as np

from aerial_gym.utils.math import *

from aerial_gym.utils.logging import CustomLogger

import gymnasium as gym
from gym.spaces import Dict, Box

logger = CustomLogger("custom_task")


def dict_to_class(dict):
    return type("ClassFromDict", (object,), dict)


class CustomTask(BaseTask):
    def __init__(self, task_config):
        super().__init__(task_config)
        self.device = self.task_config.device
        # write your own implementation herer

        self.sim_env = SimBuilder().build_env(
            sim_name=self.task_config.sim_name,
            env_name=self.task_config.env_name,
            robot_name=self.task_config.robot_name,
            device=self.device,
            args=self.task_config.args,
        )

        # Implement something here that is relevant to your task

        self.task_obs = {
            "observations": torch.zeros(
                (self.sim_env.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
            "priviliged_obs": torch.zeros(
                (
                    self.sim_env.num_envs,
                    self.task_config.privileged_observation_space_dim,
                ),
                device=self.device,
                requires_grad=False,
            ),
            "collisions": torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            ),
            "rewards": torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            ),
        }

    def close(self):
        self.sim_env.delete_env()

    def reset(self):
        # write your implementation here
        return None

    def reset_idx(self, env_ids):
        # write your implementation here
        return

    def render(self):
        return self.sim_env.render()

    def step(self, actions):
        # this uses the action, gets observations
        # calculates rewards, returns tuples
        # In this case, the episodes that are terminated need to be
        # first reset, and the first obseration of the new episode
        # needs to be returned.

        # repace this with something that is relevant to your task
        self.sim_env.step(actions=actions)

        return None  # replace this with something relevant to your task


@torch.jit.script
def compute_reward(
    pos_error, crashes, action, prev_action, curriculum_level_multiplier, parameter_dict
):
    # something here
    return 0
