from aerial_gym.task.position_setpoint_task.position_setpoint_task import PositionSetpointTask
from aerial_gym.sim.sim_builder import SimBuilder
import torch
import numpy as np

from aerial_gym.utils.math import *

from aerial_gym.utils.logging import CustomLogger

import gymnasium as gym
from gym.spaces import Dict, Box

logger = CustomLogger("position_setpoint_task")


def dict_to_class(dict):
    return type("ClassFromDict", (object,), dict)


class PositionSetpointTaskReconfigurable(PositionSetpointTask):
    def __init__(
        self, task_config, seed=None, num_envs=None, headless=None, device=None, use_warp=None
    ):
        super().__init__(
            task_config=task_config,
            seed=seed,
            num_envs=num_envs,
            headless=headless,
            device=device,
            use_warp=use_warp,
        )

        self.action_limit_min = torch.tensor(
            task_config.action_limit_min, device=self.device, dtype=torch.float32
        ).expand(self.num_envs, -1)
        self.action_limit_max = torch.tensor(
            task_config.action_limit_max, device=self.device, dtype=torch.float32
        ).expand(self.num_envs, -1)

    def step(self, actions):
        self.counter += 1
        self.prev_actions[:] = self.actions
        self.actions = self.task_config.process_actions_for_task(
            actions, self.action_limit_min, self.action_limit_max
        )

        # this uses the action, gets observations
        # calculates rewards, returns tuples
        # In this case, the episodes that are terminated need to be
        # first reset, and the first obseration of the new episode
        # needs to be returned.

        # set action before stepping
        self.sim_env.robot_manager.robot.set_dof_velocity_targets(
            self.actions[:, self.task_config.num_motors :]
        )
        self.sim_env.step(actions=self.actions[:, : self.task_config.num_motors])

        # This step must be done since the reset is done after the reward is calculated.
        # This enables the robot to send back an updated state, and an updated observation to the RL agent after the reset.
        # This is important for the RL agent to get the correct state after the reset.
        self.rewards[:], self.terminations[:] = self.compute_rewards_and_crashes(self.obs_dict)

        if self.task_config.return_state_before_reset == True:
            return_tuple = self.get_return_tuple()

        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps, 1, 0
        )
        self.sim_env.post_reward_calculation_step()

        self.infos = {}  # self.obs_dict["infos"]

        if self.task_config.return_state_before_reset == False:
            return_tuple = self.get_return_tuple()

        return return_tuple

    def process_obs_for_task(self):
        self.task_obs["observations"][:, 0:3] = (
            self.target_position - self.obs_dict["robot_position"]
        )
        self.task_obs["observations"][:, 3:7] = self.obs_dict["robot_orientation"]
        self.task_obs["observations"][:, 7:10] = self.obs_dict["robot_body_linvel"]
        self.task_obs["observations"][:, 10:13] = self.obs_dict["robot_body_angvel"]
        self.task_obs["observations"][:, 13 : 13 + self.task_config.action_space_dim] = self.actions
        self.task_obs["observations"][
            :,
            13
            + self.task_config.action_space_dim : 13
            + self.task_config.action_space_dim
            + self.task_config.num_joints,
        ] = self.obs_dict["dof_state_tensor"][..., 0].reshape(-1, self.task_config.num_joints)
        self.task_obs["observations"][
            :, 13 + self.task_config.action_space_dim + self.task_config.num_joints :
        ] = self.obs_dict["dof_state_tensor"][..., 1].reshape(-1, self.task_config.num_joints)

        # print NAN value locations in the observation tensor
        if torch.isnan(self.task_obs["observations"]).any():
            logger.info(
                "NAN values in the observation tensor: ",
                torch.isnan(self.task_obs["observations"]).nonzero(),
            )

        self.task_obs["rewards"] = self.rewards
        self.task_obs["terminations"] = self.terminations
        self.task_obs["truncations"] = self.truncations

    def compute_rewards_and_crashes(self, obs_dict):
        robot_position = obs_dict["robot_position"]
        target_position = self.target_position
        robot_vehicle_orientation = obs_dict["robot_vehicle_orientation"]
        robot_orientation = obs_dict["robot_orientation"]
        target_orientation = torch.zeros_like(robot_orientation, device=self.device)
        target_orientation[:, 3] = 1.0
        angular_velocity = obs_dict["robot_body_angvel"]
        root_quats = obs_dict["robot_orientation"]

        pos_error_vehicle_frame = quat_apply_inverse(
            robot_vehicle_orientation, (target_position - robot_position)
        )
        return compute_reward(
            pos_error_vehicle_frame,
            root_quats,
            angular_velocity,
            obs_dict["crashes"],
            1.0,  # obs_dict["curriculum_level_multiplier"],
            self.actions,
            self.prev_actions,
            self.task_config.reward_parameters,
        )


@torch.jit.script
def exp_func(x, gain, exp):
    # type: (Tensor, float, float) -> Tensor
    return gain * torch.exp(-exp * x * x)


@torch.jit.script
def exp_penalty_func(x, gain, exp):
    # type: (Tensor, float, float) -> Tensor
    return gain * (torch.exp(-exp * x * x) - 1)


@torch.jit.script
def compute_reward(
    pos_error,
    robot_quats,
    robot_angvels,
    crashes,
    curriculum_level_multiplier,
    current_action,
    prev_actions,
    parameter_dict,
):
    # type: (Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Dict[str, Tensor]) -> Tuple[Tensor, Tensor]

    dist = torch.norm(pos_error, dim=1)
    pos_reward = exp_func(dist, 3.0, 8.0) + exp_func(dist, 0.5, 1.0)
    dist_reward = (20 - dist) / 40.0  # 40

    # ups = quat_axis(robot_quats, 2)
    # tiltage = torch.abs(1 - ups[..., 2])
    roll, pitch, yaw = get_euler_xyz(robot_quats)
    roll = ssa(roll)
    pitch = ssa(pitch)
    up_reward = exp_func(roll, 3.0, 5.0) + exp_func(pitch, 3.0, 5.0)

    spinnage = torch.norm(robot_angvels, dim=1)
    ang_vel_reward = exp_func(spinnage, 3.0, 10.5)
    yaw_rate_special = exp_func(torch.abs(robot_angvels[:, 2]), 5.0, 20.5)

    total_reward = (
        pos_reward
        + dist_reward
        + yaw_rate_special
        + pos_reward * (up_reward + ang_vel_reward + yaw_rate_special)
    )  # + previous_action_penalty + absolute_action_penalty
    total_reward[:] = curriculum_level_multiplier * total_reward

    crashes[:] = torch.where(dist > 3.0, torch.ones_like(crashes), crashes)
    crashes[:] = torch.where(torch.abs(roll) > 1.0, torch.ones_like(crashes), crashes)
    crashes[:] = torch.where(torch.abs(pitch) > 1.0, torch.ones_like(crashes), crashes)

    total_reward[:] = torch.where(crashes > 0.0, -20 * torch.ones_like(total_reward), total_reward)

    return total_reward, crashes
