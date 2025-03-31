from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_rotation_6d, quaternion_to_matrix, matrix_to_euler_angles
import torch
import numpy as np

from aerial_gym.utils.math import *

from aerial_gym.utils.logging import CustomLogger

from gym.spaces import Dict, Box

logger = CustomLogger("position_setpoint_task")


def dict_to_class(dict):
    return type("ClassFromDict", (object,), dict)


class PositionSetpointTaskSim2RealEndToEnd(BaseTask):
    def __init__(
        self, task_config, seed=None, num_envs=None, headless=None, device=None, use_warp=None
    ):
        # overwrite the params if user has provided them
        if seed is not None:
            task_config.seed = seed
        if num_envs is not None:
            task_config.num_envs = num_envs
        if headless is not None:
            task_config.headless = headless
        if device is not None:
            task_config.device = device
        if use_warp is not None:
            task_config.use_warp = use_warp

        super().__init__(task_config)
        self.device = self.task_config.device
        # set the each of the elements of reward parameter to a torch tensor
        for key in self.task_config.reward_parameters.keys():
            self.task_config.reward_parameters[key] = torch.tensor(
                self.task_config.reward_parameters[key], device=self.device
            )
        logger.info("Building environment for position setpoint task.")
        logger.info(
            "\nSim Name: {},\nEnv Name: {},\nRobot Name: {}, \nController Name: {}".format(
                self.task_config.sim_name,
                self.task_config.env_name,
                self.task_config.robot_name,
                self.task_config.controller_name,
            )
        )
        logger.info(
            "\nNum Envs: {},\nUse Warp: {},\nHeadless: {}".format(
                self.task_config.num_envs,
                self.task_config.use_warp,
                self.task_config.headless,
            )
        )

        self.sim_env = SimBuilder().build_env(
            sim_name=self.task_config.sim_name,
            env_name=self.task_config.env_name,
            robot_name=self.task_config.robot_name,
            controller_name=self.task_config.controller_name,
            args=self.task_config.args,
            device=self.device,
            num_envs=self.task_config.num_envs,
            use_warp=self.task_config.use_warp,
            headless=self.task_config.headless,
        )

        self.actions = torch.zeros(
            (self.sim_env.num_envs, self.task_config.action_space_dim),
            device=self.device,
            requires_grad=False,
        )
        self.prev_actions = torch.zeros_like(self.actions)
        self.action_history = torch.zeros(
            (self.sim_env.num_envs, self.task_config.action_space_dim*10), device=self.device, requires_grad=False)
        #self.action_history[:, 2] = 0.344

        self.counter = 0

        self.target_position = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )

        # Get the dictionary once from the environment and use it to get the observations later.
        # This is to avoid constant retuning of data back anf forth across functions as the tensors update and can be read in-place.
        self.obs_dict = self.sim_env.get_obs()
        self.obs_dict["num_obstacles_in_env"] = 1
        self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(self.truncations.shape[0], device=self.device)
        self.prev_position = torch.zeros_like(self.obs_dict["robot_position"])

        self.prev_pos_error = torch.zeros((self.sim_env.num_envs, 3), device=self.device, requires_grad=False)

        self.observation_space = Dict(
            {"observations": Box(low=-1.0, high=1.0, shape=(self.task_config.observation_space_dim,), dtype=np.float32)}
        )
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.task_config.action_space_dim,),
            dtype=np.float32,
        )
        # self.action_transformation_function = self.sim_env.robot_manager.robot.action_transformation_function

        self.num_envs = self.sim_env.num_envs

        # Currently only the "observations" are sent to the actor and critic.
        # The "priviliged_obs" are not handled so far in sample-factory

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
        self.target_position[:, 0:3] = 0.0  # torch.rand_like(self.target_position) * 10.0
        self.infos = {}
        self.sim_env.reset()
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        self.target_position[:, 0:3] = (
            0.0  # (torch.rand_like(self.target_position[env_ids]) * 10.0)
        )
        self.infos = {}
        self.sim_env.reset_idx(env_ids)
        self.action_history[env_ids] = 0.0
        self.prev_pos_error[env_ids] = 0.0
        return

    def render(self):
        return None

    def handle_action_history(self, actions):
        old_action_history = self.action_history.clone()
        self.action_history[:, self.task_config.action_space_dim:] = old_action_history[:, :-self.task_config.action_space_dim]
        self.action_history[:, :self.task_config.action_space_dim] = actions

    def step(self, actions):
        self.counter += 1
        self.actions = self.task_config.process_actions_for_task(
            actions, self.task_config.action_limit_min, self.task_config.action_limit_max
        )
        self.prev_position[:] = self.obs_dict["robot_position"].clone()

        self.sim_env.step(actions=self.actions)
        self.rewards[:], self.terminations[:] = self.compute_rewards_and_crashes(self.obs_dict)

        if self.task_config.return_state_before_reset == True:
            return_tuple = self.get_return_tuple()

        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps, 1, 0
        )

        reset_envs = self.sim_env.post_reward_calculation_step()
        if len(reset_envs) > 0:
            self.reset_idx(reset_envs)

        self.infos = {}  # self.obs_dict["infos"]

        if self.task_config.return_state_before_reset == False:
            return_tuple = self.get_return_tuple()

        self.prev_actions = self.actions.clone()
        self.prev_pos_error = self.target_position - self.obs_dict["robot_position"]

        return return_tuple

    def get_return_tuple(self):
        self.process_obs_for_task()
        return (
            self.task_obs,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def process_obs_for_task(self):
        sim_with_noise = 1.

        pos_noise = torch.normal(mean=torch.zeros_like(self.obs_dict["robot_position"]), std=0.001) * sim_with_noise
        obs_pos_noisy = (self.target_position - self.obs_dict["robot_position"]) + pos_noise

        or_noise = torch.normal(mean=torch.zeros_like(self.obs_dict["robot_orientation"][:,:3]), std=torch.pi/1032) * sim_with_noise
        or_quat = self.obs_dict["robot_orientation"][:,[3, 0, 1, 2]]
        or_euler = matrix_to_euler_angles(quaternion_to_matrix(or_quat), "ZYX")[:, [2, 1, 0]]
        obs_or_euler_noisy = or_euler + or_noise

        lin_vel_noise = torch.normal(mean=torch.zeros_like(self.obs_dict["robot_linvel"]), std=0.002) * sim_with_noise
        obs_linvel_noisy = self.obs_dict["robot_linvel"] + lin_vel_noise

        ang_vel_noise = torch.normal(mean=torch.zeros_like(self.obs_dict["robot_body_angvel"]), std=0.001) * sim_with_noise
        ang_vel_noisy = self.obs_dict["robot_body_angvel"] + ang_vel_noise

        self.task_obs["observations"][:, 0:3] = obs_pos_noisy
        or_matr_with_noise = euler_angles_to_matrix(obs_or_euler_noisy[:, [2, 1, 0]], "ZYX")
        self.task_obs["observations"][:, 3:9] = matrix_to_rotation_6d(or_matr_with_noise)
        self.task_obs["observations"][:, 9:12] = obs_linvel_noisy
        self.task_obs["observations"][:, 12:15] = ang_vel_noisy

        self.task_obs["rewards"] = self.rewards
        self.task_obs["terminations"] = self.terminations
        self.task_obs["truncations"] = self.truncations


    def compute_rewards_and_crashes(self, obs_dict):
        robot_position = obs_dict["robot_position"]
        robot_linvel = obs_dict["robot_linvel"]
        target_position = self.target_position
        robot_orientation = obs_dict["robot_orientation"]
        angular_velocity = obs_dict["robot_body_angvel"]
        action_input = self.actions

        pos_error_frame = target_position - robot_position

        return compute_reward(
            pos_error_frame,
            robot_orientation,
            robot_linvel,
            angular_velocity,
            obs_dict["crashes"],
            action_input.clone(),
            self.prev_actions,
            self.prev_pos_error,
            self.task_config.crash_dist
        )


@torch.jit.script
def exp_func(x, gain, exp):
    # type: (Tensor, float, float) -> Tensor
    return gain * torch.exp(-exp * x * x)


@torch.jit.script
def exp_penalty_func(x, gain, exp):
    # type: (Tensor, float, float) -> Tensor
    return gain * (torch.exp(-exp * x * x) - 1)


def compute_reward(
                     pos_error,
                     quats,
                     linvels_err,
                     angvels_err,
                     crashes,
                     action_input,
                     prev_action,
                     prev_pos_error,
                     crash_dist):

    target_dist = torch.norm(pos_error[:, :3], dim=1)

    prev_target_dist = torch.norm(prev_pos_error, dim=1)

    pos_error[:,2] = pos_error[:,2]*10.
    pos_reward = torch.sum(exp_func(pos_error[:, :3], 10.1, 10.0), dim=1) + torch.sum(exp_func(pos_error[:, :3], 2.0, 2.0), dim=1)

    ups = quat_axis(quats, 2)
    tiltage = 1 - ups[..., 2]
    upright_reward = exp_func(tiltage, 2.5, 5.0) + exp_func(tiltage, 2.5, 2.0)

    # forw = quat_axis(quats, 0)
    # alignment = 1 - forw[..., 0]
    # alignment_reward = exp_func(alignment, 4., 5.0) + exp_func(alignment, 2., 2.0)

    euler = get_euler_xyz_tensor(quats)
    yaw = euler[:, 2]
    yaw_reward = exp_func(yaw, 2, 2.0) + exp_func(yaw, 3, 8.0)
    alignment_reward = yaw_reward

    angvel_reward = torch.sum(exp_func(angvels_err, .75 , 10.0), dim=1)
    vel_reward = torch.sum(exp_func(linvels_err, 1., 5.0), dim=1)

    action_input_offset = action_input - 9.81 * 1.2350000515580177/4
    action_cost = torch.sum(exp_penalty_func(action_input_offset, 0.01, 10.0), dim=1)

    closer_by_dist = prev_target_dist - target_dist
    towards_goal_reward = torch.where(closer_by_dist >= 0, 50*closer_by_dist, 100*closer_by_dist)

    action_difference = action_input - prev_action
    action_difference_penalty = torch.sum(exp_penalty_func(action_difference, 0.5, 6.0), dim=1)

    reward = towards_goal_reward + (pos_reward * (alignment_reward + vel_reward + angvel_reward + action_difference_penalty) + (angvel_reward + vel_reward + upright_reward + pos_reward + action_cost)) / 100.0

    crashes[:] = torch.where(target_dist > crash_dist, torch.ones_like(crashes), crashes)

    return reward, crashes


