from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
import torch
import numpy as np

from aerial_gym.utils.math import *

from aerial_gym.utils.logging import CustomLogger

import gymnasium as gym
from gym.spaces import Dict, Box

logger = CustomLogger("navigation_task")


def dict_to_class(dict):
    return type("ClassFromDict", (object,), dict)


class LiDARNavigationTask(BaseTask):
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
        logger.info("Building environment for navigation task.")
        logger.info(
            "Sim Name: {}, Env Name: {}, Robot Name: {}, Controller Name: {}".format(
                self.task_config.sim_name,
                self.task_config.env_name,
                self.task_config.robot_name,
                self.task_config.controller_name,
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

        self.target_position = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )

        self.target_min_ratio = torch.tensor(
            self.task_config.target_min_ratio, device=self.device, requires_grad=False
        ).expand(self.sim_env.num_envs, -1)
        self.target_max_ratio = torch.tensor(
            self.task_config.target_max_ratio, device=self.device, requires_grad=False
        ).expand(self.sim_env.num_envs, -1)

        self.success_aggregate = 0
        self.crashes_aggregate = 0
        self.timeouts_aggregate = 0

        self.pos_error_vehicle_frame_prev = torch.zeros_like(
            self.target_position)
        self.pos_error_vehicle_frame = torch.zeros_like(self.target_position)

        self.world_dir_vectors = torch.ones(
            (self.sim_env.num_envs, 48, 120, 3), device=self.device
        )

        self.current_action = torch.zeros((self.sim_env.num_envs, 4), device=self.device)
        self.prev_action = torch.zeros((self.sim_env.num_envs, 4), device=self.device)

        self.time_to_collision = torch.zeros((self.sim_env.num_envs), device=self.device)
        self.target_yaw = torch.zeros((self.sim_env.num_envs), device=self.device)

        # Get the dictionary once from the environment and use it to get the observations later.
        # This is to avoid constant retuning of data back anf forth across functions as the tensors update and can be read in-place.
        self.obs_dict = self.sim_env.get_obs()
        if "curriculum_level" not in self.obs_dict.keys():
            self.curriculum_level = self.task_config.curriculum.min_level
            self.obs_dict["curriculum_level"] = self.curriculum_level
        else:
            self.curriculum_level = self.obs_dict["curriculum_level"]
        self.obs_dict["num_obstacles_in_env"] = self.curriculum_level
        self.curriculum_progress_fraction = (
            self.curriculum_level - self.task_config.curriculum.min_level
        ) / (self.task_config.curriculum.max_level - self.task_config.curriculum.min_level)

        self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(
            self.truncations.shape[0], device=self.device)

        self.observation_space = Dict(
            {
                "observations": Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.task_config.observation_space_dim,),
                    dtype=np.float32,
                ),
            }
        )
        self.downsampled_lidar_data = torch.zeros(
            (self.sim_env.num_envs, 16*20), device=self.device, requires_grad=False
        )
        self.action_space = Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_transformation_function = self.task_config.action_transformation_function

        self.num_envs = self.sim_env.num_envs

        # Currently only the "observations" are sent to the actor and critic.
        # The "priviliged_obs" are not handled so far in sample-factory

        self.task_obs = {
            "observations": torch.zeros(
                (self.sim_env.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
            # "priviliged_obs": torch.zeros(
            #     (
            #         self.sim_env.num_envs,
            #         self.task_config.privileged_observation_space_dim,
            #     ),
            #     device=self.device,
            #     requires_grad=False,
            # ),
            # "collisions": torch.zeros(
            #     (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            # ),
            # "rewards": torch.zeros(
            #     (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            # ),
        }

        self.num_task_steps = 0

    def close(self):
        self.sim_env.delete_env()

    def reset(self):
        self.reset_idx(torch.arange(self.sim_env.num_envs))
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        target_ratio = torch_rand_float_tensor(
            self.target_min_ratio, self.target_max_ratio)
        self.target_position[env_ids] = torch_interpolate_ratio(
            min=self.obs_dict["env_bounds_min"][env_ids],
            max=self.obs_dict["env_bounds_max"][env_ids],
            ratio=target_ratio[env_ids],
        )
        self.obs_dict["robot_prev_actions"][env_ids] = 0.0

        self.target_yaw[env_ids] = torch_rand_float_tensor(
            -torch.pi * torch.ones(len(env_ids), device=self.device),
            torch.pi * torch.ones(len(env_ids), device=self.device),
        )

        # logger.warning(f"reset envs: {env_ids}")
        self.infos = {}
        return

    def render(self):
        return self.sim_env.render()

    def logging_sanity_check(self, infos):
        successes = infos["successes"]
        crashes = infos["crashes"]
        timeouts = infos["timeouts"]
        time_at_crash = torch.where(
            crashes > 0,
            self.sim_env.sim_steps,
            self.task_config.episode_len_steps *
            torch.ones_like(self.sim_env.sim_steps),
        )
        env_list_for_toc = (time_at_crash < 5).nonzero(
            as_tuple=False).squeeze(-1)
        crash_envs = crashes.nonzero(as_tuple=False).squeeze(-1)
        success_envs = successes.nonzero(as_tuple=False).squeeze(-1)
        timeout_envs = timeouts.nonzero(as_tuple=False).squeeze(-1)

        if len(env_list_for_toc) > 0:
            logger.critical("Crash is happening too soon.")
            logger.critical(f"Envs crashing too soon: {env_list_for_toc}")
            logger.critical(
                f"Time at crash: {time_at_crash[env_list_for_toc]}")

        if torch.sum(torch.logical_and(successes, crashes)) > 0:
            logger.critical("Success and crash are occuring at the same time")
            logger.critical(
                f"Number of crashes: {torch.count_nonzero(crashes)}, Crashed envs: {crash_envs}"
            )
            logger.critical(
                f"Number of successes: {torch.count_nonzero(successes)}, Success envs: {success_envs}"
            )
            logger.critical(
                f"Number of common instances: {torch.count_nonzero(torch.logical_and(crashes, successes))}"
            )
        if torch.sum(torch.logical_and(successes, timeouts)) > 0:
            logger.critical(
                "Success and timeout are occuring at the same time")
            logger.critical(
                f"Number of successes: {torch.count_nonzero(successes)}, Success envs: {success_envs}"
            )
            logger.critical(
                f"Number of timeouts: {torch.count_nonzero(timeouts)}, Timeout envs: {timeout_envs}"
            )
            logger.critical(
                f"Number of common instances: {torch.count_nonzero(torch.logical_and(successes, timeouts))}"
            )
        if torch.sum(torch.logical_and(crashes, timeouts)) > 0:
            logger.critical("Crash and timeout are occuring at the same time")
            logger.critical(
                f"Number of crashes: {torch.count_nonzero(crashes)}, Crashed envs: {crash_envs}"
            )
            logger.critical(
                f"Number of timeouts: {torch.count_nonzero(timeouts)}, Timeout envs: {timeout_envs}"
            )
            logger.critical(
                f"Number of common instances: {torch.count_nonzero(torch.logical_and(crashes, timeouts))}"
            )
        return

    def check_and_update_curriculum_level(self, successes, crashes, timeouts):
        self.success_aggregate += torch.sum(successes)
        self.crashes_aggregate += torch.sum(crashes)
        self.timeouts_aggregate += torch.sum(timeouts)

        instances = self.success_aggregate + \
            self.crashes_aggregate + self.timeouts_aggregate

        if instances >= self.task_config.curriculum.check_after_log_instances:
            success_rate = self.success_aggregate / instances
            crash_rate = self.crashes_aggregate / instances
            timeout_rate = self.timeouts_aggregate / instances

            if success_rate > self.task_config.curriculum.success_rate_for_increase:
                self.curriculum_level += self.task_config.curriculum.increase_step
            elif success_rate < self.task_config.curriculum.success_rate_for_decrease:
                self.curriculum_level -= self.task_config.curriculum.decrease_step

            # clamp curriculum_level
            self.curriculum_level = min(
                max(self.curriculum_level, self.task_config.curriculum.min_level),
                self.task_config.curriculum.max_level,
            )
            self.obs_dict["curriculum_level"] = self.curriculum_level
            self.obs_dict["num_obstacles_in_env"] = self.curriculum_level
            self.curriculum_progress_fraction = (
                self.curriculum_level - self.task_config.curriculum.min_level
            ) / (self.task_config.curriculum.max_level - self.task_config.curriculum.min_level)

            logger.warning(
                f"Curriculum Level: {self.curriculum_level}, Curriculum progress fraction: {self.curriculum_progress_fraction}"
            )
            logger.warning(
                f"\nSuccess Rate: {success_rate}\nCrash Rate: {crash_rate}\nTimeout Rate: {timeout_rate}"
            )
            logger.warning(
                f"\nSuccesses: {self.success_aggregate}\nCrashes : {self.crashes_aggregate}\nTimeouts: {self.timeouts_aggregate}"
            )
            self.success_aggregate = 0
            self.crashes_aggregate = 0
            self.timeouts_aggregate = 0
    
    def add_noise_to_downsampled_lidar_data(self, ds_lidar_data):
        # random noise to 3% pixels
        noise_mask = torch.bernoulli(
            0.03 * torch.ones_like(ds_lidar_data)).to(self.device)
        ds_lidar_data[noise_mask == 1] += torch_rand_float_tensor(
            0.2 * torch.ones_like(noise_mask[noise_mask == 1]),
            10.0 * torch.ones_like(noise_mask[noise_mask == 1])
        ).to(self.device)

        # bernoulli sampling to have 1-2% points max range
        max_range_points_mask = torch.bernoulli(
            0.02 * torch.ones_like(ds_lidar_data)).to(self.device)
        ds_lidar_data[max_range_points_mask == 1] = 10.0

        # 1-2% points in the bottom half of the image to be a low value between 0.2 to 1 meter
        # 5% pixels below 10: index of ds_lidar_data should have random range between 0.2 to 1.0 metres

        low_range_points_mask = torch.bernoulli(
            0.02 * torch.ones_like(ds_lidar_data[:, 10:])).to(self.device)
        random_low_ranges = torch_rand_float_tensor(
            0.2 * torch.ones_like(low_range_points_mask),
            1.0 * torch.ones_like(low_range_points_mask)
        ).to(self.device)
        ds_lidar_data[:, 10:][low_range_points_mask == 1] = random_low_ranges[low_range_points_mask == 1]
        return ds_lidar_data


    def process_image_observation(self):
        pointcloud_obs = self.obs_dict["depth_range_pixels"].squeeze(1)
        self.world_dir_vectors[:] = pointcloud_obs - self.obs_dict["robot_position"].unsqueeze(1).unsqueeze(1)
        range_obs = torch.norm(self.world_dir_vectors, dim=-1)
        range_obs_flat = range_obs.view(self.num_envs, -1)
        self.world_unit_dir = self.world_dir_vectors.view(self.num_envs, -1, 3) / (range_obs_flat.unsqueeze(-1) + 1e-6)
        
        range_obs[range_obs > 10] = 10.0
        range_obs[range_obs < 0.2] = 10.0

        image_obs = range_obs.clone()


        self.world_dir_vectors_flat = self.world_dir_vectors.view(self.num_envs, -1, 3)
        self.vel_component_along_dir = torch.sum(
            self.obs_dict["robot_linvel"].unsqueeze(1) * self.world_unit_dir, dim=-1
        )

        time_to_collision = torch.where(
            self.vel_component_along_dir > 0,
            (range_obs_flat) / (self.vel_component_along_dir + 1e-6),
            10.0 * torch.ones_like(range_obs_flat)
        )

        # print(self.vel_component_along_dir.shape,range_obs_flat.shape)

        self.time_to_collision[:] = torch.clamp(torch.min(time_to_collision, dim=-1).values, 0.0, 10.0)
        min_indices = torch.min(time_to_collision, dim=-1).indices
        # print(min_indices)
        # print("time to collision: ", self.time_to_collision[0])
        # print(range_obs_flat[0, min_indices[0]], self.vel_component_along_dir[0, min_indices[0]])

        # Min pooling to downsample the image
        image_obs_ds = -torch.nn.functional.max_pool2d(
            -image_obs.unsqueeze(1), (3, 6)).squeeze(1)
        
        # Noise after min pooling
        image_obs_noisy = self.add_noise_to_downsampled_lidar_data(image_obs_ds)
        inv_range_image = 1 / image_obs_noisy
        self.downsampled_lidar_data[:] = inv_range_image.reshape(
            (self.num_envs, -1)).to(self.device)

        # invalid pixels are set to max distance
        # # downsample the image using max pooling
        # inv_image_obs_ds = torch.nn.functional.max_pool2d(
        #     inv_range_image.unsqueeze(1), (3, 6)).squeeze(1)

        # self.downsampled_lidar_data[:] = inv_image_obs_ds.reshape(
        #     (self.num_envs, -1)).to(self.device)
        return
        

    def step(self, actions):
        # this uses the action, gets observations
        # calculates rewards, returns tuples
        # In this case, the episodes that are terminated need to be
        # first reset, and the first obseration of the new episode
        # needs to be returned.
        self.prev_action[:] = self.current_action[:]
        transformed_action = self.action_transformation_function(actions)
        self.current_action[:] = transformed_action[:]
        logger.debug(
            f"raw_action: {actions[0]}, transformed action: {transformed_action[0]}")
        self.sim_env.step(actions=transformed_action)

        # This step must be done since the reset is done after the reward is calculated.
        # This enables the robot to send back an updated state, and an updated observation to the RL agent after the reset.
        # This is important for the RL agent to get the correct state after the reset.
        self.rewards[:], self.terminations[:] = self.compute_rewards_and_crashes(
            self.obs_dict)

        # logger.info(f"Curricluum Level: {self.curriculum_level}")

        if self.task_config.return_state_before_reset == True:
            return_tuple = self.get_return_tuple()

        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps,
            torch.ones_like(self.truncations),
            torch.zeros_like(self.truncations),
        )

        # successes are are the sum of the environments which are to be truncated and have reached the target within a distance threshold
        successes = self.truncations * (
            torch.norm(self.target_position -
                       self.obs_dict["robot_position"], dim=1) < 1.0
        )
        successes = torch.where(self.terminations > 0,
                                torch.zeros_like(successes), successes)
        timeouts = torch.where(
            self.truncations > 0, torch.logical_not(
                successes), torch.zeros_like(successes)
        )
        timeouts = torch.where(
            self.terminations > 0, torch.zeros_like(timeouts), timeouts
        )  # timeouts are not counted if there is a crash

        self.infos["successes"] = successes
        self.infos["timeouts"] = timeouts
        self.infos["crashes"] = self.terminations

        self.logging_sanity_check(self.infos)
        self.check_and_update_curriculum_level(
            self.infos["successes"], self.infos["crashes"], self.infos["timeouts"]
        )
        # rendering happens at the post-reward calculation step since the newer measurement is required to be
        # sent to the RL algorithm as an observation and it helps if the camera image is updated then
        reset_envs = self.sim_env.post_reward_calculation_step()
        if len(reset_envs) > 0:
            self.reset_idx(reset_envs)
        self.num_task_steps += 1
        # do stuff with the image observations here
        self.process_image_observation()
        if self.task_config.return_state_before_reset == False:
            return_tuple = self.get_return_tuple()
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
        vec_to_tgt = quat_rotate_inverse(
            self.obs_dict["robot_vehicle_orientation"],
            (self.target_position - self.obs_dict["robot_position"]),
        )
        perturbed_vec_to_tgt = vec_to_tgt + 0.1 * \
            2 * (torch.rand_like(vec_to_tgt) - 0.5)
        dist_to_tgt = torch.norm(vec_to_tgt, dim=-1)
        perturbed_unit_vec_to_tgt = perturbed_vec_to_tgt / \
            dist_to_tgt.unsqueeze(1)
        self.task_obs["observations"][:, 0:3] = perturbed_unit_vec_to_tgt
        self.task_obs["observations"][:, 3] = dist_to_tgt
        # self.task_obs["observation"][:, 3] = self.infos["successes"]
        # self.task_obs["observations"][:, 3:7] = self.obs_dict["robot_vehicle_orientation"]
        euler_angles = ssa(self.obs_dict["robot_euler_angles"])
        perturbed_euler_angles = euler_angles + 0.1 * \
            (torch.rand_like(euler_angles) - 0.5)
        self.task_obs["observations"][:, 4] = perturbed_euler_angles[:, 0]
        self.task_obs["observations"][:, 5] = perturbed_euler_angles[:, 1]
        self.task_obs["observations"][:, 6] = ssa(
            self.target_yaw - euler_angles[:, 2]
        )
        # print("yaw error obs: ", self.task_obs["observations"][0, 6])
        self.task_obs["observations"][:,
                                      7:10] = self.obs_dict["robot_body_linvel"]
        self.task_obs["observations"][:,
                                      10:13] = self.obs_dict["robot_body_angvel"]
        self.task_obs["observations"][:,
                                      13:17] = self.obs_dict["robot_actions"]
        self.task_obs["observations"][:, 17:] = self.downsampled_lidar_data

    def compute_rewards_and_crashes(self, obs_dict):
        robot_position = obs_dict["robot_position"]
        target_position = self.target_position
        robot_vehicle_orientation = obs_dict["robot_vehicle_orientation"]
        robot_orientation = obs_dict["robot_orientation"]
        target_orientation = torch.zeros_like(
            robot_orientation, device=self.device)
        target_orientation[:, 3] = 1.0
        self.pos_error_vehicle_frame_prev[:] = self.pos_error_vehicle_frame
        self.pos_error_vehicle_frame[:] = quat_rotate_inverse(
            robot_vehicle_orientation, (target_position - robot_position)
        )
        euler_angles = ssa(obs_dict["robot_euler_angles"])
        yaw_error = ssa(self.target_yaw - euler_angles[:, 2])
        return compute_reward(
            self.pos_error_vehicle_frame,
            self.pos_error_vehicle_frame_prev,
            self.obs_dict["robot_vehicle_linvel"],
            self.obs_dict["robot_body_angvel"],
            yaw_error,
            obs_dict["crashes"],
            # obs_dict["robot_actions"],
            # obs_dict["robot_prev_actions"],
            self.current_action,
            self.prev_action,
            self.time_to_collision,
            self.curriculum_progress_fraction,
            self.task_config.reward_parameters,
        )


@torch.jit.script
def exponential_reward_function(
    magnitude: float, exponent: float, value: torch.Tensor
) -> torch.Tensor:
    """Exponential reward function"""
    return magnitude * torch.exp(-(value * value) * exponent)


@torch.jit.script
def exponential_penalty_function(
    magnitude: float, exponent: float, value: torch.Tensor
) -> torch.Tensor:
    """Exponential reward function"""
    return magnitude * (torch.exp(-(value * value) * exponent) - 1.0)


erf = exponential_reward_function
epf = exponential_penalty_function

@torch.jit.script
def smooth_linear_reward(
    magnitude: float, limit: float, error: torch.Tensor
) -> torch.Tensor:
    """
    Smooth Linear Reward (Inverted Huber Loss).
    Quadratic near 0 (stable), Linear far away (strong gradient).
    
    Args:
        magnitude: Max reward value (at error=0).
        limit: The error value where transition from quadratic to linear happens.
        error: The error tensor.
    """
    abs_error = torch.abs(error)
    
    # R = M - scale * Huber(e)
    # Huber(e) = 0.5*e^2 if |e|<d else d*(|e|-0.5d)
    # We want R(pi) = 0.
    # Huber(pi) = d*(pi - 0.5d).
    # So scale = M / (d*(pi - 0.5d)).
    
    d = limit
    huber = torch.where(
        abs_error < d,
        0.5 * abs_error * abs_error,
        d * (abs_error - 0.5 * d)
    )
    max_huber = d * (torch.pi - 0.5 * d)
    return magnitude * (1.0 - huber / max_huber)

slr = smooth_linear_reward

@torch.jit.script
def compute_reward(
    pos_error,
    prev_pos_error,
    robot_vehicle_linvel,
    robot_body_angvel,
    yaw_error,
    crashes,
    action,
    prev_action,
    time_to_collision,
    curriculum_progress_fraction,
    parameter_dict,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Dict[str, Tensor]) -> Tuple[Tensor, Tensor]
    MULTIPLICATION_FACTOR_REWARD = 1.0 + (2.0) * curriculum_progress_fraction
    dist = torch.norm(pos_error, dim=1)
    prev_dist_to_goal = torch.norm(prev_pos_error, dim=1)
    pos_reward = exponential_reward_function(
        parameter_dict["pos_reward_magnitude"],
        parameter_dict["pos_reward_exponent"],
        dist,
    )
    very_close_to_goal_reward = exponential_reward_function(
        parameter_dict["very_close_to_goal_reward_magnitude"],
        parameter_dict["very_close_to_goal_reward_exponent"],
        dist,
    )

    robot_vel_norm = torch.norm(robot_vehicle_linvel, dim=1)
    robot_vel_dir = robot_vehicle_linvel / (
        robot_vel_norm.unsqueeze(1) + 1e-6)
    unit_vec_to_goal = pos_error / (dist.unsqueeze(1) + 1e-6)

    reasonable_vel = exponential_reward_function(
        2.0,
        2.0,
        (robot_vel_norm - 2.0)
    )
    
    
    vel_dir_component = torch.sum(robot_vel_dir * unit_vec_to_goal, dim=1)

    vel_dir_component_reward = torch.where(vel_dir_component > 0,
                                           parameter_dict["vel_direction_component_reward_magnitude"] * vel_dir_component * reasonable_vel,
                                           -0.2 * torch.ones_like(vel_dir_component)
                                           ) * torch.min((dist/3.0) , torch.ones_like(dist))
    
    # for acceleration setpoint task
    # any velocity greater than 3m/s be penalized
    vel_magnitude_penalty = exponential_penalty_function(
        2.0,
        2.0,
        torch.clamp(robot_vel_norm - 3.0, min=0.0),
    )

    close_to_goal = 1.0 - exponential_reward_function(
        1.0,
        2.0,
        dist
    )

    # relax the negative penalty when relatively close to goal
    negative_x_vel_penalty = exponential_penalty_function(
        2.0,
        8.0,
        torch.clamp(robot_vehicle_linvel[:, 0], min=0.0),
    ) * close_to_goal

    vel_penalty_for_acc = vel_magnitude_penalty + negative_x_vel_penalty

    # stable at goal reward 
    low_vel_reward = erf(1.5, 10.0, robot_vel_norm) + erf(1.5, 0.5, robot_vel_norm)

    # correct_yaw_reward = erf(2.0, 0.5, yaw_error) + erf(4.0, 15.0, yaw_error)
    # correct_yaw_reward = 6.0 * (1.0 - torch.abs(yaw_error) / torch.pi)
    # correct_yaw_reward = smooth_linear_reward(8.0, 0.3, yaw_error)
    correct_yaw_reward = erf(2.0, 0.2, yaw_error) + erf(4.0, 15.0, yaw_error)


    # Gate the angular velocity reward so it only applies when aligned
    # This prevents the robot from being penalized for turning when it needs to correct yaw
    alignment_factor = erf(1.0, 2.0, yaw_error)
    low_angvel_reward = erf(1.5, 5.0, robot_body_angvel[:, 2]) * alignment_factor

    stable_at_goal_reward = torch.where(
        dist < 1.0,
        (low_vel_reward + correct_yaw_reward + low_angvel_reward),
        torch.zeros_like(low_vel_reward),
    )


    distance_from_goal_reward = (20.0 - dist) / 20.0
    action_diff = action - prev_action
    x_diff_penalty = exponential_penalty_function(
        parameter_dict["x_action_diff_penalty_magnitude"],
        parameter_dict["x_action_diff_penalty_exponent"],
        action_diff[:, 0],
    )
    y_diff_penalty = exponential_penalty_function(
        parameter_dict["y_action_diff_penalty_magnitude"],
        parameter_dict["y_action_diff_penalty_exponent"],
        action_diff[:, 1],
    )
    z_diff_penalty = exponential_penalty_function(
        parameter_dict["z_action_diff_penalty_magnitude"],
        parameter_dict["z_action_diff_penalty_exponent"],
        action_diff[:, 2],
    )
    yawrate_diff_penalty = exponential_penalty_function(
        parameter_dict["yawrate_action_diff_penalty_magnitude"],
        parameter_dict["yawrate_action_diff_penalty_exponent"],
        action_diff[:, 3],
    )
    action_diff_penalty = x_diff_penalty + y_diff_penalty + z_diff_penalty + yawrate_diff_penalty
    # absolute action penalty
    x_absolute_penalty = curriculum_progress_fraction * exponential_penalty_function(
        parameter_dict["x_absolute_action_penalty_magnitude"],
        parameter_dict["x_absolute_action_penalty_exponent"],
        action[:, 0],
    )
    z_absolute_penalty = curriculum_progress_fraction * exponential_penalty_function(
        parameter_dict["z_absolute_action_penalty_magnitude"],
        parameter_dict["z_absolute_action_penalty_exponent"],
        action[:, 2],
    )
    yawrate_absolute_penalty = curriculum_progress_fraction * exponential_penalty_function(
        parameter_dict["yawrate_absolute_action_penalty_magnitude"],
        parameter_dict["yawrate_absolute_action_penalty_exponent"],
        action[:, 3],
    )
    y_absolute_penalty = curriculum_progress_fraction * exponential_penalty_function(
        parameter_dict["y_absolute_action_penalty_magnitude"],
        parameter_dict["y_absolute_action_penalty_exponent"],
        action[:, 1],
    )
    absolute_action_penalty = x_absolute_penalty + \
        z_absolute_penalty + yawrate_absolute_penalty + y_absolute_penalty
    total_action_penalty = action_diff_penalty + absolute_action_penalty

    time_to_collision_penalty = exponential_reward_function(
        -3.0,
        2.0,
        time_to_collision**2
    )

    # combined reward
    reward = (
        MULTIPLICATION_FACTOR_REWARD
        * (
            pos_reward
            + very_close_to_goal_reward * alignment_factor
            + vel_dir_component_reward
            + distance_from_goal_reward
            + stable_at_goal_reward
            + vel_penalty_for_acc
            + total_action_penalty
            + time_to_collision_penalty
        )
    )

    reward[:] = torch.where(
        crashes > 0,
        parameter_dict["collision_penalty"] * torch.ones_like(reward),
        reward,
    )
    return reward, crashes
