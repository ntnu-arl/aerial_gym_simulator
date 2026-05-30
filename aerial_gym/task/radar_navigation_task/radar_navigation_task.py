from aerial_gym.task.lidar_navigation_task.lidar_navigation_task import *


class RadarNavigationTask(LiDARNavigationTask):
    
    def add_noise_to_downsampled_lidar_data(self, ds_lidar_data):
        # random noise to 3% pixels
        noise_mask = torch.bernoulli(
            0.03 * torch.ones_like(ds_lidar_data)).to(self.device)
        ds_lidar_data[noise_mask == 1] += torch_rand_float_tensor(
            0.2 * torch.ones_like(noise_mask[noise_mask == 1]),
            10.0 * torch.ones_like(noise_mask[noise_mask == 1])
        ).to(self.device)

        # bernoulli sampling to have 1-2% points max range
        invalid_points_mask = torch.bernoulli(
            0.8 * torch.ones_like(ds_lidar_data)).to(self.device)
        # set the 
        ds_lidar_data[invalid_points_mask == 1] = -1.0

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
        
        # Min pooling to downsample the image
        image_obs_ds = -torch.nn.functional.max_pool2d(
            -image_obs.unsqueeze(1), (3, 6)).squeeze(1)
        
        # Noise after min pooling
        image_obs_noisy = self.add_noise_to_downsampled_lidar_data(image_obs_ds)
        inv_range_image = 1 / image_obs_noisy
        self.downsampled_lidar_data[:] = inv_range_image.reshape(
            (self.num_envs, -1)).to(self.device)

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
        torch.clamp(robot_vehicle_linvel[:, 0], max=0.0),
    ) * close_to_goal

    vel_penalty_for_acc = vel_magnitude_penalty + negative_x_vel_penalty

    # stable at goal reward 
    low_vel_reward = erf(1.5, 10.0, robot_vel_norm) + erf(1.5, 0.5, robot_vel_norm)

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
