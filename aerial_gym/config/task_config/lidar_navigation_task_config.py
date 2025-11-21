import torch
from aerial_gym import AERIAL_GYM_DIRECTORY


class task_config:
    seed = -1
    sim_name = "base_sim"
    env_name = "env_with_obstacles"
    robot_name = "magpie"
    # controller_name = "lmf2_acceleration_control"
    controller_name = "lmf2_velocity_control"
    args = {}
    num_envs = 512
    use_warp = True
    headless = False
    device = "cuda:0"
    observation_space_dim = 13 + 4 + 16*20  # root_state + action_dim _+ latent_dims
    privileged_observation_space_dim = 0
    action_space_dim = 4
    episode_len_steps = 110  # real physics time for simulation is this value multiplied by sim.dt

    return_state_before_reset = (
        False  # False as usually state is returned for next episode after reset
    )
    # user can set the above to true if they so desire

    target_min_ratio = [0.90, 0.15, 0.15]  # target ratio w.r.t environment bounds in x,y,z
    target_max_ratio = [0.92, 0.80, 0.80]  # target ratio w.r.t environment bounds in x,y,z

    reward_parameters = {
        "pos_reward_magnitude": 3.0,
        "pos_reward_exponent": 1.0,
        "very_close_to_goal_reward_magnitude": 5.0,
        "very_close_to_goal_reward_exponent": 8.0,
        "vel_direction_component_reward_magnitude": 1.0,
        "x_action_diff_penalty_magnitude": 0.1,
        "x_action_diff_penalty_exponent": 5.0,
        "y_action_diff_penalty_magnitude": 0.1,
        "y_action_diff_penalty_exponent": 5.0,
        "z_action_diff_penalty_magnitude": 0.1,
        "z_action_diff_penalty_exponent": 5.0,
        "yawrate_action_diff_penalty_magnitude": 0.1,
        "yawrate_action_diff_penalty_exponent": 5.0,
        "x_absolute_action_penalty_magnitude": 0.1,
        "x_absolute_action_penalty_exponent": 0.3,
        "y_absolute_action_penalty_magnitude": 0.1,
        "y_absolute_action_penalty_exponent": 0.3,
        "z_absolute_action_penalty_magnitude": 0.15,
        "z_absolute_action_penalty_exponent": 1.0,
        "yawrate_absolute_action_penalty_magnitude": 0.15,
        "yawrate_absolute_action_penalty_exponent": 2.0,
        "collision_penalty": -10.0,
    }

    class vae_config:
        use_vae = False
        latent_dims = 64
        model_file = (
            AERIAL_GYM_DIRECTORY
            + "/aerial_gym/utils/vae/weights/ICRA_test_set_more_sim_data_kld_beta_3_LD_64_epoch_49.pth"
        )
        model_folder = AERIAL_GYM_DIRECTORY
        image_res = (270, 480)
        interpolation_mode = "nearest"
        return_sampled_latent = True

    class curriculum:
        min_level = 25
        max_level = 70
        check_after_log_instances = 2048
        increase_step = 2
        decrease_step = 1
        success_rate_for_increase = 0.7
        success_rate_for_decrease = 0.6

        def update_curriculim_level(self, success_rate, current_level):
            if success_rate > self.success_rate_for_increase:
                return min(current_level + self.increase_step, self.max_level)
            elif success_rate < self.success_rate_for_decrease:
                return max(current_level - self.decrease_step, self.min_level)
            return current_level



    def action_transformation_function(action):
        clamped_action = torch.clamp(action, -1.0, 1.0)
        max_yawrate = torch.pi / 3  # [rad/s]

        processed_action = torch.zeros(
            (clamped_action.shape[0], 4), device=task_config.device, requires_grad=False
        )

        processed_action[:, 0] = -(clamped_action[:, 0]+1.0)
        processed_action[:, 1] = clamped_action[:, 1]
        processed_action[:, 2] = clamped_action[:, 2]
        processed_action[:, 3] = clamped_action[:, 3] * max_yawrate
        return processed_action

    # def action_transformation_function(action):
    #     clamped_action = torch.clamp(action, -1.0, 1.0)
    #     max_yawrate = torch.pi / 3  # [rad/s]

    #     processed_action = torch.zeros(
    #         (clamped_action.shape[0], 4), device=task_config.device, requires_grad=False
    #     )

    #     processed_action[:, 0:3] = 2 * (clamped_action[:, 0:3])
    #     # processed_action[:, 1] = clamped_action[:, 1]
    #     # processed_action[:, 2] = clamped_action[:, 2]
    #     processed_action[:, 3] = clamped_action[:, 3] * max_yawrate
    #     return processed_action
