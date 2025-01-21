import torch

from aerial_gym.utils.math import *


class task_config:
    seed = 1
    sim_name = "base_sim"
    env_name = "empty_env"
    robot_name = "tinyprop"
    controller_name = "no_control"
    args = {}
    num_envs = 16384
    use_warp = False
    headless = False
    device = "cuda:0"
    observation_space_dim = 17
    privileged_observation_space_dim = 0
    action_space_dim = 4
    episode_len_steps = 600  # real physics time for simulation is this value multiplied by sim.dt
    return_state_before_reset = False
    reward_parameters = {
        "pos_error_gain1": [2.0, 2.0, 2.0],
        "pos_error_exp1": [1 / 3.5, 1 / 3.5, 1 / 3.5],
        "pos_error_gain2": [2.0, 2.0, 2.0],
        "pos_error_exp2": [2.0, 2.0, 2.0],
        "dist_reward_coefficient": 7.5,
        "max_dist": 15.0,
        "action_diff_penalty_gain": [1.0, 1.0, 1.0],
        "absolute_action_reward_gain": [2.0, 2.0, 2.0],
        "crash_penalty": -100,
    }

    def action_transformation_function(action):
        return 0.2 + 0.5*(1.0 + torch.clamp(action, -1.0, 1.0))
