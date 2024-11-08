import torch
from aerial_gym.utils.math import torch_interpolate_ratio


class task_config:
    seed = 1
    sim_name = "base_sim_2ms"
    env_name = "empty_env_2ms"
    controller_name = "no_control"
    args = {}
    num_envs = 1024
    use_warp = False
    headless = False
    device = "cuda:0"
    episode_len_steps = 500  # real physics time for simulation is this value multiplied by sim.dt
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

    # robot_name = "morphy_stiff"
    # num_joints = 0
    # num_motors = 4
    # action_space_dim = 4
    # observation_space_dim = 13+action_space_dim+num_joints*2
    # privileged_observation_space_dim = 0
    # # # for velocity targets
    # action_limit_max = [2.0]*num_motors
    # action_limit_min = [0.0]*num_motors

    robot_name = "morphy"
    num_joints = 8
    num_motors = 4
    action_space_dim = 4
    observation_space_dim = 13 + action_space_dim + num_joints * 2
    privileged_observation_space_dim = 0
    # # for velocity targets
    action_limit_max = [2.0] * num_motors
    action_limit_min = [0.0] * num_motors

    def process_actions_for_task(actions, min_limit, max_limit):
        actions_clipped = torch.clamp(actions, 0, 1)
        scaled_actions = torch_interpolate_ratio(
            min=min_limit, max=max_limit, ratio=actions_clipped
        )
        return scaled_actions
