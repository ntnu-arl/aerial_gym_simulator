import torch


class task_config:
    seed = 1
    sim_name = "base_sim"
    env_name = "empty_env"
    robot_name = "lmf2"
    controller_name = "lmf2_acceleration_control"
    args = {}
    num_envs = 16
    use_warp = False
    headless = False
    device = "cuda:0"
    observation_space_dim = 17
    privileged_observation_space_dim = 0
    action_space_dim = 4
    episode_len_steps = 800  # real physics time for simulation is this value multiplied by sim.dt
    return_state_before_reset = False
    reward_parameters = {}
