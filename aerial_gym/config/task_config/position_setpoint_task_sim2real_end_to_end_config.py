import torch

EVAL = False

if EVAL == False:
    class task_config:
        seed = 56 #16 #26 #36 #46 #56
        sim_name = "base_sim"
        env_name = "empty_env"
        robot_name = "lmf1"
        controller_name = "no_control"
        args = {}
        num_envs = 24
        use_warp = False
        headless = True
        device = "cuda:0"
        privileged_observation_space_dim = 0
        action_space_dim = 4
        observation_space_dim = 15
        episode_len_steps = 500
        return_state_before_reset = False
        reward_parameters = { }
        crash_dist = 6.5

        action_limit_max = torch.ones(action_space_dim,device=device) * 6.0
        action_limit_min = torch.ones(action_space_dim,device=device) * 0.0

        def process_actions_for_task(actions, min_limit, max_limit):
            actions_clipped = torch.clamp(actions, -1, 1)

            rescaled_command_actions = actions_clipped * (max_limit - min_limit)/2 + (max_limit + min_limit)/2

            return rescaled_command_actions
else:
    class task_config:
        seed = 41
        sim_name = "base_sim_4ms"
        env_name = "empty_env"
        robot_name = "lmf1"
        controller_name = "no_control"
        args = {}
        num_envs = 4096
        use_warp = False
        headless = True
        device = "cuda:0"
        privileged_observation_space_dim = 0
        action_space_dim = 4
        observation_space_dim = 15
        episode_len_steps = 10000
        return_state_before_reset = False
        reward_parameters = { }

        crash_dist = 5.5

        action_limit_max = torch.ones(action_space_dim,device=device) * 6.0
        action_limit_min = torch.ones(action_space_dim,device=device) * 0.0

        def process_actions_for_task(actions, min_limit, max_limit):
            actions_clipped = torch.clamp(actions, -1, 1)

            rescaled_command_actions = actions_clipped * (max_limit - min_limit)/2 + (max_limit + min_limit)/2

            return rescaled_command_actions

