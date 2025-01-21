import torch
from aerial_gym.utils.math import torch_interpolate_ratio

EVAL = True

if EVAL == False:
    class task_config:
        seed = 56 #46 #36 #16 #26 #56
        sim_name = "base_sim"
        env_name = "empty_env"
        robot_name = "tinyprop"
        controller_name = "no_control"
        args = {}
        num_envs = 4096
        use_warp = False
        headless = True
        device = "cuda:0"
        privileged_observation_space_dim = 0
        action_space_dim = 4
        observation_space_dim = 15
        episode_len_steps = 600 #2000 #800 for dt = 0.01  # real physics time for simulation is this value multiplied by sim.dt
        return_state_before_reset = False
        reward_parameters = { }
        crash_dist = 1.5
        
        action_limit_max = torch.ones(action_space_dim,device=device) * 1.2
        action_limit_min = torch.ones(action_space_dim,device=device) * 0.2

        def process_actions_for_task(actions, min_limit, max_limit):
            actions_clipped = torch.clamp(actions, -1, 1)
            
            rescaled_command_actions = actions_clipped * (max_limit - min_limit)/2 + (max_limit + min_limit)/2
            
            return rescaled_command_actions
else:
    class task_config:
        seed = 41
        sim_name = "base_sim"
        env_name = "empty_env"
        robot_name = "tinyprop"
        controller_name = "no_control" #"split_architecture"
        args = {}
        num_envs = 4096
        use_warp = False
        headless = False
        device = "cuda:0"
        privileged_observation_space_dim = 0
        action_space_dim = 4
        observation_space_dim = 15 
        episode_len_steps = 10000 #2000 #800 for dt = 0.01  # real physics time for simulation is this value multiplied by sim.dt
        return_state_before_reset = False
        reward_parameters = { }
        
        crash_dist = 5.5
        
        action_limit_max = torch.ones(action_space_dim,device=device) * 1.2
        action_limit_min = torch.ones(action_space_dim,device=device) * 0.2

        def process_actions_for_task(actions, min_limit, max_limit):
            actions_clipped = torch.clamp(actions, -1, 1)
            
            rescaled_command_actions = actions_clipped * (max_limit - min_limit)/2 + (max_limit + min_limit)/2
            
            return rescaled_command_actions

