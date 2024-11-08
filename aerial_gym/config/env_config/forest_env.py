from aerial_gym.config.asset_config.env_object_config import (
    tree_asset_params,
    object_asset_params,
    bottom_wall,
)

import numpy as np


class ForestEnvCfg:
    class env:
        num_envs = 64
        num_env_actions = 4  # this is the number of actions handled by the environment
        # potentially some of these can be input from the RL agent for the robot and
        # some of them can be used to control various entities in the environment
        # e.g. motion of obstacles, etc.
        env_spacing = 5.0  # not used with heightfields/trimeshes

        num_physics_steps_per_env_step_mean = 10  # number of steps between camera renders mean
        num_physics_steps_per_env_step_std = 0  # number of steps between camera renders std

        render_viewer_every_n_steps = 1  # render the viewer every n steps
        reset_on_collision = (
            True  # reset environment when contact force on quadrotor is above a threshold
        )
        collision_force_threshold = 0.005  # collision force threshold [N]
        create_ground_plane = False  # create a ground plane
        sample_timestep_for_latency = True  # sample the timestep for the latency noise
        perturb_observations = True
        keep_same_env_for_num_episodes = 1
        write_to_sim_at_every_timestep = False  # write to sim at every timestep

        use_warp = True
        lower_bound_min = [-5.0, -5.0, -1.0]  # lower bound for the environment space
        lower_bound_max = [-5.0, -5.0, -1.0]  # lower bound for the environment space
        upper_bound_min = [5.0, 5.0, 3.0]  # upper bound for the environment space
        upper_bound_max = [5.0, 5.0, 3.0]  # upper bound for the environment space

    class env_config:
        include_asset_type = {
            "trees": True,
            "objects": True,
            "bottom_wall": True,
        }

        # maps the above names to the classes defining the assets. They can be enabled and disabled above in include_asset_type
        asset_type_to_dict_map = {
            "trees": tree_asset_params,
            "objects": object_asset_params,
            "bottom_wall": bottom_wall,
        }
