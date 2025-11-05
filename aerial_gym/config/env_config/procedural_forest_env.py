from aerial_gym.config.asset_config.env_object_config import (
    target_asset_params,
    tree_asset_params,
)


class ProceduralForestEnvCfg:
    """Configurable forest environment with procedural terrain generation."""

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
        reset_on_collision = True  # reset environment when contact force on quadrotor is above a threshold
        collision_force_threshold = 0.005  # collision force threshold [N]
        create_ground_plane = True  # create a ground plane
        sample_timestep_for_latency = True  # sample the timestep for the latency noise
        perturb_observations = True
        keep_same_env_for_num_episodes = 1
        write_to_sim_at_every_timestep = True  # write to sim at every timestep

        use_warp = True

        # Tree density: trees per square meter (num_assets = tree_density * env_area)
        tree_density = 0.004  # 40 trees per 100m × 100m environment

        # Terrain generation configuration (Simplex noise)
        enable_terrain = True
        terrain_resolution = 256
        terrain_amplitude = 24.0  # Total height range (terrain offset to start at z=0)
        terrain_frequency = 0.3  # Base frequency (lower=smooth, higher=rough)
        terrain_octaves = 8  # Number of noise layers
        terrain_lacunarity = 2.0  # Frequency multiplier per octave
        terrain_persistence = 0.6  # Amplitude multiplier per octave
        # Terrain seed: None = random per environment (fixed across resets due to Isaac Gym limitation)
        terrain_seed = None

        # Environment bounds [x_min, y_min, z_min] to [x_max, y_max, z_max] in meters
        # Terrain starts at z=0 and extends up to terrain_amplitude
        lower_bound_min = [-50.0, -50.0, 0.0]
        lower_bound_max = [-50.0, -50.0, 0.0]
        upper_bound_min = [50.0, 50.0, 30.0]
        upper_bound_max = [50.0, 50.0, 30.0]

        # Target movement configuration
        target_velocity_change_interval = 100  # Steps between velocity changes
        target_stop_probability = 0.15  # Chance to stop moving per interval
        target_velocity_max = 1.0  # Maximum velocity magnitude in m/s
        target_velocity_min = 0.3  # Minimum velocity when moving in m/s

    class env_config:
        include_asset_type = {
            "trees": True,
            "target": True,
        }

        asset_type_to_dict_map = {
            "trees": tree_asset_params,
            "target": target_asset_params,
        }
