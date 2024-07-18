# Simulation Components

The simulator is made of composable entities that allow changing the physics engine parameters, adding various environment handling functionalities to manipulate the environment at runtime, load the assets, select robots and controllers and design custom sensors.

## Registries

The code contains registries for having a named mapping to various configuration files and classes. The registries allow ease of configuration through parameters and allow to mix-and-match various settings, robots, environments, sensors within the simulation. New configurations can be created on-the-fly and registered to be used programmatically by the active code, without having to stop to manually configure the simulation. 

The code contains registries for the following components:

[TOC]


TO register your own custom components, please take a look at the [Customization](./5_customization.md) page.


## Simulation Parameters

We use NVIDIA's Isaac Gym as the simulation engine. The simulator allows for selection of different physics backends, such as PhysX and Flex. The simulator provides a set of APIs to interact with the environment, such as setting the gravity, time step, and rendering options. We provide a set of default configurations for the physics engine (based on PhysX) that can be set as per the user's needs. The simulation params are set here: 

??? example "Default simulation parameters"
    ```python
    class BaseSimConfig:
    # viewer camera:
    class viewer:
        headless = False
        ref_env = 0
        camera_position = [-5, -5, 4]  # [m]
        lookat = [0, 0, 0]  # [m]
        camera_orientation_euler_deg = [0, 0, 0]  # [deg]
        camera_follow_type = "FOLLOW_TRANSFORM"
        width = 1280
        height = 720
        max_range = 100.0  # [m]
        min_range = 0.1
        horizontal_fov_deg = 90
        use_collision_geometry = False
        camera_follow_transform_local_offset = [-1.0, 0.0, 0.2]  # m
        camera_follow_position_global_offset = [-1.0, 0.0, 0.4]  # m

    class sim:
        dt = 0.01
        substeps = 1
        gravity = [0.0, 0.0, -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z
        use_gpu_pipeline = True

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 2
            contact_offset = 0.002  # [m]
            rest_offset = 0.001  # [m]
            bounce_threshold_velocity = 0.1  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**24  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 10
            contact_collection = 1  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
    ```

While the default setting of the physics engine is PhysX for this simulator, this can be changed to Flex with minor modifications. It has not been tested for this purpose.

## Assets

Simulation assets in Aerial Gym are generally URDF files that can each have their own parameters for simulation. The asset configuration files for assets are stored in `config/asset_config`. In our implementation we have defined asset classes based on the type of the asset of our purposes and ground their properties together. Assets of each type can be represented through different URDF files inherently enabling randomization across environments without a further need to specify which asset is to be loaded. Just adding additional URDFs to the appropriate folder path suffices to add them in the pool for selection fr simulation.

The parameters for each asset are derived from the `BaseAssetParams` class that includes the number of assets of that type to be loaded per environment, specifies root asset folder, specifies the position, orientation ratios and physics properties of the asset such as damping coefficients, density etc. Additional parameters can be used to specify properties such as presence of force sensors on the asset, per-link or whole body segmentation labels, etc.

The `BaseAssetParams` file is as follows:

??? example "Example of configurable `BaseAssetParams` class"
    ```python
    class BaseAssetParams:
        num_assets = 1  # number of assets to include per environment

        asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets"
        file = None  # if file=None, random assets will be selected. If not None, this file will be used

        min_position_ratio = [0.5, 0.5, 0.5]  # min position as a ratio of the bounds
        max_position_ratio = [0.5, 0.5, 0.5]  # max position as a ratio of the bounds

        collision_mask = 1

        disable_gravity = False
        replace_cylinder_with_capsule = True  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = (
            True  # Some .obj meshes must be flipped from y-up to z-up
        )
        density = 0.000001
        angular_damping = 0.0001
        linear_damping = 0.0001
        max_angular_velocity = 100.0
        max_linear_velocity = 100.0
        armature = 0.001

        collapse_fixed_joints = True
        fix_base_link = True
        color = None
        keep_in_env = False

        body_semantic_label = 0
        link_semantic_label = 0
        per_link_semantic = False
        semantic_masked_links = {}
        place_force_sensor = False
        force_sensor_parent_link = "base_link"
        force_sensor_transform = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]  # position, quat x, y, z, w
        use_collision_mesh_instead_of_visual = False
    ```


## Environments

Environment specification determines what are the components in a simulation environment. A configuration file can be used to select a particular robot (with it's sensors), a specified controller for thr robot and selection of obstacles that are present in the environment alongside a strategy to spawn and randomize their positions w.r.t the environment bounds. The environment configuration files are stored in `config/env_config` folder. The environment manager calls each of the environment entities to allow specific user-coded behaviors at each timestep or to perform certain actions when the environment is reset. The environment manager is responsible for spawning the assets, robots and obstacles in the environment and managing their interactions.

??? example "Example of an Environment Configuration for an empty environment with a robot"
    ```python
    class EmptyEnvCfg:
        class env:
            num_envs = 3  # number of environments
            num_env_actions = 0  # this is the number of actions handled by the environment
            # these are the actions that are sent to environment entities
            # and some of them may be used to control various entities in the environment
            # e.g. motion of obstacles, etc.
            env_spacing = 1.0  # not used with heightfields/trimeshes
            num_physics_steps_per_env_step_mean = 1  # number of steps between camera renders mean
            num_physics_steps_per_env_step_std = 0  # number of steps between camera renders std
            render_viewer_every_n_steps = 10  # render the viewer every n steps
            collision_force_threshold = 0.010  # collision force threshold
            manual_camera_trigger = False  # trigger camera captures manually
            reset_on_collision = (
                True  # reset environment when contact force on quadrotor is above a threshold
            )
            create_ground_plane = False  # create a ground plane
            sample_timestep_for_latency = True  # sample the timestep for the latency noise
            perturb_observations = True
            keep_same_env_for_num_episodes = 1

            use_warp = False
            e_s = env_spacing
            lower_bound_min = [-e_s, -e_s, -e_s]  # lower bound for the environment space
            lower_bound_max = [-e_s, -e_s, -e_s]  # lower bound for the environment space
            upper_bound_min = [e_s, e_s, e_s]  # upper bound for the environment space
            upper_bound_max = [e_s, e_s, e_s]  # upper bound for the environment space

        class env_config:
            include_asset_type = {}

            asset_type_to_dict_map = {}
    ```

In order to add assets to the environment, the `include_asset_type` dictionary can be used to specify the assets that are to be included in the environment. The `asset_type_to_dict_map` dictionary maps the asset type to the class defining the asset parameters.


??? example "Environment configuration file for an environment with obstacles"
    for the case of an environment with obstacles can be seen below:
    ```python3
    from aerial_gym.config.env_config.env_object_config import EnvObjectConfig

    import numpy as np


    class EnvWithObstaclesCfg(EnvObjectConfig):
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
            collision_force_threshold = 0.05  # collision force threshold [N]
            create_ground_plane = False  # create a ground plane
            sample_timestep_for_latency = True  # sample the timestep for the latency noise
            perturb_observations = True
            keep_same_env_for_num_episodes = 1

            use_warp = True
            lower_bound_min = [-2.0, -4.0, -3.0]  # lower bound for the environment space
            lower_bound_max = [-1.0, -2.5, -2.0]  # lower bound for the environment space
            upper_bound_min = [9.0, 2.5, 2.0]  # upper bound for the environment space
            upper_bound_max = [10.0, 4.0, 3.0]  # upper bound for the environment space

        class env_config:
            include_asset_type = {
                "panels": True,
                "thin": False,
                "trees": False,
                "objects": True,
                "left_wall": True,
                "right_wall": True,
                "back_wall": True,
                "front_wall": True,
                "top_wall": True,
                "bottom_wall": True,
            }

            # maps the above names to the classes defining the assets. They can be enabled and disabled above in include_asset_type
            asset_type_to_dict_map = {
                "panels": EnvObjectConfig.panel_asset_params,
                "thin": EnvObjectConfig.thin_asset_params,
                "trees": EnvObjectConfig.tree_asset_params,
                "objects": EnvObjectConfig.object_asset_params,
                "left_wall": EnvObjectConfig.left_wall,
                "right_wall": EnvObjectConfig.right_wall,
                "back_wall": EnvObjectConfig.back_wall,
                "front_wall": EnvObjectConfig.front_wall,
                "bottom_wall": EnvObjectConfig.bottom_wall,
                "top_wall": EnvObjectConfig.top_wall,
            }
    ```

## Tasks

An environment specification determines what is populated in an independent simulation instance and how the collective simulation steps through time based on commanded actions. The task here is however slightly different. We intend to use this term of interpreting task-specific information from the environment. A task class instantiates the entire simulation with all its parallel robots and assets and therefore, has access to all the simulation information. We intend to use this class to determine how the environment is interpreted for RL tasks. For example, a given simulation instance with sim params, environment and asset specification, robot, sensors and controller specifications, can be utilized to train a policy to perform completely different tasks. Example of these could include setpoint navigation through clutter, observing a specific asset in the simulation, perching on that specific asset in simulation and so on and so forth. All these tasks can be performed with the same set of objects in the environment, but require a different interpretation of the environment data for training the RL algorithm appropriately. This can be done within the Task classes. A task can be specified in the files in the `config/task_config` folder as:

??? example "Example of a Task Configuration"
    ```python
    class task_config:
        seed = 10
        sim_name = "base_sim"
        env_name = "empty_env"
        robot_name = "base_quadrotor"
        args = {}
        num_envs = 2
        device = "cuda:0"
        observation_space_dim = 13
        privileged_observation_space_dim = 0
        action_space_dim = 4
        episode_len_steps = 1000 # real physics time for simulation is this value multiplied by sim.dt
        return_state_before_reset = False
        reward_parameters = {
            "pos_error_gain1": [2.0, 2.0, 2.0],
            "pos_error_exp1": [1/3.5, 1/3.5, 1/3.5],
            "pos_error_gain2": [2.0, 2.0, 2.0],
            "pos_error_exp2": [2.0, 2.0, 2.0],
            "dist_reward_coefficient": 7.5,
            "max_dist": 15.0,
            "action_diff_penalty_gain": [1.0, 1.0, 1.0],
            "absolute_action_reward_gain": [2.0, 2.0, 2.0],
            "crash_penalty": -100,
        }

        # a + bx for action scaling
        consant_for_action = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)

        scale_for_action = torch.tensor([3.0, 3.0, 3.0, 1.50], dtype=torch.float32, device=device)


        def action_transformation_function(action):
            clamped_action = torch.clamp(action, -1.0, 1.0)
            return task_config.consant_for_action + task_config.scale_for_action * clamped_action     
    ```

A sample task for position setpoint navigation (without sensors or obstacles) is provided as an example:

??? example "`PositionSetpointTask` class definition example"
    ```python
    class PositionSetpointTask(BaseTask):
        def __init__(self, task_config):
            super().__init__(task_config)
            self.device = self.task_config.device
            # set the each of the elements of reward parameter to a torch tensor
            # common boilerplate code here

            # Currently only the "observations" are sent to the actor and critic.
            # The "privileged_obs" are not handled so far in sample-factory

            self.task_obs = {
                "observations": torch.zeros(
                    (self.sim_env.num_envs, self.task_config.observation_space_dim),
                    device=self.device,
                    requires_grad=False,
                ),
                "priviliged_obs": torch.zeros(
                    (self.sim_env.num_envs, self.task_config.privileged_observation_space_dim),
                    device=self.device,
                    requires_grad=False,
                ),
                "collisions": torch.zeros(
                    (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
                ),
                "rewards": torch.zeros(
                    (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
                ),
            }

        # common boilerplate code here

        def step(self, actions):
            # this uses the action, gets observations
            # calculates rewards, returns tuples
            # In this case, the episodes that are terminated need to be
            # first reset, and the first observation of the new episode
            # needs to be returned.

            transformed_action = self.action_transformation_function(actions)
            self.sim_env.step(actions=transformed_action)
            
            # This step must be done since the reset is done after the reward is calculated.
            # This enables the robot to send back an updated state, and an updated observation to the RL agent after the reset.
            # This is important for the RL agent to get the correct state after the reset.
            self.rewards[:], self.terminations[:] = self.compute_rewards_and_crashes(self.obs_dict)

            if self.task_config.return_state_before_reset == True:
                return_tuple = self.get_return_tuple()


            self.truncations[:] = torch.where(self.sim_env.sim_steps > self.task_config.episode_len_steps, 1, 0)
            self.sim_env.post_reward_calculation_step()

            self.infos = {}  # self.obs_dict["infos"]

            if self.task_config.return_state_before_reset == False:
                return_tuple = self.get_return_tuple()
            return return_tuple
            
        ...

        def process_obs_for_task(self):
            self.task_obs["observations"][:, 0:3] = (
                self.target_position - self.obs_dict["robot_position"] # position in environment/world frame
            )
            self.task_obs["observations"][:, 3:7] = self.obs_dict["robot_orientation"] # orientation in environment/world frame
            self.task_obs["observations"][:, 7:10] = self.obs_dict["robot_body_linvel"] # linear velocity in the body/imu frame
            self.task_obs["observations"][:, 10:13] = self.obs_dict["robot_body_angvel"] # angular velocity in the body/imu frame
            self.task_obs["rewards"] = self.rewards # reward for the time step after it is calculated
            self.task_obs["terminations"] = self.terminations # terminations/crashes 
            self.task_obs["truncations"] = self.truncations # truncations or premature resets (for purposes of diversifying data and episodes)


    @torch.jit.script
    def exp_func(x, gain, exp):
        return gain * torch.exp(-exp * x * x)


    @torch.jit.script
    def compute_reward(
        pos_error, crashes, action, prev_action, curriculum_level_multiplier, parameter_dict
    ):
        # type: (Tensor, Tensor, Tensor, Tensor, float, Dict[str, Tensor]) -> Tuple[Tensor, Tensor]
        dist = torch.norm(pos_error, dim=1)

        pos_reward = 2.0 / ( 1.0 + dist * dist)

        dist_reward = (20 - dist) / 20.0

        total_reward = (
            pos_reward + dist_reward # + up_reward + action_diff_reward + absolute_action_reward
        )
        total_reward[:] = curriculum_level_multiplier * total_reward
        crashes[:] = torch.where(dist > 8.0, torch.ones_like(crashes), crashes)

        total_reward[:] = torch.where(
            crashes > 0.0, -2 * torch.ones_like(total_reward), total_reward
        )
        return total_reward, crashes
    ```

The Task class is ultimately designed to be used with RL frameworks, therefore conforms to the [Gymnasium API](https://gymnasium.farama.org) specification. In this above class, the `step(...)` function first translates the commands from the RL agent to a control command for the specific robot platform by transforming the action input. Subsequently this is commanded to the robot and the environment is stepped. Finally, a reward is computed for the new state and [truncations and terminations](https://farama.org/Gymnasium-Terminated-Truncated-Step-API) are determined and the final tuple is returned for use by the RL framework. Similarly, for trajectory tracking, only the reward function and observation needs to be changed to train the RL algorithm without making any changes to the asset, robots or the environment.

To add your own [custom tasks](./5_customization.md/#custom-tasks) please refer to the section on [customizing the simulator](./5_customization.md).


??? question "**Difference between Environment and Task**"
    A lot of different simulator implementations interchange the terminologies. In our case, we view the environment as the components that define the robot, it's physics surroundings, i.e., the assets near the robot, the parameters of the physics engine that detemine how the various entities in the simulation world interact with one another, and how sensors perceive the data via the sensor parameters.
    
    The task on the other hand is an interpretation of the simulation world and the information provided by / collected from it to reach a particular goal that is desired by the user. The same environment can be used to train multiple tasks and the tasks can be changed without changing an environment definition.
    
    For example, an empty environment with a quadrotor can be used to train a position setpoint task, or a trajectory tracking task. An environment with a set of obstacles can be used to train a policy that can either navigate through the obstacles or perch on a specific asset in the environment. The task is the interpretation of the environment data for the RL algorithm to learn the desired behavior.

    To relate it with a familiar environment from the OpenAI Gym suite of tasks, an "environment" in our case could refer to a CartPole world with its associated dynamics, however a "task" in our case would allow the same cartpole to be controlled to balance the pole upright, or to keep swinging the pole at a given angular rate or to have the endpoint of the pole at a given location in the environment. All of which require different formulation of rewards and observations for the RL algorithm to learn the desired behavior.

## Robots

Robots can be specified and configured independently of the sensors and the environment using a robot [registry](#registries). More about robots can be found on the page for [Robots and Controllers](./3_robots_and_controllers.md/#robots)


## Controllers

Controllers can be specified and selected independently of the robot platform. Note however, that all combinations of controllers will not produce optimal results with all platforms. The controllers can be registered and selected from the controller registry. More about controllers can be found on the page for [Robots and Controllers](./3_robots_and_controllers.md/#controllers).

## Sensors

Similar to the above, the sensors can be specified and selected independently. However since the sensors are mounted on the robot platform, we have made a choice to select the sensors for the robot in the robot config file, and not directly as a registry (it is possible to do so yourself with very minor changes to the code). More about the capabilities of the sensors can be found on the page for [Sensors and Rendering](./8_sensors_and_rendering.md).