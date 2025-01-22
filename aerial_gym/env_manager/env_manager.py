from aerial_gym.env_manager.IGE_env_manager import IsaacGymEnv

from aerial_gym.env_manager.base_env_manager import BaseManager
from aerial_gym.env_manager.asset_manager import AssetManager
from aerial_gym.env_manager.warp_env_manager import WarpEnv
from aerial_gym.env_manager.asset_loader import AssetLoader
from aerial_gym.robots.robot_manager import RobotManagerIGE
from aerial_gym.env_manager.obstacle_manager import ObstacleManager


from aerial_gym.registry.env_registry import env_config_registry
from aerial_gym.registry.sim_registry import sim_config_registry
from aerial_gym.registry.robot_registry import robot_registry

import torch

from aerial_gym.utils.logging import CustomLogger

import math, random

logger = CustomLogger("env_manager")


class EnvManager(BaseManager):
    """
    This class manages the environment. This can handle the creation of the
    robot, environment, and asset managers. This class handles the tensor creation and destruction.

    Moreover, the environment manager can be called within the main environment
    class to manipulate the environment by abstracting the interface.

    This script can remain as generic as possible to handle different types of
    environments, while changes can be made in the individual robot or environment
    managers to handle specific cases.
    """

    def __init__(
        self,
        sim_name,
        env_name,
        robot_name,
        controller_name,
        device,
        args=None,
        num_envs=None,
        use_warp=None,
        headless=None,
    ):
        self.robot_name = robot_name
        self.controller_name = controller_name
        self.sim_config = sim_config_registry.make_sim(sim_name)

        super().__init__(env_config_registry.make_env(env_name), device)

        if num_envs is not None:
            self.cfg.env.num_envs = num_envs
        if use_warp is not None:
            self.cfg.env.use_warp = use_warp
        if headless is not None:
            self.sim_config.viewer.headless = headless

        self.num_envs = self.cfg.env.num_envs
        self.use_warp = self.cfg.env.use_warp

        self.asset_manager = None
        self.tensor_manager = None
        self.env_args = args

        self.keep_in_env = None

        self.global_tensor_dict = {}

        logger.info("Populating environments.")
        self.populate_env(env_cfg=self.cfg, sim_cfg=self.sim_config)
        logger.info("[DONE] Populating environments.")
        self.prepare_sim()

        self.sim_steps = torch.zeros(
            self.num_envs, dtype=torch.int32, requires_grad=False, device=self.device
        )

    def create_sim(self, env_cfg, sim_cfg):
        """
        This function creates the environment and the robot manager. Does the necessary things to create the environment
        for an IsaacGym environment instance.
        """
        logger.info("Creating simulation instance.")
        logger.info("Instantiating IGE object.")

        # === Need to check this here otherwise IGE will crash with segfault for different CUDA GPUs ====
        has_IGE_cameras = False
        robot_config = robot_registry.get_robot_config(self.robot_name)
        if robot_config.sensor_config.enable_camera == True and self.use_warp == False:
            has_IGE_cameras = True
        # ===============================================================================================

        self.IGE_env = IsaacGymEnv(env_cfg, sim_cfg, has_IGE_cameras, self.device)

        # define a global dictionary to store the simulation objects and important parameters
        # that are shared across the environment, asset, and robot managers
        self.global_sim_dict = {}
        self.global_sim_dict["gym"] = self.IGE_env.gym
        self.global_sim_dict["sim"] = self.IGE_env.sim
        self.global_sim_dict["env_cfg"] = self.cfg
        self.global_sim_dict["use_warp"] = self.IGE_env.cfg.env.use_warp
        self.global_sim_dict["num_envs"] = self.IGE_env.cfg.env.num_envs
        self.global_sim_dict["sim_cfg"] = sim_cfg

        logger.info("IGE object instantiated.")

        if self.cfg.env.use_warp:
            logger.info("Creating warp environment.")
            self.warp_env = WarpEnv(self.global_sim_dict, self.device)
            logger.info("Warp environment created.")

        self.asset_loader = AssetLoader(self.global_sim_dict, self.device)

        logger.info("Creating robot manager.")
        self.robot_manager = RobotManagerIGE(
            self.global_sim_dict, self.robot_name, self.controller_name, self.device
        )
        self.global_sim_dict["robot_config"] = self.robot_manager.cfg
        logger.info("[DONE] Creating robot manager.")

        logger.info("[DONE] Creating simulation instance.")

    def populate_env(self, env_cfg, sim_cfg):
        """
        This function populates the environment with the necessary assets and robots.
        """
        # create the simulation instance with the environment and robot manager
        self.create_sim(env_cfg, sim_cfg)

        self.robot_manager.create_robot(self.asset_loader)

        # first select assets for the environments:
        self.global_asset_dicts, keep_in_env_num = self.asset_loader.select_assets_for_sim()

        if self.keep_in_env == None:
            self.keep_in_env = keep_in_env_num
        elif self.keep_in_env != keep_in_env_num:
            raise Exception(
                "Inconsistent number of assets kept in the environment. The number of keep_in_env assets must be equal for all environments. Check."
            )

        # add the assets to the environment
        segmentation_ctr = 100

        self.global_asset_counter = 0
        self.step_counter = 0

        self.asset_min_state_ratio = None
        self.asset_max_state_ratio = None

        self.global_tensor_dict["crashes"] = torch.zeros(
            (self.num_envs), device=self.device, requires_grad=False, dtype=torch.bool
        )
        self.global_tensor_dict["truncations"] = torch.zeros(
            (self.num_envs), device=self.device, requires_grad=False, dtype=torch.bool
        )

        self.num_env_actions = self.cfg.env.num_env_actions
        self.global_tensor_dict["num_env_actions"] = self.num_env_actions
        self.global_tensor_dict["env_actions"] = None
        self.global_tensor_dict["prev_env_actions"] = None

        self.collision_tensor = self.global_tensor_dict["crashes"]
        self.truncation_tensor = self.global_tensor_dict["truncations"]

        # Before we populate the environment, we need to create the ground plane
        if self.cfg.env.create_ground_plane:
            logger.info("Creating ground plane in Isaac Gym Simulation.")
            self.IGE_env.create_ground_plane()
            logger.info("[DONE] Creating ground plane in Isaac Gym Simulation")

        for i in range(self.cfg.env.num_envs):
            logger.debug(f"Populating environment {i}")
            if i % 1000 == 0:
                logger.info(f"Populating environment {i}")

            env_handle = self.IGE_env.create_env(i)
            if self.cfg.env.use_warp:
                self.warp_env.create_env(i)

            # add robot asset in the environment
            self.robot_manager.add_robot_to_env(
                self.IGE_env, env_handle, self.global_asset_counter, i, segmentation_ctr
            )
            self.global_asset_counter += 1

            self.num_obs_in_env = 0
            # add regular assets in the environment
            for asset_info_dict in self.global_asset_dicts[i]:
                asset_handle, ige_seg_ctr = self.IGE_env.add_asset_to_env(
                    asset_info_dict,
                    env_handle,
                    i,
                    self.global_asset_counter,
                    segmentation_ctr,
                )
                self.num_obs_in_env += 1
                warp_segmentation_ctr = 0
                if self.cfg.env.use_warp:
                    empty_handle, warp_segmentation_ctr = self.warp_env.add_asset_to_env(
                        asset_info_dict,
                        i,
                        self.global_asset_counter,
                        segmentation_ctr,
                    )
                # Update this after added in WARP
                self.global_asset_counter += 1
                segmentation_ctr += max(ige_seg_ctr, warp_segmentation_ctr)
                if self.asset_min_state_ratio is None or self.asset_max_state_ratio is None:
                    self.asset_min_state_ratio = torch.tensor(
                        asset_info_dict["min_state_ratio"], requires_grad=False
                    ).unsqueeze(0)
                    self.asset_max_state_ratio = torch.tensor(
                        asset_info_dict["max_state_ratio"], requires_grad=False
                    ).unsqueeze(0)
                else:
                    self.asset_min_state_ratio = torch.vstack(
                        (
                            self.asset_min_state_ratio,
                            torch.tensor(asset_info_dict["min_state_ratio"], requires_grad=False),
                        )
                    )
                    self.asset_max_state_ratio = torch.vstack(
                        (
                            self.asset_max_state_ratio,
                            torch.tensor(asset_info_dict["max_state_ratio"], requires_grad=False),
                        )
                    )

        # check if environment has 0 objects. If so, skip this step
        if self.asset_min_state_ratio is not None:
            self.asset_min_state_ratio = self.asset_min_state_ratio.to(self.device)
            self.asset_max_state_ratio = self.asset_max_state_ratio.to(self.device)
            self.global_tensor_dict["asset_min_state_ratio"] = self.asset_min_state_ratio.view(
                self.cfg.env.num_envs, -1, 13
            )
            self.global_tensor_dict["asset_max_state_ratio"] = self.asset_max_state_ratio.view(
                self.cfg.env.num_envs, -1, 13
            )
        else:
            self.global_tensor_dict["asset_min_state_ratio"] = torch.zeros(
                (self.cfg.env.num_envs, 0, 13), device=self.device
            )
            self.global_tensor_dict["asset_max_state_ratio"] = torch.zeros(
                (self.cfg.env.num_envs, 0, 13), device=self.device
            )

        self.global_tensor_dict["num_obstacles_in_env"] = self.num_obs_in_env

    def prepare_sim(self):
        """
        This function prepares the simulation for the environment.
        """
        if not self.IGE_env.prepare_for_simulation(self, self.global_tensor_dict):
            raise Exception("Failed to prepare the simulation")
        if self.cfg.env.use_warp:
            if not self.warp_env.prepare_for_simulation(self.global_tensor_dict):
                raise Exception("Failed to prepare the simulation")

        self.asset_manager = AssetManager(self.global_tensor_dict, self.keep_in_env)
        self.asset_manager.prepare_for_sim()
        self.robot_manager.prepare_for_sim(self.global_tensor_dict)
        self.obstacle_manager = ObstacleManager(
            self.IGE_env.num_assets_per_env, self.cfg, self.device
        )
        self.obstacle_manager.prepare_for_sim(self.global_tensor_dict)
        self.num_robot_actions = self.global_tensor_dict["num_robot_actions"]

    def reset_idx(self, env_ids=None):
        """
        This function resets the environment for the given environment indices.
        """
        # first reset the Isaac Gym environment since that determines the environment bounds
        # then reset the asset managet that respositions assets within the environment
        # then reset the warp environment if it is being used that reads the state tensors from the assets and transforms meshes
        # finally reset the robot manager that resets the robot state tensors and the sensors
        # logger.debug(f"Resetting environments {env_ids}.")
        self.IGE_env.reset_idx(env_ids)
        self.asset_manager.reset_idx(env_ids, self.global_tensor_dict["num_obstacles_in_env"])
        if self.cfg.env.use_warp:
            self.warp_env.reset_idx(env_ids)
        self.robot_manager.reset_idx(env_ids)
        self.IGE_env.write_to_sim()
        self.sim_steps[env_ids] = 0

    def log_memory_use(self):
        """
        This function logs the memory usage of the GPU.
        """
        logger.warning(
            f"torch.cuda.memory_allocated: {torch.cuda.memory_allocated(0)/1024/1024/1024}GB"
        )
        logger.warning(
            f"torch.cuda.memory_reserved: {torch.cuda.memory_reserved(0)/1024/1024/1024}GB"
        )
        logger.warning(
            f"torch.cuda.max_memory_reserved: {torch.cuda.max_memory_reserved(0)/1024/1024/1024}GB"
        )

        # Calculate and system RAM usage used by the objects of this class
        total_memory = 0
        for key, value in self.__dict__.items():
            total_memory += value.__sizeof__()
        logger.warning(
            f"Total memory used by the objects of this class: {total_memory/1024/1024}MB"
        )

    def reset(self):
        self.reset_idx(env_ids=torch.arange(self.cfg.env.num_envs))

    def pre_physics_step(self, actions, env_actions):
        # first let the robot compute the actions
        self.robot_manager.pre_physics_step(actions)
        # then the asset manager applies the actions here
        self.asset_manager.pre_physics_step(env_actions)
        # apply actions to obstacle manager
        self.obstacle_manager.pre_physics_step(env_actions)
        # then the simulator applies them here
        self.IGE_env.pre_physics_step(actions)
        # if warp is used, the warp environment applies the actions here
        # If you change the mesh, refit() needs to be called (expensive).
        if self.use_warp:
            self.warp_env.pre_physics_step(actions)

    def reset_tensors(self):
        self.collision_tensor[:] = 0
        self.truncation_tensor[:] = 0

    def simulate(self, actions, env_actions):
        self.pre_physics_step(actions, env_actions)
        self.IGE_env.physics_step()
        self.post_physics_step(actions, env_actions)

    def post_physics_step(self, actions, env_actions):
        self.IGE_env.post_physics_step()
        self.robot_manager.post_physics_step()
        if self.use_warp:
            self.warp_env.post_physics_step()
        self.asset_manager.post_physics_step()

    def compute_observations(self):
        self.collision_tensor[:] += (
            torch.norm(self.global_tensor_dict["robot_contact_force_tensor"], dim=1)
            > self.cfg.env.collision_force_threshold
        )

    def reset_terminated_and_truncated_envs(self):
        collision_envs = self.collision_tensor.nonzero(as_tuple=False).squeeze(-1)
        truncation_envs = self.truncation_tensor.nonzero(as_tuple=False).squeeze(-1)
        envs_to_reset = (
            (self.collision_tensor * int(self.cfg.env.reset_on_collision) + self.truncation_tensor)
            .nonzero(as_tuple=False)
            .squeeze(-1)
        )
        # reset the environments that have a collision
        if len(envs_to_reset) > 0:
            self.reset_idx(envs_to_reset)
        return envs_to_reset

    def render(self, render_components="sensors"):
        if render_components == "viewer":
            self.render_viewer()
        elif render_components == "sensors":
            self.render_sensors()

    def render_sensors(self):
        # render sensors after the physics step
        if self.robot_manager.has_IGE_sensors:
            self.IGE_env.step_graphics()
        self.robot_manager.capture_sensors()

    def render_viewer(self):
        # render viewer GUI
        self.IGE_env.render_viewer()

    def post_reward_calculation_step(self):
        envs_to_reset = self.reset_terminated_and_truncated_envs()
        # render is performed after reset to ensure that the sensors are updated from the new robot state.
        self.render(render_components="sensors")
        return envs_to_reset

    def step(self, actions, env_actions=None):
        """
        This function steps the simulation for the environment.
        actions: The actions that are sent to the robot.
        env_actions: The actions that are sent to the environment entities.
        """
        self.reset_tensors()
        if env_actions is not None:
            if self.global_tensor_dict["env_actions"] is None:
                self.global_tensor_dict["env_actions"] = env_actions
                self.global_tensor_dict["prev_env_actions"] = env_actions
                self.prev_env_actions = self.global_tensor_dict["prev_env_actions"]
                self.env_actions = self.global_tensor_dict["env_actions"]
            logger.warning(
                f"Env actions shape: {env_actions.shape}, Previous env actions shape: {self.env_actions.shape}"
            )
            self.prev_env_actions[:] = self.env_actions
            self.env_actions[:] = env_actions
        num_physics_step_per_env_step = max(
            math.floor(
                random.gauss(
                    self.cfg.env.num_physics_steps_per_env_step_mean,
                    self.cfg.env.num_physics_steps_per_env_step_std,
                )
            ),
            0,
        )
        for timestep in range(num_physics_step_per_env_step):
            self.simulate(actions, env_actions)
            self.compute_observations()
        self.sim_steps[:] = self.sim_steps[:] + 1
        self.step_counter += 1
        if self.step_counter % self.cfg.env.render_viewer_every_n_steps == 0:
            self.render(render_components="viewer")

    def get_obs(self):
        # Just return the dict of all tensors. Whatever the task needs can be used to compute the rewards.
        return self.global_tensor_dict
