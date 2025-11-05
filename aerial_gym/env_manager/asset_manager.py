import random

import numpy as np
import torch

from aerial_gym.config.asset_config.env_object_config import TARGET_SEMANTIC_ID
from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.math import *

logger = CustomLogger("asset_manager")
logger.setLevel("DEBUG")


class AssetManager:
    def __init__(
        self,
        global_tensor_dict,
        num_keep_in_env,
        terrain_generators=None,
        cfg=None,
        global_asset_dicts=None,
        sim_config=None,
    ):
        self.init_tensors(global_tensor_dict, num_keep_in_env)
        self.terrain_generators = terrain_generators if terrain_generators else {}
        self.cfg = cfg
        self.global_asset_dicts = global_asset_dicts
        self.sim_config = sim_config

        # Find target asset index (asset with TARGET_SEMANTIC_ID)
        self.target_asset_idx = None
        if self.global_asset_dicts is not None and len(self.global_asset_dicts) > 0:
            for asset_idx, asset_dict in enumerate(self.global_asset_dicts[0]):
                if asset_dict.get("semantic_id") == TARGET_SEMANTIC_ID:
                    self.target_asset_idx = asset_idx
                    logger.info(f"Found target asset at index {asset_idx}")
                    break

        # Initialize target movement state
        self.target_movement_steps = None
        if self.target_asset_idx is not None:
            num_envs = self.env_asset_state_tensor.shape[0]
            self.target_movement_steps = torch.zeros(
                num_envs, dtype=torch.int32, device=self.env_asset_state_tensor.device
            )

    def init_tensors(self, global_tensor_dict, num_keep_in_env):
        self.env_asset_state_tensor = global_tensor_dict["env_asset_state_tensor"]
        self.asset_min_state_ratio = global_tensor_dict["asset_min_state_ratio"]
        self.asset_max_state_ratio = global_tensor_dict["asset_max_state_ratio"]
        self.env_bounds_min = (
            global_tensor_dict["env_bounds_min"].unsqueeze(1).expand(-1, self.env_asset_state_tensor.shape[1], -1)
        )
        self.env_bounds_max = (
            global_tensor_dict["env_bounds_max"].unsqueeze(1).expand(-1, self.env_asset_state_tensor.shape[1], -1)
        )
        self.num_keep_in_env = num_keep_in_env

    def prepare_for_sim(self):
        self.reset(self.num_keep_in_env)
        logger.warning(f"Number of obstacles to be kept in the environment: {self.num_keep_in_env}")

    def pre_physics_step(self, actions):
        pass

    def post_physics_step(self):
        """
        Update target Z position to match terrain height and keep orientation upright.
        """
        if self.target_asset_idx is None:
            return

        # Get target positions and orientations
        target_positions = self.env_asset_state_tensor[:, self.target_asset_idx, 0:3].clone()
        target_orientations = self.env_asset_state_tensor[:, self.target_asset_idx, 3:7].clone()

        # Update Z based on terrain height at X, Y position
        if self.terrain_generators:
            env_ids_list = list(range(target_positions.shape[0]))
            for env_id in env_ids_list:
                if env_id in self.terrain_generators:
                    terrain_gen = self.terrain_generators[env_id]
                    heightmap = terrain_gen.generate_heightmap(use_cache=True)

                    x = target_positions[env_id, 0].item()
                    y = target_positions[env_id, 1].item()
                    terrain_height = terrain_gen.sample_height(x, y, heightmap)
                    terrain_offset = terrain_gen.amplitude / 2.0

                    # Set Z to terrain surface (no offset)
                    target_positions[env_id, 2] = terrain_height + terrain_offset

        # Keep orientation upright (no roll, no pitch) - identity quaternion [0, 0, 0, 1]
        target_orientations[:, 0] = 0.0  # x
        target_orientations[:, 1] = 0.0  # y
        target_orientations[:, 2] = 0.0  # z
        target_orientations[:, 3] = 1.0  # w

        # Update tensors
        self.env_asset_state_tensor[:, self.target_asset_idx, 0:3] = target_positions
        self.env_asset_state_tensor[:, self.target_asset_idx, 3:7] = target_orientations

    def step(self, actions):
        """
        Update target movement - random walk with configurable parameters.
        """
        if self.target_asset_idx is None or self.cfg is None:
            return

        if not hasattr(self.cfg.env, "target_velocity_change_interval"):
            return

        num_envs = self.env_asset_state_tensor.shape[0]
        device = self.env_asset_state_tensor.device

        # Increment step counters
        self.target_movement_steps += 1

        # Get target positions and velocities
        target_positions = self.env_asset_state_tensor[:, self.target_asset_idx, 0:3].clone()
        target_velocities = self.env_asset_state_tensor[:, self.target_asset_idx, 7:10].clone()

        # Check which environments need velocity updates
        interval = self.cfg.env.target_velocity_change_interval
        stop_prob = self.cfg.env.target_stop_probability
        vel_max = self.cfg.env.target_velocity_max
        vel_min = self.cfg.env.target_velocity_min

        # Find environments where interval is reached
        needs_update = self.target_movement_steps >= interval

        if torch.any(needs_update):
            # Sample new velocities for environments that need updates
            env_ids_to_update = torch.where(needs_update)[0]

            for env_id in env_ids_to_update:
                # Sample whether to stop
                if random.random() < stop_prob:
                    # Stop moving
                    target_velocities[env_id, :] = 0.0
                else:
                    # Sample new random velocity direction (XY plane only)
                    angle = random.uniform(0, 2 * np.pi)
                    speed = random.uniform(vel_min, vel_max)
                    target_velocities[env_id, 0] = speed * np.cos(angle)
                    target_velocities[env_id, 1] = speed * np.sin(angle)
                    target_velocities[env_id, 2] = 0.0  # No vertical movement

                # Reset step counter
                self.target_movement_steps[env_id] = 0

        # Update positions based on velocity and dt
        # Get dt from sim config (default 0.01 if not available)
        dt = getattr(self.sim_config.sim, "dt", 0.01) if self.sim_config is not None else 0.01

        # Update X, Y positions (Z will be set by terrain in post_physics_step)
        target_positions[:, 0:2] += target_velocities[:, 0:2] * dt

        # Clamp positions to environment bounds
        env_bounds_min = self.env_bounds_min[:, self.target_asset_idx, :]
        env_bounds_max = self.env_bounds_max[:, self.target_asset_idx, :]
        target_positions[:, 0:2] = torch.clamp(target_positions[:, 0:2], env_bounds_min[:, 0:2], env_bounds_max[:, 0:2])

        # Keep orientation upright (no roll, no pitch) - identity quaternion [0, 0, 0, 1]
        target_orientations = self.env_asset_state_tensor[:, self.target_asset_idx, 3:7].clone()
        target_orientations[:, 0] = 0.0  # x
        target_orientations[:, 1] = 0.0  # y
        target_orientations[:, 2] = 0.0  # z
        target_orientations[:, 3] = 1.0  # w

        # Update tensors
        self.env_asset_state_tensor[:, self.target_asset_idx, 0:3] = target_positions
        self.env_asset_state_tensor[:, self.target_asset_idx, 3:7] = target_orientations
        self.env_asset_state_tensor[:, self.target_asset_idx, 7:10] = target_velocities

    def reset(self, num_obstacles_per_env):
        self.reset_idx(torch.arange(self.env_asset_state_tensor.shape[0]), num_obstacles_per_env)

    def reset_idx(self, env_ids, num_obstacles_per_env=0):
        if num_obstacles_per_env < self.num_keep_in_env:
            logger.info(
                "Number of obstacles required in the environment by the \
                  code is lesser than the minimum number of obstacles that the environment configuration specifies."
            )
            num_obstacles_per_env = self.num_keep_in_env

        sampled_asset_state_ratio = torch_rand_float_tensor(self.asset_min_state_ratio, self.asset_max_state_ratio)
        positions = torch_interpolate_ratio(
            min=self.env_bounds_min,
            max=self.env_bounds_max,
            ratio=sampled_asset_state_ratio[..., 0:3],
        )[env_ids, :, :]

        if self.terrain_generators:
            env_ids_list = env_ids.cpu().numpy() if isinstance(env_ids, torch.Tensor) else env_ids
            for idx, env_id in enumerate(env_ids_list):
                if env_id in self.terrain_generators:
                    terrain_gen = self.terrain_generators[env_id]
                    heightmap = terrain_gen.generate_heightmap(use_cache=True)

                    for asset_idx in range(positions.shape[1]):
                        x = positions[idx, asset_idx, 0].item()
                        y = positions[idx, asset_idx, 1].item()
                        terrain_height = terrain_gen.sample_height(x, y, heightmap)
                        terrain_offset = terrain_gen.amplitude / 2.0
                        positions[idx, asset_idx, 2] = terrain_height + terrain_offset

        self.env_asset_state_tensor[env_ids, :, 0:3] = positions

        # Reset target movement step counters and velocities for reset environments
        if self.target_asset_idx is not None and self.target_movement_steps is not None:
            env_ids_tensor = (
                env_ids
                if isinstance(env_ids, torch.Tensor)
                else torch.tensor(env_ids, device=self.env_asset_state_tensor.device)
            )
            self.target_movement_steps[env_ids_tensor] = 0
            # Reset target velocity to zero or random initial velocity
            if isinstance(env_ids, torch.Tensor):
                target_velocities = self.env_asset_state_tensor[env_ids, self.target_asset_idx, 7:10]
                # Optionally set random initial velocity
                # For now, start with zero velocity
                target_velocities.zero_()
            else:
                self.env_asset_state_tensor[env_ids, self.target_asset_idx, 7:10] = 0.0
        self.env_asset_state_tensor[env_ids, :, 3:7] = quat_from_euler_xyz_tensor(
            sampled_asset_state_ratio[env_ids, :, 3:6]
        )
        # put those obstacles not needed in the environment outside
        self.env_asset_state_tensor[env_ids, num_obstacles_per_env:, 0:3] = -1000.0
