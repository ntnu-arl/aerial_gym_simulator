from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.math import *

logger = CustomLogger("asset_manager")
logger.setLevel("DEBUG")


class AssetManager:
    def __init__(self, global_tensor_dict, num_keep_in_env, terrain_generators=None):
        self.init_tensors(global_tensor_dict, num_keep_in_env)
        self.terrain_generators = terrain_generators if terrain_generators else {}

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
        pass

    def step(self, actions):
        pass
        # Implement this function if needed.
        # this functionality can do speciic things with the environment assets on stepping.
        # nothing really needs to be done for static environments.
        # if force needs to be applied, it should be done in the other classes and it's
        # better to leave this class to manipulate the state tensors.

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
        self.env_asset_state_tensor[env_ids, :, 3:7] = quat_from_euler_xyz_tensor(
            sampled_asset_state_ratio[env_ids, :, 3:6]
        )
        # put those obstacles not needed in the environment outside
        self.env_asset_state_tensor[env_ids, num_obstacles_per_env:, 0:3] = -1000.0
