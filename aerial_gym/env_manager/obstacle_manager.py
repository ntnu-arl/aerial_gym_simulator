from aerial_gym.env_manager.base_env_manager import BaseManager

# from aerial_gym.registry.controller_registry import controller_registry
from aerial_gym.utils.math import *


from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("obstacle_manager")


class ObstacleManager(BaseManager):
    def __init__(self, num_assets, config, device):
        super().__init__(config, device)
        self.global_tensor_dict = {}
        self.num_assets = num_assets

        logger.debug("Obstacle Manager initialized")

    def prepare_for_sim(self, global_tensor_dict):
        if self.num_assets <= 1:
            return
        self.global_tensor_dict = global_tensor_dict
        self.obstacle_position = global_tensor_dict["obstacle_position"]
        self.obstacle_orientation = global_tensor_dict["obstacle_orientation"]
        self.obstacle_linvel = global_tensor_dict["obstacle_linvel"]
        self.obstacle_angvel = global_tensor_dict["obstacle_angvel"]

        # self.obstacle_force_tensors = global_tensor_dict["obstacle_force_tensor"]
        # self.obstacle_torque_tensors = global_tensor_dict["obstacle_torque_tensor"]

    def reset(self):
        # self.controller.reset()
        return

    def reset_idx(self, env_ids):
        # self.controller.reset_idx(env_ids)
        return

    def pre_physics_step(self, actions=None):
        if self.num_assets <= 1 or actions is None:
            return
        self.obstacle_linvel[:] = actions[:, :, 0:3]
        self.obstacle_angvel[:] = actions[:, :, 3:6]
        # self.update_states()
        # self.obstacle_wrench[:] = self.controller(actions)
        # self.obstacle_force_tensors[:] = self.obstacle_wrench[:, :, 0:3]
        # self.obstacle_torque_tensors[:] = self.obstacle_wrench[:, :, 3:6]

    def step(self):
        pass

    # def update_states(self):
    #     self.obstacle_euler_angles[:] = ssa(get_euler_xyz_tensor(self.obstacle_orientation))
    #     self.obstacle_vehicle_orientation[:] = vehicle_frame_quat_from_quat(self.obstacle_orientation)
    #     self.obstacle_vehicle_linvel[:] = quat_rotate_inverse(
    #         self.obstacle_vehicle_orientation, self.obstacle_linvel
    #     )
    #     self.obstacle_body_linvel[:] = quat_rotate_inverse(self.obstacle_orientation, self.obstacle_linvel)
    #     self.obstacle_body_angvel[:] = quat_rotate_inverse(self.obstacle_orientation, self.obstacle_angvel)
