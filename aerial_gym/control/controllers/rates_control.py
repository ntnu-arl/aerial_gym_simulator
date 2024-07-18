import torch
import pytorch3d.transforms as p3d_transforms
from aerial_gym.utils.math import *


from aerial_gym.control.controllers.base_lee_controller import *


class LeeRatesController(BaseLeeController):
    def __init__(self, config, num_envs, device):
        super().__init__(config, num_envs, device)

    def init_tensors(self, global_tensor_dict=None):
        super().init_tensors(global_tensor_dict)

    def update(self, command_actions):
        """
        Lee attitude controller
        :param robot_state: tensor of shape (num_envs, 13) with state of the robot
        :param command_actions: tensor of shape (num_envs, 4) with desired thrust, roll, pitch and yaw_rate command in vehicle frame
        :return: m*g normalized thrust and interial normalized torques
        """
        self.reset_commands()
        # quaternion desired
        self.wrench_command[:, 2] = (command_actions[:, 0] - self.gravity) * self.mass
        self.wrench_command[:, 3:6] = self.compute_body_torque(
            self.robot_orientation, command_actions[:, 1:4]
        )

        return self.wrench_command
