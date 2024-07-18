import torch
import pytorch3d.transforms as p3d_transforms
from aerial_gym.utils.math import *
from aerial_gym.control.controllers.base_lee_controller import *


class FullyActuatedController(BaseLeeController):
    def __init__(self, config, num_envs, device):
        super().__init__(config, num_envs, device)

    def init_tensors(self, global_tensor_dict=None):
        super().init_tensors(global_tensor_dict)

    def update(self, command_actions):
        """
        Fully actuated controller. Input is in the form of desired position and orientation.
        command_actions = [p_x, p_y, p_z, qx, qy, qz, qw]
        Position setpoint is in the world frame
        Orientation reference is w.r.t world frame
        """
        self.reset_commands()
        command_actions[:, 3:7] = normalize(command_actions[:, 3:7])
        self.accel[:] = self.compute_acceleration(
            command_actions[:, 0:3], torch.zeros_like(command_actions[:, 0:3])
        )
        forces = self.mass * (self.accel - self.gravity)
        self.wrench_command[:, 0:3] = quat_rotate_inverse(self.robot_orientation, forces)
        self.desired_quat[:] = command_actions[:, 3:]
        self.wrench_command[:, 3:6] = self.compute_body_torque(
            self.desired_quat, torch.zeros_like(command_actions[:, 0:3])
        )
        return self.wrench_command
