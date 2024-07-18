import torch
import pytorch3d.transforms as p3d_transforms
from aerial_gym.utils.math import *
from aerial_gym.control.controllers.base_lee_controller import *


class LeeVelocitySteeringAngleController(BaseLeeController):
    def __init__(self, config, num_envs, device):
        super().__init__(config, num_envs, device)

    def init_tensors(self, global_tensor_dict=None):
        super().init_tensors(global_tensor_dict)
        self.euler_angle_rates = torch.zeros_like(self.robot_body_angvel)

    def update(self, command_actions):
        """
        Lee attitude controller
        :param robot_state: tensor of shape (num_envs, 13) with state of the robot
        :param command_actions: tensor of shape (num_envs, 4) with desired thrust, roll, pitch and yaw_rate command in vehicle frame
        :return: m*g normalized thrust and interial normalized torques
        """
        self.reset_commands()
        self.accel[:] = self.compute_acceleration(
            setpoint_position=self.robot_position,
            setpoint_velocity=command_actions[:, 0:3],
        )
        forces = (self.accel[:] - self.gravity) * self.mass
        # thrust command is transformed by the body orientation's z component
        self.wrench_command[:, 2] = torch.sum(
            forces * quat_to_rotation_matrix(self.robot_orientation)[:, :, 2], dim=1
        )

        # after calculating forces, we calculate the desired euler angles
        self.desired_quat[:] = calculate_desired_orientation_for_position_velocity_control(
            forces, command_actions[:, 3], self.buffer_tensor
        )
        self.euler_angle_rates[:] = 0.0
        self.desired_body_angvel[:] = euler_rates_to_body_rates(
            self.robot_euler_angles, self.euler_angle_rates, self.buffer_tensor
        )
        self.wrench_command[:, 3:6] = self.compute_body_torque(
            self.desired_quat, self.desired_body_angvel
        )

        return self.wrench_command
