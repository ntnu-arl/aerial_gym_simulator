# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import pytorch3d.transforms as p3d_transforms
from aerial_gym.utils.math import * 


class LeePositionController:
    def __init__(self, K_pos_tensor, K_vel_tensor, K_rot_tensor, K_angvel_tensor):
        self.K_pos_tensor = K_pos_tensor
        self.K_vel_tensor = K_vel_tensor
        self.K_rot_tensor = K_rot_tensor
        self.K_angvel_tensor = K_angvel_tensor

    def __call__(self, robot_state, command_actions):
        """
        Lee position controller
        :param robot_state: tensor of shape (num_envs, 13) with state of the robot
        :param command_actions: tensor of shape (num_envs, 4) with desired position setpoint and yaw command (from -pi to pi scaled between -1 to 1)
        :return: m*g normalized thrust and interial normalized torques
        """
        # perform calculation for transformation matrices

        rotation_matrices = p3d_transforms.quaternion_to_matrix(
            robot_state[:, [6, 3, 4, 5]])
        rotation_matrix_transpose = torch.transpose(rotation_matrices, 1, 2)
        euler_angles = p3d_transforms.matrix_to_euler_angles(
            rotation_matrices, "ZYX")[:, [2, 1, 0]]

        vehicle_position = robot_state[:, 0:3]

        desired_vehicle_position = command_actions[:, :3]

        # Compute desired accelerations
        pos_error = desired_vehicle_position - vehicle_position
        accel_command = self.K_pos_tensor * pos_error - self.K_vel_tensor*robot_state[:, 7:10]
        accel_command[:, 2] += 1

        forces_command = accel_command
        thrust_command = torch.sum(forces_command * rotation_matrices[:, :, 2], dim=1)

        # # print(accel_command.shape, torch.norm(accel_command, dim=1).shape)
        b3_c = torch.div(accel_command, torch.norm(accel_command, dim=1).unsqueeze(1))


        temp_dir = torch.zeros_like(euler_angles)
        temp_dir[:, 0] = torch.cos(euler_angles[:, 2])
        temp_dir[:, 1] = torch.sin(euler_angles[:, 2])

        # temp_dir = torch.rand_like(euler_angles)

        b2_c = torch.cross(b3_c, temp_dir, dim=1)
        b2_c = torch.div(b2_c, torch.norm(b2_c, dim=1).unsqueeze(1))
        
        
        b1_c = torch.cross(b2_c, b3_c)
        rotation_matrix_desired = torch.zeros_like(rotation_matrices)
        rotation_matrix_desired[:, :, 0] = b1_c
        rotation_matrix_desired[:, :, 1] = b2_c
        rotation_matrix_desired[:, :, 2] = b3_c

        ## Stuff that works below here
        rotation_matrix_desired_transpose = torch.transpose(
            rotation_matrix_desired, 1, 2)
        rot_err_mat = torch.bmm(rotation_matrix_desired_transpose, rotation_matrices) - \
            torch.bmm(rotation_matrix_transpose, rotation_matrix_desired)
        rot_err = 0.5 * compute_vee_map(rot_err_mat)

        rotmat_euler_to_body_rates = torch.zeros_like(rotation_matrices)

        s_pitch = torch.sin(euler_angles[:, 1])
        c_pitch = torch.cos(euler_angles[:, 1])

        s_roll = torch.sin(euler_angles[:, 0])
        c_roll = torch.cos(euler_angles[:, 0])

        rotmat_euler_to_body_rates[:, 0, 0] = 1.0
        rotmat_euler_to_body_rates[:, 1, 1] = c_roll
        rotmat_euler_to_body_rates[:, 0, 2] = -s_pitch
        rotmat_euler_to_body_rates[:, 2, 1] = -s_roll
        rotmat_euler_to_body_rates[:, 1, 2] = s_roll * c_pitch
        rotmat_euler_to_body_rates[:, 2, 2] = c_roll * c_pitch

        euler_angle_rates = torch.zeros_like(euler_angles)
        yaw_setpoint = euler_angles[:, 2]
        euler_angle_rates[:, 2] = torch.remainder((command_actions[:, 3] - yaw_setpoint), 3.14159265358979323846 * 2.0)
        
        euler_angle_rates[:, 2] = torch.where(euler_angle_rates[:, 2] > 3.14159265358979323846, euler_angle_rates[:, 2] - 3.14159265358979323846 * 2.0, euler_angle_rates[:, 2])

        omega_desired_body = torch.bmm(rotmat_euler_to_body_rates, euler_angle_rates.unsqueeze(2)).squeeze(2)

        # omega_des_body = [0, 0, yaw_rate] ## approximated body_rate as yaw_rate
        # omega_body = R_t @ omega_world
        # angvel_err = omega_body - R_t @ R_des @ omega_des_body
        # Refer to Lee et. al. (2010) for details (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5717652)
        
        desired_angvel_err = torch.bmm(rotation_matrix_transpose, torch.bmm(
            rotation_matrix_desired, omega_desired_body.unsqueeze(2))).squeeze(2)

        actual_angvel_err = torch.bmm(
            rotation_matrix_transpose, robot_state[:, 10:13].unsqueeze(2)).squeeze(2)
        
        angvel_err = actual_angvel_err - desired_angvel_err
        torque = - self.K_rot_tensor * rot_err - self.K_angvel_tensor * angvel_err + torch.cross(robot_state[:, 10:13],robot_state[:, 10:13], dim=1)
        return thrust_command, torque
