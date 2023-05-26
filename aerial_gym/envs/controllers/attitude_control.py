# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import pytorch3d.transforms as p3d_transforms
from aerial_gym.utils.math import * 

class LeeAttitudeContoller:
    def __init__(self, K_rot_tensor, K_angvel_tensor):
        self.K_rot_tensor = K_rot_tensor
        self.K_angvel_tensor = K_angvel_tensor


    def __call__(self, robot_state, command_actions):
        """
            Lee attitude controller
            :param robot_state: tensor of shape (num_envs, 13) with state of the robot
            :param command_actions: tensor of shape (num_envs, 4) with desired thrust, roll, pitch and yaw_rate command in vehicle frame
            :return: m*g normalized thrust and interial normalized torques
            """
        # quaternion with real component first
        rotation_matrices = p3d_transforms.quaternion_to_matrix(
            robot_state[:, [6, 3, 4, 5]])
        # Convert current rotation matrix to euler angles
        euler_angles = p3d_transforms.matrix_to_euler_angles(
            rotation_matrices, "ZYX")[:, [2, 1, 0]]
        rotation_matrix_transpose = torch.transpose(rotation_matrices, 1, 2)

        # desired euler angle is equal to commanded roll, commanded pitch, and current yaw
        desired_euler_angles_zyx = torch.zeros_like(euler_angles)
        desired_euler_angles_zyx[:, 0] = euler_angles[:, 2]
        desired_euler_angles_zyx[:, [1,2]] = command_actions[:, [2, 1]]

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
        euler_angle_rates[:, 2] = command_actions[:, 3]

        omega_desired_body = torch.bmm(rotmat_euler_to_body_rates, euler_angle_rates.unsqueeze(2)).squeeze(2)
        
        # Convert target euler angles to rotation matrix
        rotation_matrix_desired = p3d_transforms.euler_angles_to_matrix(
            desired_euler_angles_zyx, "ZYX")
        rotation_matrix_desired_transpose = torch.transpose(
            rotation_matrix_desired, 1, 2)
        
        # Rotation error matrix based on: https://mathweb.ucsd.edu/~mleok/pdf/LeLeMc2010_quadrotor.pdf
        rot_err_mat = torch.bmm(rotation_matrix_desired_transpose, rotation_matrices) - \
            torch.bmm(rotation_matrix_transpose, rotation_matrix_desired)
        rot_err = 0.5 * compute_vee_map(rot_err_mat)
        
        desired_angvel_err = torch.bmm(rotation_matrix_transpose, torch.bmm(
            rotation_matrix_desired, omega_desired_body.unsqueeze(2))).squeeze(2)

        actual_angvel_err = torch.bmm(
            rotation_matrix_transpose, robot_state[:, 10:13].unsqueeze(2)).squeeze(2)
        
        angvel_err = actual_angvel_err - desired_angvel_err

        torques = -self.K_rot_tensor * rot_err - self.K_angvel_tensor * angvel_err + torch.cross(robot_state[:, 10:13],robot_state[:, 10:13], dim=1)
        
        return (command_actions[:, 0] + 1), torques

