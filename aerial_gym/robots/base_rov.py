from aerial_gym.robots.base_robot import BaseRobot

from aerial_gym.control.control_allocation import ControlAllocator
from aerial_gym.registry.controller_registry import controller_registry

import torch
import numpy as np

from aerial_gym.utils.math import *

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("base_rov")


class BaseROV(BaseRobot):
    """
    Base class for a fully actuated ROV robot.
    """

    def __init__(self, robot_config, controller_name, env_config, device):
        logger.debug("Initializing BaseROV")
        super().__init__(
            robot_config=robot_config,
            controller_name=controller_name,
            env_config=env_config,
            device=device,
        )
        logger.warning(f"Creating {self.num_envs} ROVs.")
        self.force_application_level = self.cfg.control_allocator_config.force_application_level
        if controller_name == "no_control":
            self.output_mode = "forces"
        else:
            self.output_mode = "wrench"

        if self.force_application_level == "root_link" and controller_name == "no_control":
            raise ValueError(
                "Force application level 'root_link' cannot be used with 'no_control'."
            )

        # Initialize the tensors
        self.robot_state = None
        self.robot_force_tensors = None
        self.robot_torque_tensors = None
        self.action_tensor = None
        self.max_init_state = None
        self.min_init_state = None
        self.max_force_and_torque_disturbance = None
        self.max_torque_disturbance = None
        self.controller_input = None
        self.control_allocator = None
        self.output_forces = None
        self.output_torques = None

        logger.debug("[DONE] Initializing BaseROV")

    def init_tensors(self, global_tensor_dict):
        """
        Initialize the tensors for the robot state, force, torque, and action.
        The tensors used in this function call are sent as slices from the main tensors in the environment.
        These slices are only detemine the robot state, force, torque, and action.
        The full tensors are not passed to this function to avoid access to data that is not needed by the robot.
        """
        super().init_tensors(global_tensor_dict)
        # Adding more tensors to the global tensor dictionary
        self.robot_vehicle_orientation = torch.zeros_like(
            self.robot_orientation, requires_grad=False, device=self.device
        )
        self.robot_vehicle_linvel = torch.zeros_like(
            self.robot_linvel, requires_grad=False, device=self.device
        )
        self.robot_body_angvel = torch.zeros_like(
            self.robot_linvel, requires_grad=False, device=self.device
        )
        self.robot_body_linvel = torch.zeros_like(
            self.robot_linvel, requires_grad=False, device=self.device
        )
        self.robot_euler_angles = torch.zeros_like(
            self.robot_linvel, requires_grad=False, device=self.device
        )
        # Add to tensor dictionary
        global_tensor_dict["robot_vehicle_orientation"] = self.robot_vehicle_orientation
        global_tensor_dict["robot_vehicle_linvel"] = self.robot_vehicle_linvel
        global_tensor_dict["robot_body_angvel"] = self.robot_body_angvel
        global_tensor_dict["robot_body_linvel"] = self.robot_body_linvel
        global_tensor_dict["robot_euler_angles"] = self.robot_euler_angles

        global_tensor_dict["num_robot_actions"] = self.controller_config.num_actions

        self.controller.init_tensors(global_tensor_dict)
        self.action_tensor = torch.zeros(
            (self.num_envs, self.controller_config.num_actions), device=self.device
        )

        # Initialize the robot state
        # [x, y, z, roll, pitch, yaw, 1.0 (for maintaining shape), vx, vy, vz, wx, wy, wz]
        self.min_init_state = torch.tensor(
            self.cfg.init_config.min_init_state, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)
        self.max_init_state = torch.tensor(
            self.cfg.init_config.max_init_state, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)

        # Disturbance params
        # [fx, fy, fz, tx, ty, tz]
        self.max_force_and_torque_disturbance = torch.tensor(
            self.cfg.disturbance.max_force_and_torque_disturbance,
            device=self.device,
            requires_grad=False,
        ).expand(self.num_envs, -1)

        # Controller params
        self.controller_input = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, requires_grad=False
        )
        self.control_allocator = ControlAllocator(
            num_envs=self.num_envs,
            dt=self.dt,
            config=self.cfg.control_allocator_config,
            device=self.device,
        )

        self.body_vel_linear_damping_coefficient = torch.tensor(
            self.cfg.damping.linvel_linear_damping_coefficient,
            device=self.device,
            requires_grad=False,
        )

        self.body_vel_quadratic_damping_coefficient = torch.tensor(
            self.cfg.damping.linvel_quadratic_damping_coefficient,
            device=self.device,
            requires_grad=False,
        )

        self.angvel_linear_damping_coefficient = torch.tensor(
            self.cfg.damping.angular_linear_damping_coefficient,
            device=self.device,
            requires_grad=False,
        )

        self.angvel_quadratic_damping_coefficient = torch.tensor(
            self.cfg.damping.angular_quadratic_damping_coefficient,
            device=self.device,
            requires_grad=False,
        )
        if self.force_application_level == "motor_link":
            self.application_mask = torch.tensor(
                self.cfg.control_allocator_config.application_mask,
                device=self.device,
                requires_grad=False,
            )
        else:
            self.application_mask = torch.tensor([0], device=self.device)

        self.motor_directions = torch.tensor(
            self.cfg.control_allocator_config.motor_directions,
            device=self.device,
            requires_grad=False,
        )

        self.output_forces = torch.zeros_like(
            global_tensor_dict["robot_force_tensor"], device=self.device, requires_grad=False
        )
        self.output_torques = torch.zeros_like(
            global_tensor_dict["robot_torque_tensor"], device=self.device, requires_grad=False
        )

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs))

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        # robot_state is defined as a tensor of shape (num_envs, 13)
        # init_state tensor if of the format [ratio_x, ratio_y, ratio_z, roll, pitch, yaw, 1.0 (for maintaining shape), vx, vy, vz, wx, wy, wz]
        random_state = torch_rand_float_tensor(self.min_init_state, self.max_init_state)

        self.robot_state[env_ids, 0:3] = torch_interpolate_ratio(
            self.env_bounds_min, self.env_bounds_max, random_state[:, 0:3]
        )[env_ids]

        # logger.debug(
        #     f"Random state: {random_state[0]}, min init state: {self.min_init_state[0]}, max init state: {self.max_init_state[0]}"
        # )
        # logger.debug(
        #     f"env_bounds_min: {self.env_bounds_min[0]}, env_bounds_max: {self.env_bounds_max[0]}"
        # )

        # quat conversion is handled separately
        self.robot_state[env_ids, 3:7] = quat_from_euler_xyz_tensor(random_state[env_ids, 3:6])

        self.robot_state[env_ids, 7:10] = random_state[env_ids, 7:10]
        self.robot_state[env_ids, 10:13] = random_state[env_ids, 10:13]

        self.controller.randomize_params(env_ids=env_ids)

        # update the states after resetting because the RL agent gets the first state after reset
        self.update_states()

    def clip_actions(self):
        """
        Clip the action tensor to the range of the controller inputs.
        """
        self.action_tensor[:] = torch.clamp(self.action_tensor, -10.0, 10.0)

    def apply_disturbance(self):
        if not self.cfg.disturbance.enable_disturbance:
            return
        disturbance_occurence = torch.bernoulli(
            self.cfg.disturbance.prob_apply_disturbance
            * torch.ones((self.num_envs), device=self.device)
        )
        # logger.debug(
        #     f"Applying disturbance to {disturbance_occurence.sum().item()} out of {self.num_envs} environments"
        # )
        # logger.debug(
        #     f"Shape of disturbance tensors: {self.robot_force_tensors.shape}, {self.robot_torque_tensors.shape}"
        # )
        # logger.debug(f"Disturbance shape: {disturbance_occurence.unsqueeze(1).shape}")
        self.robot_force_tensors[:, 0, 0:3] += torch_rand_float_tensor(
            -self.max_force_and_torque_disturbance[:, 0:3],
            self.max_force_and_torque_disturbance[:, 0:3],
        ) * disturbance_occurence.unsqueeze(1)
        self.robot_torque_tensors[:, 0, 0:3] += torch_rand_float_tensor(
            -self.max_force_and_torque_disturbance[:, 3:6],
            self.max_force_and_torque_disturbance[:, 3:6],
        ) * disturbance_occurence.unsqueeze(1)

    def control_allocation(self, command_wrench, output_mode):
        """
        Allocate the thrust and torque commands to the motors. The motor model is also used to update the motor thrusts.
        """

        forces, torques = self.control_allocator.allocate_output(command_wrench, output_mode)

        self.output_forces[:, self.application_mask, :] = forces
        self.output_torques[:, self.application_mask, :] = torques

    def call_controller(self):
        """
        Convert the action tensor to the controller inputs. The action tensor is the input and can be parametrized as desired by the user.
        This function serves the purpose of converting the action tensor to the controller inputs.
        """
        self.clip_actions()
        controller_output = self.controller(self.action_tensor)
        self.control_allocation(controller_output, self.output_mode)

        self.robot_force_tensors[:] = self.output_forces
        self.robot_torque_tensors[:] = self.output_torques

    def update_states(self):
        self.robot_euler_angles[:] = get_euler_xyz_tensor(self.robot_orientation)
        self.robot_vehicle_orientation[:] = vehicle_frame_quat_from_quat(self.robot_orientation)
        self.robot_vehicle_linvel[:] = quat_rotate_inverse(
            self.robot_vehicle_orientation, self.robot_linvel
        )
        self.robot_body_linvel[:] = quat_rotate_inverse(self.robot_orientation, self.robot_linvel)
        self.robot_body_angvel[:] = quat_rotate_inverse(self.robot_orientation, self.robot_angvel)

    def simulate_drag(self):
        self.robot_body_vel_drag_linear = (
            -self.body_vel_linear_damping_coefficient * self.robot_body_linvel
        )
        self.robot_body_vel_drag_quadratic = (
            -self.body_vel_quadratic_damping_coefficient
            * self.robot_body_linvel.abs()
            * self.robot_body_linvel
        )
        self.robot_body_vel_drag = (
            self.robot_body_vel_drag_linear + self.robot_body_vel_drag_quadratic
        )
        self.robot_force_tensors[:, 0, 0:3] += self.robot_body_vel_drag

        self.robot_body_angvel_drag_linear = (
            -self.angvel_linear_damping_coefficient * self.robot_body_angvel
        )
        self.robot_body_angvel_drag_quadratic = (
            -self.angvel_quadratic_damping_coefficient
            * self.robot_body_angvel.abs()
            * self.robot_body_angvel
        )
        self.robot_body_angvel_drag = (
            self.robot_body_angvel_drag_linear + self.robot_body_angvel_drag_quadratic
        )
        self.robot_torque_tensors[:, 0, 0:3] += self.robot_body_angvel_drag

    def step(self, action_tensor):
        """
        Update the state of the quadrotor. This function is called every simulation step.
        """
        self.update_states()
        if action_tensor.shape[0] != self.num_envs:
            raise ValueError("Action tensor does not have the correct number of environments")
        self.action_tensor[:] = action_tensor
        # calling controller leads to control allocation happening, and
        self.call_controller()
        self.simulate_drag()
        self.apply_disturbance()
