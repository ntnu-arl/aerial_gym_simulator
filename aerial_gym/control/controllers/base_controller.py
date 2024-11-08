import torch


class BaseController:
    def __init__(self, control_config, num_envs, device, mode="robot"):
        self.cfg = control_config
        self.num_envs = num_envs
        self.device = device
        self.mode = mode

    def init_tensors(self, global_tensor_dict):
        if self.mode == "robot":
            self.robot_position = global_tensor_dict["robot_position"]
            self.robot_orientation = global_tensor_dict["robot_orientation"]
            self.robot_linvel = global_tensor_dict["robot_linvel"]
            self.robot_angvel = global_tensor_dict["robot_angvel"]
            self.robot_vehicle_orientation = global_tensor_dict["robot_vehicle_orientation"]
            self.robot_vehicle_linvel = global_tensor_dict["robot_vehicle_linvel"]
            self.robot_body_angvel = global_tensor_dict["robot_body_angvel"]
            self.robot_body_linvel = global_tensor_dict["robot_body_linvel"]
            self.robot_euler_angles = global_tensor_dict["robot_euler_angles"]
            self.mass = global_tensor_dict["robot_mass"].unsqueeze(1)
            self.robot_inertia = global_tensor_dict["robot_inertia"]
            self.gravity = global_tensor_dict["gravity"]
        if self.mode == "obstacle":
            self.robot_position = global_tensor_dict["obstacle_position"]
            self.robot_orientation = global_tensor_dict["obstacle_orientation"]
            self.robot_linvel = global_tensor_dict["obstacle_linvel"]
            self.robot_angvel = global_tensor_dict["obstacle_angvel"]
            self.robot_vehicle_orientation = global_tensor_dict["obstacle_vehicle_orientation"]
            self.robot_vehicle_linvel = global_tensor_dict["obstacle_vehicle_linvel"]
            self.robot_body_angvel = global_tensor_dict["obstacle_body_angvel"]
            self.robot_body_linvel = global_tensor_dict["obstacle_body_linvel"]
            self.robot_euler_angles = global_tensor_dict["obstacle_euler_angles"]
            self.mass = 1.0
            self.robot_inertia = torch.eye(3, device=self.device)
            self.gravity = global_tensor_dict["gravity"]
