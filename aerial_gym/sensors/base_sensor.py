from abc import ABC, abstractmethod
import math


class BaseSensor(ABC):
    def __init__(self, sensor_config, num_envs, device):
        self.cfg = sensor_config
        self.device = device
        self.num_envs = num_envs
        self.robot_position = None
        self.robot_orientation = None
        self.robot_linvel = None
        self.robot_angvel = None

    @abstractmethod
    def init_tensors(self, global_tensor_dict):
        # for warp sensors
        self.robot_position = global_tensor_dict["robot_position"]
        self.robot_orientation = global_tensor_dict["robot_orientation"]

        # for IMU
        self.gravity = global_tensor_dict["gravity"]
        self.dt = global_tensor_dict["dt"]
        self.sqrt_dt = math.sqrt(self.dt)
        self.robot_masses = global_tensor_dict["robot_mass"]

        if self.cfg.sensor_type in ["lidar", "camera", "stereo_camera"]:
            # for IGE and warp sensors
            self.pixels = global_tensor_dict["depth_range_pixels"]
            if self.cfg.segmentation_camera:
                self.segmentation_pixels = global_tensor_dict["segmentation_pixels"]
            else:
                self.segmentation_pixels = None
        elif self.cfg.sensor_type in ["normal_faceID_lidar", "normal_faceID_camera"]:
            self.pixels = global_tensor_dict["depth_range_pixels"]
            self.segmentation_pixels = global_tensor_dict["segmentation_pixels"]

        else:
            # for IMU (maybe for motion blur later. Who knows? ¯\_(ツ)_/¯
            self.robot_linvel = global_tensor_dict["robot_linvel"]
            self.robot_angvel = global_tensor_dict["robot_angvel"]
            self.robot_body_angvel = global_tensor_dict["robot_body_angvel"]
            self.robot_body_linvel = global_tensor_dict["robot_body_linvel"]
            self.robot_euler_angles = global_tensor_dict["robot_euler_angles"]

    @abstractmethod
    def update(self):
        raise NotImplementedError("update not implemented")

    @abstractmethod
    def reset_idx(self):
        raise NotImplementedError("reset_idx not implemented")

    @abstractmethod
    def reset(self):
        raise NotImplementedError("reset not implemented")

    @staticmethod
    def print_params(self):
        for name, value in vars(self).items():
            # if dtype is a valid field, print it as well
            if hasattr(value, "dtype"):
                print(name, type(value), value.dtype)
            else:
                print(name, type(value))
