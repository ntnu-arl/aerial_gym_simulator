import numpy as np

from aerial_gym import AERIAL_GYM_DIRECTORY

from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import (
    BaseDepthCameraConfig,
)
from aerial_gym.config.sensor_config.lidar_config.base_lidar_config import (
    BaseLidarConfig,
)
from aerial_gym.config.sensor_config.lidar_config.osdome_64_config import OSDome_64_Config
from aerial_gym.config.sensor_config.imu_config.base_imu_config import BaseImuConfig

from aerial_gym.config.robot_config.base_quad_config import BaseQuadCfg


class BaseQuadRootLinkControlCfg(BaseQuadCfg):

    # everything is pretty much the same as the BaseQuadCfg except for the following
    class robot_asset(BaseQuadCfg.robot_asset):
        file = "model.urdf"
        name = "base_quadrotor"  # actor name
        base_link_name = "base_link"

    class control_allocator_config:
        num_motors = 4
        force_application_level = "root_link"  # "motor_link" or "root_link" decides to apply combined forces acting on the robot at the root link or at the individual motor links

        application_mask = [1 + 4 + i for i in range(0, 4)]
        motor_directions = [1, -1, 1, -1]

        allocation_matrix = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [-0.13, -0.13, 0.13, 0.13],
            [-0.13, 0.13, 0.13, -0.13],
            [0.01, -0.01, 0.01, -0.01],
        ]

        class motor_model_config:
            motor_time_constant_min = 0.01
            motor_time_constant_max = 0.03
            max_thrust = 10
            min_thrust = 0
            max_thrust_rate = 100.0
            thrust_to_torque_ratio = 0.01
