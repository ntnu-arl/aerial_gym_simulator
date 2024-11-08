from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import (
    BaseDepthCameraConfig,
)
import numpy as np


class BaseNormalFaceIDCameraConfig(BaseDepthCameraConfig):
    num_sensors = 1  # number of sensors of this type

    sensor_type = "normal_faceID_camera"  # sensor type

    # If you use more than one sensors above, there is a need to specify the sensor placement for each sensor
    # this can be added here, but the user can implement this if needed.

    height = 270
    width = 480
    horizontal_fov_deg = 87.000
    # camera params VFOV is calcuated from the aspect ratio and HFOV
    # VFOV = 2 * atan(tan(HFOV/2) / aspect_ratio)
    max_range = 10.0
    min_range = 0.2

    return_pointcloud = True  # normal information comes in the form of a pointcloud

    normal_in_world_frame = True
    # transform from sensor element coordinate frame to sensor_base_link frame
    # euler_frame_rot_deg = [-90.0, 0, -90.0]

    # randomize placement of the sensor
    randomize_placement = False
    min_translation = [0.07, -0.06, 0.01]
    max_translation = [0.12, 0.03, 0.04]
    min_euler_rotation_deg = [-5.0, -5.0, -5.0]
    max_euler_rotation_deg = [5.0, 5.0, 5.0]

    use_collision_geometry = False

    class sensor_noise:
        enable_sensor_noise = False
        pixel_dropout_prob = 0.01
        pixel_std_dev_multiplier = 0.01
