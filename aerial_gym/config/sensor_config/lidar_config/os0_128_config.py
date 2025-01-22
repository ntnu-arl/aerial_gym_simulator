from aerial_gym.config.sensor_config.lidar_config.base_lidar_config import BaseLidarConfig
import numpy as np


class OS_0_128_Config(BaseLidarConfig):
    # standard OS0-128 configuration
    height = 128
    width = 512
    horizontal_fov_deg_min = -180
    horizontal_fov_deg_max = 180
    vertical_fov_deg_min = -45
    vertical_fov_deg_max = +45

    # min and max range do not match with the real sensor, but here we get to limit it for our convenience
    max_range = 35.0
    min_range = 0.2

    # Type of lidar (range, pointcloud, segmentation)
    # You can combine: (range+segmentation), (pointcloud+segmentation)
    # Other combinations are trivial and you can add support for them in the code if you want.

    return_pointcloud = (
        False  # Return a pointcloud instead of an image. Range image will be returned by default
    )
    pointcloud_in_world_frame = False
    segmentation_camera = True  # Setting to true will return a segmentation image along with the range image or pointcloud

    # transform from sensor element coordinate frame to sensor_base_link frame
    euler_frame_rot_deg = [0.0, 0.0, 0.0]

    # Type of data to be returned from the sensor
    normalize_range = True  # will be set to false when pointcloud is in world frame

    # do not change this.
    normalize_range = (
        False
        if (return_pointcloud == True and pointcloud_in_world_frame == True)
        else normalize_range
    )  # divide by max_range. Ignored when pointcloud is in world frame

    # what to do with out of range values
    far_out_of_range_value = (
        max_range if normalize_range == True else -1.0
    )  # Will be [-1]U[0,1] if normalize_range is True, otherwise will be value set by user in place of -1.0
    near_out_of_range_value = (
        -max_range if normalize_range == True else -1.0
    )  # Will be [-1]U[0,1] if normalize_range is True, otherwise will be value set by user in place of -1.0

    class sensor_noise:
        enable_sensor_noise = False
        std_a = 3.36239104e-05
        std_b = -3.17199061e-04
        std_c = 9.61903860e-03
        mean_offset = -0.05
        pixel_dropout_prob = 0.0
