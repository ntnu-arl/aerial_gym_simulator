from aerial_gym.config.sensor_config.lidar_config.base_lidar_config import BaseLidarConfig


class OSDome_64_Config(BaseLidarConfig):
    # keep everything pretty much the same and change the number of vertical rays
    height = 64
    width = 512
    horizontal_fov_deg_min = -180
    horizontal_fov_deg_max = 180
    vertical_fov_deg_min = 0
    vertical_fov_deg_max = 90
    max_range = 20.0
    min_range = 0.5

    return_pointcloud = False
    segmentation_camera = True

    # randomize placement of the sensor
    randomize_placement = False
    min_translation = [0.0, 0.0, 0.0]
    max_translation = [0.0, 0.0, 0.0]
    # example of a front-mounted dome lidar
    min_euler_rotation_deg = [0.0, 0.0, 0.0]
    max_euler_rotation_deg = [0.0, 0.0, 0.0]

    class sensor_noise:
        enable_sensor_noise = False
        std_a = 0.00038089
        std_b = -0.00343351
        std_c = 0.01553284
        mean_offset = -0.025
        pixel_dropout_prob = 0.0

