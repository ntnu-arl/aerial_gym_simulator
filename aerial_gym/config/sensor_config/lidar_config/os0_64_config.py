from aerial_gym.config.sensor_config.lidar_config.os0_128_config import OS_0_128_Config


class OS_0_64_Config(OS_0_128_Config):
    # keep everything pretty much the same and change the number of vertical rays
    height = 64

    class sensor_noise:
        enable_sensor_noise = False
        std_a = 3.36239104e-05
        std_b = -3.17199061e-04
        std_c = 9.61903860e-03
        mean_offset = -0.025
        pixel_dropout_prob = 0.0
