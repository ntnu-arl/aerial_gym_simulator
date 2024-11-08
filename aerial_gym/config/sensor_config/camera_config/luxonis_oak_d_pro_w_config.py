from aerial_gym.config.sensor_config.camera_config.luxonis_oak_d_config import (
    LuxonisOakDConfig,
)


class LuxonisOakDProWConfig(LuxonisOakDConfig):
    height = 270
    width = 480
    horizontal_fov_deg = 127.0
    # camera params VFOV is calcuated from the aspect ratio and HFOV
    # VFOV = 2 * atan(tan(HFOV/2) / aspect_ratio)
    max_range = 12.0
    min_range = 0.2
