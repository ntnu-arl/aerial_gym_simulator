from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import BaseDepthCameraConfig


class StereoCameraConfig(BaseDepthCameraConfig):
    sensor_type = "stereo_camera"  # sensor type
    height = 270
    width = 480

    baseline = -0.095 # baseline. Distance from the left to right camera in metres. +y is positive

