from aerial_gym.sensors.base_sensor import BaseSensor

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("IsaacGymCameraSensor")


class IsaacGymCameraSensor(BaseSensor):
    """
    Camera sensor class for Isaac Gym. Inherits from BaseSensor.
    Supports depth and semantic segmentation images. Color image support is not yet implemented.
    """

    def __init__(self, sensor_config, num_envs, gym, sim, device):
        super().__init__(sensor_config=sensor_config, num_envs=num_envs, device=device)
        self.device = device
        self.num_envs = num_envs
        self.cfg = sensor_config
        self.gym = gym
        self.sim = sim
        logger.warning("Initializing Isaac Gym Camera Sensor")
        logger.debug(f"Camera sensor config: {self.cfg.__dict__}")
        self.init_cam_config()
        self.depth_tensors = []
        self.segmentation_tensors = []
        self.color_tensors = []
        self.cam_handles = []

    def init_cam_config(self):
        """
        Initialize the camera properties and local transform for the camera sensor. Uses the sensor params from the config file.

        Args:
        - None

        Returns:
        - None
        """
        # Set Camera Properties
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        camera_props.width = self.cfg.width
        camera_props.height = self.cfg.height
        camera_props.far_plane = self.cfg.max_range
        camera_props.near_plane = self.cfg.min_range
        camera_props.horizontal_fov = self.cfg.horizontal_fov_deg
        camera_props.use_collision_geometry = self.cfg.use_collision_geometry
        # camera supersampling is unchanged for now
        self.camera_properties = camera_props

        # local transform for camera
        self.local_transform = gymapi.Transform()
        self.local_transform.p = gymapi.Vec3(
            self.cfg.nominal_position[0],
            self.cfg.nominal_position[1],
            self.cfg.nominal_position[2],
        )
        angle_euler = torch.deg2rad(
            torch.tensor(
                self.cfg.nominal_orientation_euler_deg,
                device=self.device,
                requires_grad=False,
            )
        )
        angle_quat = quat_from_euler_xyz(angle_euler[0], angle_euler[1], angle_euler[2])
        self.local_transform.r = gymapi.Quat(
            angle_quat[0], angle_quat[1], angle_quat[2], angle_quat[3]
        )

    def add_sensor_to_env(self, env_id, env_handle, actor_handle):
        """
        Add the camera sensor to the environment. Set each camera sensor with appriopriate properties, and attach it to the actor.\
        The camera sensor is attached to the actor using the pose_handle, which is the handle of the actor's pose in the environment.

        Args:
        - env_handle: handle of the environment
        - actor_handle: handle of the actor
        - pose_handle: handle of the actor's pose in the environment

        Returns:
        - None
        """
        logger.debug(f"Adding camera sensor to env {env_handle} and actor {actor_handle}")
        if len(self.cam_handles) == env_id:
            self.depth_tensors.append([])
            self.segmentation_tensors.append([])
            self.color_tensors.append([])
        self.cam_handle = self.gym.create_camera_sensor(env_handle, self.camera_properties)
        self.cam_handles.append(self.cam_handle)
        self.gym.attach_camera_to_body(
            self.cam_handle,
            env_handle,
            actor_handle,
            self.local_transform,
            gymapi.FOLLOW_TRANSFORM,
        )
        self.depth_tensors[env_id].append(
            gymtorch.wrap_tensor(
                self.gym.get_camera_image_gpu_tensor(
                    self.sim, env_handle, self.cam_handle, gymapi.IMAGE_DEPTH
                )
            )
        )
        self.segmentation_tensors[env_id].append(
            gymtorch.wrap_tensor(
                self.gym.get_camera_image_gpu_tensor(
                    self.sim, env_handle, self.cam_handle, gymapi.IMAGE_SEGMENTATION
                )
            )
        )
        self.color_tensors[env_id].append(
            gymtorch.wrap_tensor(
                self.gym.get_camera_image_gpu_tensor(
                    self.sim, env_handle, self.cam_handle, gymapi.IMAGE_COLOR
                )
            )
        )
        # self.color_tensors.append(gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, self.cam_handle, gymapi.IMAGE_COLOR)))
        logger.debug(f"Camera sensor added to env {env_handle} and actor {actor_handle}")

    def init_tensors(self, global_tensor_dict):
        """
        Initialize the tensors for the camera sensor. Depth tensors are mandatory, semantic tensors are optional.
        Args:
        - depth_tensors: list of depth tensors for each environment
        - segmentation_tensors: list of semantic tensors for each environment

        Returns:
        - None
        """
        super().init_tensors(global_tensor_dict)

        # At some point, RGB cam support for Warp would be added on our end. Please use Isaac Gym's native RGB Camera till then.
        self.rgb_pixels = global_tensor_dict["rgb_pixels"]

    def capture(self):
        """
        In the case of Isaac Gym cameras, it involves triggering the sensors to capture the images after fetch_results is run.
        Subsequently, the images have to be stored individually in the relevant tensor slices.
        Start and end image access needs to be done after reacding images.
        """
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        for env_id in range(self.num_envs):
            for cam_id in range(self.cfg.num_sensors):
                # the depth values are in -ve z axis, so we need to flip it to positive
                self.pixels[env_id, cam_id] = -self.depth_tensors[env_id][cam_id]
                self.rgb_pixels[env_id, cam_id] = self.color_tensors[env_id][cam_id]
                if self.cfg.segmentation_camera:
                    self.segmentation_pixels[env_id, cam_id] = self.segmentation_tensors[env_id][
                        cam_id
                    ]
        self.gym.end_access_image_tensors(self.sim)

    def update(self):
        """
        Update the camera sensor. Capture image, apply the same post-processing as other cameras.
        The values in the depth tensor are set to the aceptable limits and normalized if required.
        """
        self.capture()
        self.apply_noise()
        self.apply_range_limits()
        self.normalize_observation()

    def apply_range_limits(self):
        """ """
        # logger.debug("Applying range limits")
        self.pixels[self.pixels > self.cfg.max_range] = self.cfg.far_out_of_range_value
        self.pixels[self.pixels < self.cfg.min_range] = self.cfg.near_out_of_range_value
        # logger.debug("[DONE] Applying range limits")

    def normalize_observation(self):
        if self.cfg.normalize_range and self.cfg.pointcloud_in_world_frame == False:
            # logger.debug("Normalizing pointcloud values")
            self.pixels[:] = self.pixels / self.cfg.max_range
        if self.cfg.pointcloud_in_world_frame == True:
            logger.error("Pointcloud is in world frame. Not supported for this sensor")

    def apply_noise(self):
        if self.cfg.sensor_noise.enable_sensor_noise == True:
            # logger.debug("Applying sensor noise")
            self.pixels[:] = torch.normal(
                mean=self.pixels, std=self.cfg.sensor_noise.pixel_std_dev_multiplier * self.pixels
            )
            self.pixels[
                torch.bernoulli(torch.ones_like(self.pixels) * self.cfg.sensor_noise.pixel_dropout_prob) > 0
            ] = self.cfg.near_out_of_range_value

    def reset_idx(self, env_ids):
        """
        Reset the camera pose for the specified env_ids. Nothing to be done for Isaac Gym's camera sensor
        Changing the pose for each camera sensor w.r.t actor requires a very expensive loop operation.
        """
        # Nothing to be doen here for Isaac Gym's camera sensor
        pass

    def reset(self):
        """
        Reset the camera pose for all envs. Nothing to be done for Isaac Gym's camera sensor.
        Changing the pose for each camera sensor w.r.t actor requires a very expensive loop operation.
        """
        # Nothing to be doen here for Isaac Gym's camera sensor
        pass

    def get_observation(self):
        return self.pixels, self.segmentation_pixels
