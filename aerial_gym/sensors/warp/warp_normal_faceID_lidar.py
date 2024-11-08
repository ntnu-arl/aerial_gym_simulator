import torch
import math

# import nvtx
import warp as wp

from aerial_gym.sensors.warp.warp_kernels.warp_lidar_kernels import LidarWarpKernels


class WarpNormalFaceIDLidar:
    def __init__(self, num_envs, config, mesh_ids_array, device="cuda:0"):
        self.cfg = config
        self.num_envs = num_envs
        self.num_sensors = self.cfg.num_sensors
        self.mesh_ids_array = mesh_ids_array
        self.num_scan_lines = self.cfg.height
        self.num_points_per_line = self.cfg.width
        self.horizontal_fov_min = math.radians(self.cfg.horizontal_fov_deg_min)
        self.horizontal_fov_max = math.radians(self.cfg.horizontal_fov_deg_max)
        self.horizontal_fov = self.horizontal_fov_max - self.horizontal_fov_min
        self.horizontal_fov_mean = (self.horizontal_fov_max + self.horizontal_fov_min) / 2
        if self.horizontal_fov > 2 * math.pi:
            raise ValueError("Horizontal FOV must be less than 2pi")

        self.vertical_fov_min = math.radians(self.cfg.vertical_fov_deg_min)
        self.vertical_fov_max = math.radians(self.cfg.vertical_fov_deg_max)
        self.vertical_fov = self.vertical_fov_max - self.vertical_fov_min
        self.vertical_fov_mean = (self.vertical_fov_max + self.vertical_fov_min) / 2
        if self.vertical_fov > math.pi:
            raise ValueError("Vertical FOV must be less than pi")
        self.far_plane = self.cfg.max_range
        self.device = device

        self.lidar_position_array = None
        self.lidar_quat_array = None
        self.graph = None

        self.pixels = None
        self.face_pixels = None

        self.normal_in_world_frame = self.cfg.normal_in_world_frame

        self.initialize_ray_vectors()

    def initialize_ray_vectors(self):
        # populate a 2D torch array with the ray vectors that are 2d arrays of wp.vec3
        ray_vectors = torch.zeros(
            (self.num_scan_lines, self.num_points_per_line, 3),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        for i in range(self.num_scan_lines):
            for j in range(self.num_points_per_line):
                # Rays go from +HFoV/2 to -HFoV/2 and +VFoV/2 to -VFoV/2
                azimuth_angle = self.horizontal_fov_max - (
                    self.horizontal_fov_max - self.horizontal_fov_min
                ) * (j / (self.num_points_per_line - 1))
                elevation_angle = self.vertical_fov_max - (
                    self.vertical_fov_max - self.vertical_fov_min
                ) * (i / (self.num_scan_lines - 1))
                ray_vectors[i, j, 0] = math.cos(azimuth_angle) * math.cos(elevation_angle)
                ray_vectors[i, j, 1] = math.sin(azimuth_angle) * math.cos(elevation_angle)
                ray_vectors[i, j, 2] = math.sin(elevation_angle)
        # normalize ray_vectors
        ray_vectors = ray_vectors / torch.norm(ray_vectors, dim=2, keepdim=True)

        # recast as 2D warp array of vec3
        self.ray_vectors = wp.from_torch(ray_vectors, dtype=wp.vec3)

    def create_render_graph_pointcloud(self, debug=False):
        if not debug:
            print(f"creating render graph")
            wp.capture_begin(device=self.device)
        wp.launch(
            kernel=LidarWarpKernels.draw_optimized_kernel_normal_faceID,
            dim=(
                self.num_envs,
                self.num_sensors,
                self.num_scan_lines,
                self.num_points_per_line,
            ),
            inputs=[
                self.mesh_ids_array,
                self.lidar_position_array,
                self.lidar_quat_array,
                self.ray_vectors,
                self.far_plane,
                self.pixels,
                self.face_pixels,
                self.normal_in_world_frame,
            ],
            device=self.device,
        )
        if not debug:
            print(f"finishing capture of render graph")
            self.graph = wp.capture_end(device=self.device)

    def set_image_tensors(self, pixels, segmentation_pixels=None):
        # init buffers. None when uninitialized
        self.pixels = wp.from_torch(pixels, dtype=wp.vec3)
        self.face_pixels = wp.from_torch(segmentation_pixels, dtype=wp.int32)

    def set_pose_tensor(self, positions, orientations):
        self.lidar_position_array = wp.from_torch(positions, dtype=wp.vec3)
        self.lidar_quat_array = wp.from_torch(orientations, dtype=wp.quat)

    # @nvtx.annotate()
    def capture(self, debug=False):
        if self.graph is None:
            self.create_render_graph_pointcloud(debug)
        if self.graph is not None:
            wp.capture_launch(self.graph)

        return wp.to_torch(self.pixels)
