# import nvtx
import warp as wp
import math

from aerial_gym.sensors.warp.warp_kernels.warp_camera_kernels import (
    DepthCameraWarpKernels,
)


class WarpCam:
    def __init__(self, num_envs, config, mesh_ids_array, device="cuda:0"):
        self.cfg = config
        self.num_envs = num_envs
        self.num_sensors = self.cfg.num_sensors
        self.mesh_ids_array = mesh_ids_array

        self.width = self.cfg.width
        self.height = self.cfg.height

        self.horizontal_fov = math.radians(self.cfg.horizontal_fov_deg)
        self.far_plane = self.cfg.max_range
        self.calculate_depth = self.cfg.calculate_depth
        self.device = device

        self.camera_position_array = None
        self.camera_orientation_array = None
        self.graph = None

        self.initialize_camera_matrices()

    def initialize_camera_matrices(self):
        # Calculate camera params
        W = self.width
        H = self.height
        (u_0, v_0) = (W / 2, H / 2)
        f = W / 2 * 1 / math.tan(self.horizontal_fov / 2)

        vertical_fov = 2 * math.atan(H / (2 * f))
        alpha_u = u_0 / math.tan(self.horizontal_fov / 2)
        alpha_v = v_0 / math.tan(vertical_fov / 2)

        # simple pinhole model
        self.K = wp.mat44(
            alpha_u,
            0.0,
            u_0,
            0.0,
            0.0,
            alpha_v,
            v_0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        self.K_inv = wp.inverse(self.K)

        self.c_x = int(u_0)
        self.c_y = int(v_0)

    def create_render_graph_pointcloud(self, debug=False):
        if not debug:
            print(f"creating render graph")
            wp.capture_begin(device=self.device)
        # with wp.ScopedTimer("render"):
        if self.cfg.segmentation_camera == True:
            wp.launch(
                kernel=DepthCameraWarpKernels.draw_optimized_kernel_pointcloud_segmentation,
                dim=(self.num_envs, self.num_sensors, self.width, self.height),
                inputs=[
                    self.mesh_ids_array,
                    self.camera_position_array,
                    self.camera_orientation_array,
                    self.K_inv,
                    self.far_plane,
                    self.pixels,
                    self.segmentation_pixels,
                    self.c_x,
                    self.c_y,
                    self.pointcloud_in_world_frame,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                kernel=DepthCameraWarpKernels.draw_optimized_kernel_pointcloud,
                dim=(self.num_envs, self.num_sensors, self.width, self.height),
                inputs=[
                    self.mesh_ids_array,
                    self.camera_position_array,
                    self.camera_orientation_array,
                    self.K_inv,
                    self.far_plane,
                    self.pixels,
                    self.c_x,
                    self.c_y,
                    self.pointcloud_in_world_frame,
                ],
                device=self.device,
            )
        if not debug:
            print(f"finishing capture of render graph")
            self.graph = wp.capture_end(device=self.device)

    def create_render_graph_depth_range(self, debug=False):
        if not debug:
            print(f"creating render graph")
            wp.capture_begin(device=self.device)
        # with wp.ScopedTimer("render"):
        if self.cfg.segmentation_camera == True:
            wp.launch(
                kernel=DepthCameraWarpKernels.draw_optimized_kernel_depth_range_segmentation,
                dim=(self.num_envs, self.num_sensors, self.width, self.height),
                inputs=[
                    self.mesh_ids_array,
                    self.camera_position_array,
                    self.camera_orientation_array,
                    self.K_inv,
                    self.far_plane,
                    self.pixels,
                    self.segmentation_pixels,
                    self.c_x,
                    self.c_y,
                    self.calculate_depth,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                kernel=DepthCameraWarpKernels.draw_optimized_kernel_depth_range,
                dim=(self.num_envs, self.num_sensors, self.width, self.height),
                inputs=[
                    self.mesh_ids_array,
                    self.camera_position_array,
                    self.camera_orientation_array,
                    self.K_inv,
                    self.far_plane,
                    self.pixels,
                    self.c_x,
                    self.c_y,
                    self.calculate_depth,
                ],
                device=self.device,
            )
        if not debug:
            print(f"finishing capture of render graph")
            self.graph = wp.capture_end(device=self.device)

    def set_image_tensors(self, pixels, segmentation_pixels=None):
        # init buffers. None when uninitialized
        if self.cfg.return_pointcloud:
            self.pixels = wp.from_torch(pixels, dtype=wp.vec3)
            self.pointcloud_in_world_frame = self.cfg.pointcloud_in_world_frame
        else:
            self.pixels = wp.from_torch(pixels, dtype=wp.float32)

        if self.cfg.segmentation_camera == True:
            self.segmentation_pixels = wp.from_torch(segmentation_pixels, dtype=wp.int32)
        else:
            self.segmentation_pixels = segmentation_pixels

    def set_pose_tensor(self, positions, orientations):
        self.camera_position_array = wp.from_torch(positions, dtype=wp.vec3)
        self.camera_orientation_array = wp.from_torch(orientations, dtype=wp.quat)

    # @nvtx.annotate()
    def capture(self, debug=False):
        if self.graph is None:
            if self.cfg.return_pointcloud:
                self.create_render_graph_pointcloud(debug=debug)
            else:
                self.create_render_graph_depth_range(debug=debug)

        if self.graph is not None:
            wp.capture_launch(self.graph)

        return wp.to_torch(self.pixels)
