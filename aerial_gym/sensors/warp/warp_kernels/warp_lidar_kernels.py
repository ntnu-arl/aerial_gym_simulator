import warp as wp

NO_HIT_RAY_VAL = wp.constant(1000.0)
NO_HIT_SEGMENTATION_VAL = wp.constant(wp.int32(-2))


class LidarWarpKernels:
    def __init__(self):
        pass

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_pointcloud(
        mesh_ids: wp.array(dtype=wp.uint64),
        lidar_pos_array: wp.array(dtype=wp.vec3, ndim=2),
        lidar_quat_array: wp.array(dtype=wp.quat, ndim=2),
        ray_vectors: wp.array2d(dtype=wp.vec3),
        # ray_noise_magnitude: wp.array(dtype=float),
        far_plane: float,
        pixels: wp.array(dtype=wp.vec3, ndim=4),
        pointcloud_in_world_frame: bool,
    ):

        env_id, cam_id, scan_line, point_index = wp.tid()
        mesh = mesh_ids[env_id]
        lidar_position = lidar_pos_array[env_id, cam_id]
        lidar_quaternion = lidar_quat_array[env_id, cam_id]
        ray_origin = lidar_position
        # perturb ray_vectors with uniform noise
        ray_dir = ray_vectors[scan_line, point_index]  # + sampled_vec3_noise
        ray_dir = wp.normalize(ray_dir)
        ray_direction_world = wp.normalize(wp.quat_rotate(lidar_quaternion, ray_dir))
        t = float(0.0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)
        n = wp.vec3()
        f = int(0)
        dist = NO_HIT_RAY_VAL
        if wp.mesh_query_ray(mesh, ray_origin, ray_direction_world, far_plane, t, u, v, sign, n, f):
            dist = t
        if pointcloud_in_world_frame:
            pixels[env_id, cam_id, scan_line, point_index] = ray_origin + dist * ray_direction_world
        else:
            pixels[env_id, cam_id, scan_line, point_index] = dist * ray_dir

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_pointcloud_segmentation(
        mesh_ids: wp.array(dtype=wp.uint64),
        lidar_pos_array: wp.array(dtype=wp.vec3, ndim=2),
        lidar_quat_array: wp.array(dtype=wp.quat, ndim=2),
        ray_vectors: wp.array2d(dtype=wp.vec3),
        # ray_noise_magnitude: wp.array(dtype=float),
        far_plane: float,
        pixels: wp.array(dtype=wp.vec3, ndim=4),
        segmentation_pixels: wp.array(dtype=wp.int32, ndim=4),
        pointcloud_in_world_frame: bool,
    ):

        env_id, cam_id, scan_line, point_index = wp.tid()
        mesh = mesh_ids[env_id]
        lidar_position = lidar_pos_array[env_id, cam_id]
        lidar_quaternion = lidar_quat_array[env_id, cam_id]
        ray_origin = lidar_position
        # perturb ray_vectors with uniform noise
        ray_dir = ray_vectors[scan_line, point_index]  # + sampled_vec3_noise
        ray_dir = wp.normalize(ray_dir)
        ray_direction_world = wp.normalize(wp.quat_rotate(lidar_quaternion, ray_dir))
        t = float(0.0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)
        n = wp.vec3()
        f = int(0)
        dist = NO_HIT_RAY_VAL
        if wp.mesh_query_ray(mesh, ray_origin, ray_direction_world, far_plane, t, u, v, sign, n, f):
            dist = t
            mesh_obj = wp.mesh_get(mesh)
            face_index = mesh_obj.indices[f * 3]
            segmentation_value = wp.int32(mesh_obj.velocities[face_index][0])
        if pointcloud_in_world_frame:
            pixels[env_id, cam_id, scan_line, point_index] = ray_origin + dist * ray_direction_world
        else:
            pixels[env_id, cam_id, scan_line, point_index] = dist * ray_dir
        segmentation_pixels[env_id, cam_id, scan_line, point_index] = segmentation_value

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_normal_faceID(
        mesh_ids: wp.array(dtype=wp.uint64),
        lidar_pos_array: wp.array(dtype=wp.vec3, ndim=2),
        lidar_quat_array: wp.array(dtype=wp.quat, ndim=2),
        ray_vectors: wp.array2d(dtype=wp.vec3),
        # ray_noise_magnitude: wp.array(dtype=float),
        far_plane: float,
        pixels: wp.array(dtype=wp.vec3, ndim=4),
        face_pixels: wp.array(dtype=wp.int32, ndim=4),
        pointcloud_in_world_frame: bool,
    ):

        env_id, cam_id, scan_line, point_index = wp.tid()
        mesh = mesh_ids[env_id]
        lidar_position = lidar_pos_array[env_id, cam_id]
        lidar_quaternion = lidar_quat_array[env_id, cam_id]
        ray_origin = lidar_position
        # perturb ray_vectors with uniform noise
        ray_dir = ray_vectors[scan_line, point_index]  # + sampled_vec3_noise
        ray_dir = wp.normalize(ray_dir)
        ray_direction_world = wp.normalize(wp.quat_rotate(lidar_quaternion, ray_dir))
        t = float(0.0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)
        n = wp.vec3()
        f = int(-1)
        pixels[env_id, cam_id, scan_line, point_index] = n * NO_HIT_RAY_VAL
        wp.mesh_query_ray(mesh, ray_origin, ray_direction_world, far_plane, t, u, v, sign, n, f)
        if pointcloud_in_world_frame:
            pixels[env_id, cam_id, scan_line, point_index] = n
        else:
            # rotate to sensor frame
            pixels[env_id, cam_id, scan_line, point_index] = wp.normalize(
                wp.quat_rotate(wp.quat_inverse(lidar_quaternion), n)
            )
        face_pixels[env_id, cam_id, scan_line, point_index] = f

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_range_segmentation(
        mesh_ids: wp.array(dtype=wp.uint64),
        lidar_pos_array: wp.array(dtype=wp.vec3, ndim=2),
        lidar_quat_array: wp.array(dtype=wp.quat, ndim=2),
        ray_vectors: wp.array2d(dtype=wp.vec3),
        # ray_noise_magnitude: wp.array(dtype=float),
        far_plane: float,
        pixels: wp.array(dtype=float, ndim=4),
        segmentation_pixels: wp.array(dtype=wp.int32, ndim=4),
    ):
        env_id, cam_id, scan_line, point_index = wp.tid()
        mesh = mesh_ids[env_id]
        lidar_position = lidar_pos_array[env_id, cam_id]
        lidar_quaternion = lidar_quat_array[env_id, cam_id]
        ray_origin = lidar_position
        # perturb ray_vectors with uniform noise
        ray_dir = ray_vectors[scan_line, point_index]  # + sampled_vec3_noise
        ray_dir = wp.normalize(ray_dir)
        ray_direction_world = wp.normalize(wp.quat_rotate(lidar_quaternion, ray_dir))
        t = float(0.0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)
        n = wp.vec3()
        f = int(0)
        dist = NO_HIT_RAY_VAL
        segmentation_value = NO_HIT_SEGMENTATION_VAL
        if wp.mesh_query_ray(mesh, ray_origin, ray_direction_world, far_plane, t, u, v, sign, n, f):
            dist = t
            mesh_obj = wp.mesh_get(mesh)
            face_index = mesh_obj.indices[f * 3]
            segmentation_value = wp.int32(mesh_obj.velocities[face_index][0])
        pixels[env_id, cam_id, scan_line, point_index] = dist
        segmentation_pixels[env_id, cam_id, scan_line, point_index] = segmentation_value

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_range(
        mesh_ids: wp.array(dtype=wp.uint64),
        lidar_pos_array: wp.array(dtype=wp.vec3, ndim=2),
        lidar_quat_array: wp.array(dtype=wp.quat, ndim=2),
        ray_vectors: wp.array2d(dtype=wp.vec3),
        # ray_noise_magnitude: wp.array(dtype=float),
        far_plane: float,
        pixels: wp.array(dtype=float, ndim=4),
    ):
        env_id, cam_id, scan_line, point_index = wp.tid()
        mesh = mesh_ids[env_id]
        lidar_position = lidar_pos_array[env_id, cam_id]
        lidar_quaternion = lidar_quat_array[env_id, cam_id]
        ray_origin = lidar_position
        # perturb ray_vectors with uniform noise
        ray_dir = ray_vectors[scan_line, point_index]  # + sampled_vec3_noise
        ray_dir = wp.normalize(ray_dir)
        ray_direction_world = wp.normalize(wp.quat_rotate(lidar_quaternion, ray_dir))
        t = float(0.0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)
        n = wp.vec3()
        f = int(0)
        dist = NO_HIT_RAY_VAL
        if wp.mesh_query_ray(mesh, ray_origin, ray_direction_world, far_plane, t, u, v, sign, n, f):
            dist = t
        pixels[env_id, cam_id, scan_line, point_index] = dist
