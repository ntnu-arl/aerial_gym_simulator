import warp as wp

NO_HIT_RAY_VAL = wp.constant(1000.0)
NO_HIT_SEGMENTATION_VAL = wp.constant(wp.int32(-2))


class DepthCameraWarpKernels:
    def __init__(self):
        pass

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_pointcloud_segmentation(
        mesh_ids: wp.array(dtype=wp.uint64),
        cam_poss: wp.array(dtype=wp.vec3, ndim=2),
        cam_quats: wp.array(dtype=wp.quat, ndim=2),
        K_inv: wp.mat44,
        far_plane: float,
        pixels: wp.array(dtype=wp.vec3, ndim=4),
        segmentation_pixels: wp.array(dtype=wp.int32, ndim=4),
        c_x: int,
        c_y: int,
        pointcloud_in_world_frame: bool,
    ):

        env_id, cam_id, x, y = wp.tid()
        mesh = mesh_ids[env_id]
        cam_pos = cam_poss[env_id, cam_id]
        cam_quat = cam_quats[env_id, cam_id]
        cam_coords = wp.vec3(
            float(x), float(y), 1.0
        )  # this only converts the frame from warp's z-axis front to Isaac Gym's x-axis front
        cam_coords_principal = wp.vec3(
            float(c_x), float(c_y), 1.0
        )  # get the vector of principal axis
        # transform to uv [-1,1]
        uv = wp.normalize(wp.transform_vector(K_inv, cam_coords))
        uv_principal = wp.normalize(
            wp.transform_vector(K_inv, cam_coords_principal)
        )  # uv for principal axis
        # compute camera ray
        # cam origin in world space
        ro = cam_pos
        # tf the direction from camera to world space and normalize
        rd = wp.normalize(wp.quat_rotate(cam_quat, uv))
        rd_principal = wp.normalize(
            wp.quat_rotate(cam_quat, uv_principal)
        )  # ray direction of principal axis
        t = float(0.0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)
        n = wp.vec3()
        f = int(0)
        dist = NO_HIT_RAY_VAL
        segmentation_value = NO_HIT_SEGMENTATION_VAL
        if wp.mesh_query_ray(mesh, ro, rd, far_plane, t, u, v, sign, n, f):
            dist = t
            mesh_obj = wp.mesh_get(mesh)
            face_index = mesh_obj.indices[f * 3]
            segmentation_value = wp.int32(mesh_obj.velocities[face_index][0])
        if pointcloud_in_world_frame:
            pixels[env_id, cam_id, y, x] = ro + dist * rd
        else:
            pixels[env_id, cam_id, y, x] = dist * uv
        segmentation_pixels[env_id, cam_id, y, x] = segmentation_value

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_normal_faceID(
        mesh_ids: wp.array(dtype=wp.uint64),
        cam_poss: wp.array(dtype=wp.vec3, ndim=2),
        cam_quats: wp.array(dtype=wp.quat, ndim=2),
        K_inv: wp.mat44,
        far_plane: float,
        pixels: wp.array(dtype=wp.vec3, ndim=4),
        face_pixels: wp.array(dtype=wp.int32, ndim=4),
        c_x: int,
        c_y: int,
        normal_in_world_frame: bool,
    ):

        env_id, cam_id, x, y = wp.tid()
        mesh = mesh_ids[env_id]
        cam_pos = cam_poss[env_id, cam_id]
        cam_quat = cam_quats[env_id, cam_id]
        cam_coords = wp.vec3(
            float(x), float(y), 1.0
        )  # this only converts the frame from warp's z-axis front to Isaac Gym's x-axis front
        cam_coords_principal = wp.vec3(
            float(c_x), float(c_y), 1.0
        )  # get the vector of principal axis
        # transform to uv [-1,1]
        uv = wp.normalize(wp.transform_vector(K_inv, cam_coords))
        uv_principal = wp.normalize(
            wp.transform_vector(K_inv, cam_coords_principal)
        )  # uv for principal axis
        # compute camera ray
        # cam origin in world space
        ro = cam_pos
        # tf the direction from camera to world space and normalize
        rd = wp.normalize(wp.quat_rotate(cam_quat, uv))
        rd_principal = wp.normalize(wp.quat_rotate(cam_quat, uv_principal))
        t = float(0.0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)
        n = wp.vec3()
        f = int(-1)
        pixels[env_id, cam_id, y, x] = n * NO_HIT_RAY_VAL
        wp.mesh_query_ray(mesh, ro, rd, far_plane, t, u, v, sign, n, f)
        if normal_in_world_frame:
            pixels[env_id, cam_id, y, x] = n
        else:
            # transform the normal to camera frame where the x axis is defined as rd_principal
            pixels[env_id, cam_id, y, x] = wp.vec3(
                wp.dot(n, rd_principal),
                wp.dot(n, wp.cross(rd_principal, wp.vec3(0.0, 0.0, 1.0))),
                wp.dot(n, wp.cross(rd_principal, wp.vec3(0.0, 1.0, 0.0))),
            )
        face_pixels[env_id, cam_id, y, x] = f

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_pointcloud(
        mesh_ids: wp.array(dtype=wp.uint64),
        cam_poss: wp.array(dtype=wp.vec3, ndim=2),
        cam_quats: wp.array(dtype=wp.quat, ndim=2),
        K_inv: wp.mat44,
        far_plane: float,
        pixels: wp.array(dtype=wp.vec3, ndim=4),
        c_x: int,
        c_y: int,
        pointcloud_in_world_frame: bool,
    ):

        env_id, cam_id, x, y = wp.tid()
        mesh = mesh_ids[env_id]
        cam_pos = cam_poss[env_id, cam_id]
        cam_quat = cam_quats[env_id, cam_id]
        cam_coords = wp.vec3(
            float(x), float(y), 1.0
        )  # this only converts the frame from warp's z-axis front to Isaac Gym's x-axis front
        cam_coords_principal = wp.vec3(
            float(c_x), float(c_y), 1.0
        )  # get the vector of principal axis
        # transform to uv [-1,1]
        uv = wp.normalize(wp.transform_vector(K_inv, cam_coords))
        uv_principal = wp.normalize(
            wp.transform_vector(K_inv, cam_coords_principal)
        )  # uv for principal axis
        # compute camera ray
        # cam origin in world space
        ro = cam_pos
        # tf the direction from camera to world space and normalize
        rd = wp.normalize(wp.quat_rotate(cam_quat, uv))
        rd_principal = wp.normalize(
            wp.quat_rotate(cam_quat, uv_principal)
        )  # ray direction of principal axis
        t = float(0.0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)
        n = wp.vec3()
        f = int(0)
        dist = NO_HIT_RAY_VAL
        if wp.mesh_query_ray(mesh, ro, rd, far_plane, t, u, v, sign, n, f):
            dist = t
        if pointcloud_in_world_frame:
            pixels[env_id, cam_id, y, x] = ro + dist * rd
        else:
            pixels[env_id, cam_id, y, x] = dist * uv

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_depth_range(
        mesh_ids: wp.array(dtype=wp.uint64),
        cam_poss: wp.array(dtype=wp.vec3, ndim=2),
        cam_quats: wp.array(dtype=wp.quat, ndim=2),
        K_inv: wp.mat44,
        far_plane: float,
        pixels: wp.array(dtype=float, ndim=4),
        c_x: int,
        c_y: int,
        calculate_depth: bool,
    ):

        env_id, cam_id, x, y = wp.tid()
        mesh = mesh_ids[env_id]
        cam_pos = cam_poss[env_id, cam_id]
        cam_quat = cam_quats[env_id, cam_id]
        cam_coords = wp.vec3(
            float(x), float(y), 1.0
        )  # this only converts the frame from warp's z-axis front to Isaac Gym's x-axis front
        cam_coords_principal = wp.vec3(
            float(c_x), float(c_y), 1.0
        )  # get the vector of principal axis
        # transform to uv [-1,1]
        uv = wp.transform_vector(K_inv, cam_coords)
        uv_principal = wp.transform_vector(K_inv, cam_coords_principal)  # uv for principal axis
        # compute camera ray
        # cam origin in world space
        ro = cam_pos
        # tf the direction from camera to world space and normalize
        rd = wp.normalize(wp.quat_rotate(cam_quat, uv))
        rd_principal = wp.normalize(
            wp.quat_rotate(cam_quat, uv_principal)
        )  # ray direction of principal axis
        t = float(0.0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)
        n = wp.vec3()
        f = int(0)
        multiplier = 1.0
        if calculate_depth:
            multiplier = wp.dot(
                rd, rd_principal
            )  # multiplier to project each ray on principal axis for depth instead of range
        dist = NO_HIT_RAY_VAL
        if wp.mesh_query_ray(mesh, ro, rd, far_plane / multiplier, t, u, v, sign, n, f):
            dist = multiplier * t

        pixels[env_id, cam_id, y, x] = dist

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_depth_range_segmentation(
        mesh_ids: wp.array(dtype=wp.uint64),
        cam_poss: wp.array(dtype=wp.vec3, ndim=2),
        cam_quats: wp.array(dtype=wp.quat, ndim=2),
        K_inv: wp.mat44,
        far_plane: float,
        pixels: wp.array(dtype=float, ndim=4),
        segmentation_pixels: wp.array(dtype=wp.int32, ndim=4),
        c_x: int,
        c_y: int,
        calculate_depth: bool,
    ):

        env_id, cam_id, x, y = wp.tid()
        mesh = mesh_ids[env_id]
        cam_pos = cam_poss[env_id, cam_id]
        cam_quat = cam_quats[env_id, cam_id]
        cam_coords = wp.vec3(
            float(x), float(y), 1.0
        )  # this only converts the frame from warp's z-axis front to Isaac Gym's x-axis front
        cam_coords_principal = wp.vec3(
            float(c_x), float(c_y), 1.0
        )  # get the vector of principal axis
        # transform to uv [-1,1]
        uv = wp.transform_vector(K_inv, cam_coords)
        uv_principal = wp.transform_vector(K_inv, cam_coords_principal)  # uv for principal axis
        # compute camera ray
        # cam origin in world space
        ro = cam_pos
        # tf the direction from camera to world space and normalize
        rd = wp.normalize(wp.quat_rotate(cam_quat, uv))
        rd_principal = wp.normalize(
            wp.quat_rotate(cam_quat, uv_principal)
        )  # ray direction of principal axis
        t = float(0.0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)
        n = wp.vec3()
        f = int(0)
        multiplier = 1.0
        if calculate_depth:
            multiplier = wp.dot(
                rd, rd_principal
            )  # multiplier to project each ray on principal axis for depth instead of range
        dist = NO_HIT_RAY_VAL
        segmentation_value = NO_HIT_SEGMENTATION_VAL
        if wp.mesh_query_ray(mesh, ro, rd, far_plane / multiplier, t, u, v, sign, n, f):
            dist = multiplier * t
            mesh_obj = wp.mesh_get(mesh)
            face_index = mesh_obj.indices[f * 3]
            segmentation_value = wp.int32(mesh_obj.velocities[face_index][0])

        pixels[env_id, cam_id, y, x] = dist
        segmentation_pixels[env_id, cam_id, y, x] = segmentation_value
