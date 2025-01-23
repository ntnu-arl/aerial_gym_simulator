import warp as wp

INVALID_PIXEL_VAL = wp.constant(-1.0)
NO_HIT_RAY_VAL = wp.constant(1000.0)
NO_HIT_SEGMENTATION_VAL = wp.constant(wp.int32(-2))

class StereoCameraWarpKernels:
    def __init__(self):
        pass

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_pointcloud_segmentation(
        mesh_ids: wp.array(dtype=wp.uint64),
        cam_poss: wp.array(dtype=wp.vec3, ndim=2),
        cam_quats: wp.array(dtype=wp.quat, ndim=2),
        baseline: float,
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
        baseline_offser = wp.vec3(
            -baseline, 0.0, 0.0
        )
        second_cam_pos = cam_pos + wp.quat_rotate(cam_quat, baseline_offser)

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
        t2 = float(0.0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)
        n = wp.vec3()
        f = int(0)
        dist = INVALID_PIXEL_VAL
        segmentation_value = NO_HIT_SEGMENTATION_VAL
        if wp.mesh_query_ray(mesh, ro, rd, far_plane, t, u, v, sign, n, f):
            endpoint = ro + rd * t*0.999
            distance_to_second_cam = wp.length(second_cam_pos - endpoint)
            rd_reverse = wp.normalize(second_cam_pos - endpoint)
            if not wp.mesh_query_ray(mesh, endpoint, rd_reverse, distance_to_second_cam, t2, u, v, sign, n, f):
                dist = t
                mesh_obj = wp.mesh_get(mesh)
                face_index = mesh_obj.indices[f * 3]
                segmentation_value = wp.int32(mesh_obj.velocities[face_index][0])
        else:
            # this means the ray is not hitting any object
            # check if projecting back from max range to stereo pair hits anything
            endpoint = ro + rd * far_plane
            distance_to_second_cam = wp.length(second_cam_pos - endpoint)
            rd_reverse = wp.normalize(second_cam_pos - endpoint)
            # start ray from the point which it reached the far plane
            if not wp.mesh_query_ray(mesh, ro + rd * far_plane, rd_reverse, distance_to_second_cam, t2, u, v, sign, n, f):
                dist = NO_HIT_RAY_VAL
        
        if pointcloud_in_world_frame:
            pixels[env_id, cam_id, y, x] = ro + dist * rd
        else:
            pixels[env_id, cam_id, y, x] = dist * uv
        segmentation_pixels[env_id, cam_id, y, x] = segmentation_value

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_pointcloud(
        mesh_ids: wp.array(dtype=wp.uint64),
        cam_poss: wp.array(dtype=wp.vec3, ndim=2),
        cam_quats: wp.array(dtype=wp.quat, ndim=2),
        baseline: float,
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
        offset = wp.vec3(-baseline, 0.0, 0.0)
        second_cam_pos = cam_pos + wp.quat_rotate(cam_quat, offset)
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
        dist = INVALID_PIXEL_VAL
        if wp.mesh_query_ray(mesh, ro, rd, far_plane, t, u, v, sign, n, f):
            endpoint = ro + rd * t*0.999
            distance_to_second_cam = wp.length(second_cam_pos - endpoint)
            rd_reverse = wp.normalize(second_cam_pos - endpoint)
            if not wp.mesh_query_ray(mesh, endpoint, rd_reverse, distance_to_second_cam, t, u, v, sign, n, f):
                dist = t
        else:
            # this means the ray is not hitting any object
            # check if projecting back from max range to stereo pair hits anything
            endpoint = ro + rd * far_plane
            distance_to_second_cam = wp.length(second_cam_pos - endpoint)
            rd_reverse = wp.normalize(second_cam_pos - endpoint)
            # start ray from the point which it reached the far plane
            if not wp.mesh_query_ray(mesh, ro + rd * far_plane, rd_reverse, distance_to_second_cam, t, u, v, sign, n, f):
                dist = NO_HIT_RAY_VAL
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
        baseline: float,
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
        offset = wp.vec3(-baseline, 0.0, 0.0)
        stereo_cam_pos = cam_pos + wp.quat_rotate(cam_quat, offset)
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
        t2 = float(0.0)
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
        dist = INVALID_PIXEL_VAL
        if wp.mesh_query_ray(mesh, ro, rd, far_plane / multiplier, t, u, v, sign, n, f):
            endpoint = ro + rd * t*0.999
            distance_to_stereo = wp.length(stereo_cam_pos - endpoint)
            rd_reverse = wp.normalize(stereo_cam_pos - endpoint)
            if not wp.mesh_query_ray(mesh, endpoint, rd_reverse, distance_to_stereo, t2, u, v, sign, n, f):
                dist = t*multiplier
        else:
            # this means the ray is not hitting any object
            # check if projecting back from max range to stereo pair hits anything
            endpoint = ro + rd * far_plane / multiplier
            distance_to_stereo = wp.length(stereo_cam_pos - endpoint)
            rd_reverse = wp.normalize(stereo_cam_pos - endpoint)
            # start ray from the point which it reached the far plane
            if not wp.mesh_query_ray(mesh, ro + rd * far_plane / multiplier, rd_reverse, distance_to_stereo, t2, u, v, sign, n, f):
                dist = NO_HIT_RAY_VAL
        
        pixels[env_id, cam_id, y, x] = dist

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_depth_range_segmentation(
        mesh_ids: wp.array(dtype=wp.uint64),
        cam_poss: wp.array(dtype=wp.vec3, ndim=2),
        cam_quats: wp.array(dtype=wp.quat, ndim=2),
        baseline: float,
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
        offset = wp.vec3(-baseline, 0.0, 0.0)
        stereo_cam_pos = cam_pos + wp.quat_rotate(cam_quat, offset)
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
        t2 = float(0.0)
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
        dist = INVALID_PIXEL_VAL
        segmentation_value = NO_HIT_SEGMENTATION_VAL
        
        if wp.mesh_query_ray(mesh, ro, rd, far_plane / multiplier, t, u, v, sign, n, f):
            endpoint = ro + rd * t*0.999
            distance_to_stereo = wp.length(stereo_cam_pos - endpoint)
            rd_reverse = wp.normalize(stereo_cam_pos - endpoint)
            if not wp.mesh_query_ray(mesh, endpoint, rd_reverse, distance_to_stereo, t2, u, v, sign, n, f):
                dist = t*multiplier
                mesh_obj = wp.mesh_get(mesh)
                face_index = mesh_obj.indices[f * 3]
                segmentation_value = wp.int32(mesh_obj.velocities[face_index][0])
        else:
            # this means the ray is not hitting any object
            # check if projecting back from max range to stereo pair hits anything
            endpoint = ro + rd * far_plane / multiplier
            distance_to_stereo = wp.length(stereo_cam_pos - endpoint)
            rd_reverse = wp.normalize(stereo_cam_pos - endpoint)
            # start ray from the point which it reached the far plane
            if not wp.mesh_query_ray(mesh, ro + rd * far_plane / multiplier, rd_reverse, distance_to_stereo, t2, u, v, sign, n, f):
                dist = NO_HIT_RAY_VAL

        pixels[env_id, cam_id, y, x] = dist
        segmentation_pixels[env_id, cam_id, y, x] = segmentation_value
