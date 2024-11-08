class SimCfg2Ms:
    # viewer camera:
    class viewer:
        headless = False
        ref_env = 0
        camera_position = [-5, -5, 4]  # [m]
        lookat = [0, 0, 0]  # [m]
        camera_orientation_euler_deg = [0, 0, 0]  # [deg]
        camera_follow_type = "FOLLOW_TRANSFORM"
        width = 1280
        height = 720
        max_range = 100.0  # [m]
        min_range = 0.1
        horizontal_fov_deg = 90
        use_collision_geometry = False
        camera_follow_transform_local_offset = [-1.0, 0.0, 0.2]  # m
        camera_follow_position_global_offset = [-1.0, 0.0, 0.4]  # m

    class sim:
        dt = 0.002
        substeps = 1
        gravity = [0.0, 0.0, -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z
        use_gpu_pipeline = True

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 2
            contact_offset = 0.002  # [m]
            rest_offset = 0.001  # [m]
            bounce_threshold_velocity = 0.1  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**24  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 10
            contact_collection = 1  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
