import numpy as np

from aerial_gym import AERIAL_GYM_DIRECTORY

from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import (
    BaseDepthCameraConfig,
)
from aerial_gym.config.sensor_config.lidar_config.base_lidar_config import (
    BaseLidarConfig,
)
from aerial_gym.config.sensor_config.imu_config.base_imu_config import BaseImuConfig


class BaseRandCfg:

    class init_config:
        # init_state tensor is of the format [ratio_x, ratio_y, ratio_z, roll_radians, pitch_radians, yaw_radians, 1.0 (for maintaining shape), vx, vy, vz, wx, wy, wz]
        min_init_state = [
            0.0,
            0.0,
            0.0,
            0,  # -np.pi / 6,
            0,  # -np.pi / 6,
            -np.pi,
            1.0,
            -0.2,
            -0.2,
            -0.2,
            -0.2,
            -0.2,
            -0.2,
        ]
        max_init_state = [
            1.0,
            1.0,
            1.0,
            0,  # np.pi / 6,
            0,  # np.pi / 6,
            np.pi,
            1.0,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
        ]

    class sensor_config:
        enable_camera = False
        camera_config = BaseDepthCameraConfig

        enable_lidar = False
        lidar_config = BaseLidarConfig

        enable_imu = False
        imu_config = BaseImuConfig

    class disturbance:
        enable_disturbance = True
        prob_apply_disturbance = 0.05
        max_force_and_torque_disturbance = [1.5, 1.5, 1.5, 0.25, 0.25, 0.25]

    class damping:
        linvel_linear_damping_coefficient = [0.0, 0.0, 0.0]  # along the body [x, y, z] axes
        linvel_quadratic_damping_coefficient = [0.0, 0.0, 0.0]  # along the body [x, y, z] axes
        angular_linear_damping_coefficient = [0.0, 0.0, 0.0]  # along the body [x, y, z] axes
        angular_quadratic_damping_coefficient = [0.0, 0.0, 0.0]  # along the body [x, y, z] axes

    class robot_asset:
        asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/robots/random"
        file = "random.urdf"
        name = "base_random"  # actor name
        base_link_name = "base_link"
        disable_gravity = False
        collapse_fixed_joints = False  # merge bodies connected by fixed joints.
        fix_base_link = False  # fix the base of the robot
        collision_mask = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True  # Some .obj meshes must be flipped from y-up to z-up
        density = 0.000001
        angular_damping = 0.0000001
        linear_damping = 0.0000001
        max_angular_velocity = 100.0
        max_linear_velocity = 100.0
        armature = 0.001

        semantic_id = 0
        per_link_semantic = False

        min_state_ratio = [
            0.1,
            0.1,
            0.1,
            0,
            0,
            -np.pi,
            1.0,
            -0.5,
            -0.5,
            -0.5,
            -0.2,
            -0.2,
            -0.2,
        ]  # [ratio_x, ratio_y, ratio_z, roll_rad, pitch_rad, yaw_rad, 1.0, vx, vy, vz, wx, wy, wz]
        max_state_ratio = [
            0.3,
            0.9,
            0.9,
            0,
            0,
            np.pi,
            1.0,
            0.5,
            0.5,
            0.5,
            0.2,
            0.2,
            0.2,
        ]  # [ratio_x, ratio_y, ratio_z, roll_rad, pitch_rad, yaw_rad, 1.0, vx, vy, vz, wx, wy, wz]

        max_force_and_torque_disturbance = [
            0.1,
            0.1,
            0.1,
            0.05,
            0.05,
            0.05,
        ]  # [fx, fy, fz, tx, ty, tz]

        color = None
        semantic_masked_links = {}
        keep_in_env = True  # this does nothing for the robot

        min_position_ratio = None
        max_position_ratio = None

        min_euler_angles = [-np.pi, -np.pi, -np.pi]
        max_euler_angles = [np.pi, np.pi, np.pi]

        place_force_sensor = True  # set this to True if IMU is desired
        force_sensor_parent_link = "base_link"
        force_sensor_transform = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]  # [x, y, z, qx, qy, qz, qw]

        use_collision_mesh_instead_of_visual = False  # does nothing for the robot

    class control_allocator_config:
        num_motors = 8
        force_application_level = "motor_link"  # "motor_link" or "root_link" decides to apply combined forces acting on the robot at the root link or at the individual motor links

        application_mask = [1 + 8 + i for i in range(0, 8)]
        motor_directions = [-1, 1, -1, 1, -1, 1, -1, 1]

        allocation_matrix = [
            [
                5.55111512e-17,
                -3.21393805e-01,
                -4.54519478e-01,
                -3.42020143e-01,
                9.69846310e-01,
                3.42020143e-01,
                8.66025404e-01,
                -7.54406507e-01,
            ],
            [
                1.00000000e00,
                -3.42020143e-01,
                -7.07106781e-01,
                0.00000000e00,
                -1.73648178e-01,
                9.39692621e-01,
                5.00000000e-01,
                -1.73648178e-01,
            ],
            [
                1.66533454e-16,
                -8.83022222e-01,
                5.41675220e-01,
                9.39692621e-01,
                1.71010072e-01,
                1.11022302e-16,
                1.11022302e-16,
                6.33022222e-01,
            ],
            [
                1.75000000e-01,
                1.23788742e-01,
                -5.69783368e-02,
                1.34977168e-01,
                3.36959042e-02,
                -2.66534135e-01,
                -7.88397460e-02,
                -2.06893989e-02,
            ],
            [
                1.00000000e-02,
                2.78845133e-01,
                -4.32852308e-02,
                -2.72061766e-01,
                -1.97793856e-01,
                8.63687139e-02,
                1.56554446e-01,
                -1.71261290e-01,
            ],
            [
                2.82487373e-01,
                -1.41735490e-01,
                -8.58541103e-02,
                3.84858939e-02,
                -3.33468026e-01,
                8.36741468e-02,
                8.46777988e-03,
                -8.74336259e-02,
            ],
        ]

        class motor_model_config:
            use_rps = False
            motor_thrust_constant_min = 0.00000926312
            motor_thrust_constant_max = 0.00001826312
            motor_time_constant_increasing_min = 0.01
            motor_time_constant_increasing_max = 0.03
            motor_time_constant_decreasing_min = 0.005
            motor_time_constant_decreasing_max = 0.005
            max_thrust = 5.0
            min_thrust = -5.0
            max_thrust_rate = 100000.0
            thrust_to_torque_ratio = (
                0.01  # thrust to torque ratio is related to inertia matrix dont change
            )
            use_discrete_approximation = (
                True  # Setting to false will compute f' based on difference and time constant
            )
