import numpy as np

from aerial_gym import AERIAL_GYM_DIRECTORY

from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import (
    BaseDepthCameraConfig,
)
from aerial_gym.config.sensor_config.lidar_config.base_lidar_config import (
    BaseLidarConfig,
)
from aerial_gym.config.sensor_config.imu_config.base_imu_config import BaseImuConfig


class BaseROVCfg:

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

    class damping:
        linvel_linear_damping_coefficient = [0.0, 0.0, 0.0]  # along the body [x, y, z] axes
        linvel_quadratic_damping_coefficient = [0.0, 0.0, 0.0]  # along the body [x, y, z] axes
        angular_linear_damping_coefficient = [0.0, 0.0, 0.0]  # along the body [x, y, z] axes
        angular_quadratic_damping_coefficient = [0.0, 0.0, 0.0]  # along the body [x, y, z] axes

    class disturbance:
        enable_disturbance = True
        prob_apply_disturbance = 0.05
        max_force_and_torque_disturbance = [1.5, 1.5, 1.5, 0.25, 0.25, 0.25]

    class robot_asset:
        asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/robots/BlueROV"
        file = "rov.urdf"
        name = "base_rov"  # actor name
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
            0,
            0,
            0,
            0,
            0,
            0,
        ]  # [ratio_x, ratio_y, ratio_z, roll_rad, pitch_rad, yaw_rad, 1.0, vx, vy, vz, wx, wy, wz]
        max_state_ratio = [
            0.3,
            0.9,
            0.9,
            0,
            0,
            np.pi,
            1.0,
            0,
            0,
            0,
            0,
            0,
            0,
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
        motor_directions = [1, -1, 1, -1, 1, -1, 1, -1]

        allocation_matrix = [
            [
                -0.78867513,
                0.21132487,
                -0.21132487,
                0.78867513,
                0.78867513,
                -0.21132487,
                0.21132487,
                -0.78867513,
            ],
            [
                0.21132487,
                0.78867513,
                -0.78867513,
                -0.21132487,
                -0.21132487,
                -0.78867513,
                0.78867513,
                0.21132487,
            ],
            [
                0.57735027,
                -0.57735027,
                -0.57735027,
                0.57735027,
                0.57735027,
                -0.57735027,
                -0.57735027,
                0.57735027,
            ],
            [
                0.14226497,
                -0.21547005,
                0.25773503,
                0.01547005,
                -0.01547005,
                -0.25773503,
                0.21547005,
                -0.14226497,
            ],
            [
                -0.25773503,
                0.01547005,
                0.14226497,
                0.21547005,
                -0.21547005,
                -0.14226497,
                -0.01547005,
                0.25773503,
            ],
            [
                0.11547005,
                -0.23094011,
                -0.11547005,
                0.23094011,
                -0.23094011,
                0.11547005,
                0.23094011,
                -0.11547005,
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
            max_thrust = 6.25
            min_thrust = -6.25
            max_thrust_rate = 100000.0
            thrust_to_torque_ratio = (
                0.01  # thrust to torque ratio is related to inertia matrix dont change
            )
            use_discrete_approximation = (
                True  # Setting to false will compute f' based on difference and time constant
            )
