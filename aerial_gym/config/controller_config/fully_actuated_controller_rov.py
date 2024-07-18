import numpy as np


class control:
    """
    Controller configuration for the fully-actuated controller
    """

    num_actions = 7
    max_inclination_angle_rad = np.pi / 3.0
    max_yaw_rate = np.pi / 3.0

    K_pos_tensor_max = [1.0, 1.0, 1.0]  # used for lee_position_control only
    K_pos_tensor_min = [1.0, 1.0, 1.0]  # used for lee_position_control only

    K_vel_tensor_max = [
        8.0,
        8.0,
        8.0,
    ]  # used for lee_position_control, lee_velocity_control only
    K_vel_tensor_min = [8.0, 8.0, 8.0]

    K_rot_tensor_max = [
        2.2,
        2.2,
        2.6,
    ]  # used for lee_position_control, lee_velocity_control and lee_attitude_control
    K_rot_tensor_min = [2.2, 2.2, 2.6]

    K_angvel_tensor_max = [
        2.2,
        2.2,
        2.2,
    ]  # used for lee_position_control, lee_velocity_control and lee_attitude_control
    K_angvel_tensor_min = [2.1, 2.1, 2.1]

    randomize_params = True
