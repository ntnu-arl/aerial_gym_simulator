import numpy as np


class control:
    """
    Control parameters
    controller:
        lee_position_control: command_actions = [x, y, z, yaw] in environment frame scaled between -1 and 1
        lee_velocity_control: command_actions = [vx, vy, vz, yaw_rate] in vehicle frame scaled between -1 and 1
        lee_attitude_control: command_actions = [thrust, roll, pitch, yaw_rate] in vehicle frame scaled between -1 and 1
    kP: gains for position
    kV: gains for velocity
    kR: gains for attitude
    kOmega: gains for angular velocity
    """

    num_actions = 4
    max_inclination_angle_rad = np.pi / 3.0
    max_yaw_rate = np.pi / 3.0

    K_pos_tensor_max = [2.0, 2.0, 1.0]  # used for lee_position_control only
    K_pos_tensor_min = [2.0, 2.0, 1.0]  # used for lee_position_control only

    K_vel_tensor_max = [
        3.3,
        3.3,
        1.3,
    ]  # used for lee_position_control, lee_velocity_control only
    K_vel_tensor_min = [2.7, 2.7, 1.7]

    K_rot_tensor_max = [
        1.85,
        1.85,
        0.4,
    ]  # used for lee_position_control, lee_velocity_control and lee_attitude_control
    K_rot_tensor_min = [1.6, 1.6, 0.25]

    K_angvel_tensor_max = [
        0.5,
        0.5,
        0.09,
    ]  # used for lee_position_control, lee_velocity_control and lee_attitude_control
    K_angvel_tensor_min = [0.4, 0.4, 0.075]

    randomize_params = True
