from aerial_gym.config.sensor_config.imu_config.base_imu_config import BaseImuConfig


class VN100Config(BaseImuConfig):
    num_sensors = 1  # number of sensors of this type. More than 1 sensor on the same link don't make sense. Can be implemented if needed for multiple different links.

    sensor_type = "imu"  # sensor type

    world_frame = False

    # enable or disable noise and bias. Setting to False will simulate a perfect, noise- and bias-free IMU
    enable_noise = True
    enable_bias = True

    # bias and noise values as obtained from a sample VN100 IMU
    bias_std = [
        9.782812831313576e-07,
        9.782812831313576e-07,
        9.782812831313576e-07,
        2.6541629581345176e-05,
        2.6541629581345176e-05,
        2.6541629581345176e-05,
    ]  # first 3 values for acc bias std, next 3 for gyro bias std
    imu_noise_std = [
        0.001372931,
        0.001372931,
        0.001372931,
        6.108652381980153e-05,
        6.108652381980153e-05,
        6.108652381980153e-05,
    ]  # first 3 vaues for acc noise std, next 3 for gyro noise std
    max_measurement_value = [
        100.0,
        100.0,
        100.0,
        10.0,
        10.0,
        10.0,
    ]  # max measurement value for acc and gyro outputs will be clamped by + & - of these

    max_bias_init_value = [
        1.0e-03,
        1.0e-03,
        1.0e-03,
        1.0e-03,
        1.0e-03,
        1.0e-03,
    ]  # max bias init value for acc and gyro biases will be sampled within +/- of this range

    # Setting this to true will provide acceelration of the object in a static frame w.r.t ground.

    gravity_compensation = False  # usually the force sensor computes total force including gravity, so set this to False

    # The position of this is hardcoded at the center of the asset. This can be changed by the user in code if needed.

    # Randomize the orientation of the sensor w.r.t the parent link. The position is still [0,0,0] in the parent link frame
    randomize_placement = True
    min_euler_rotation_deg = [-2.0, -2.0, -2.0]
    max_euler_rotation_deg = [2.0, 2.0, 2.0]
