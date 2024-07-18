from aerial_gym.sensors.base_sensor import BaseSensor
import torch
from aerial_gym.utils.math import (
    quat_from_euler_xyz,
    tensor_clamp,
    quat_rotate_inverse,
    quat_mul,
    torch_rand_float_tensor,
    quat_from_euler_xyz_tensor,
)


class IMUSensor(BaseSensor):
    def __init__(self, sensor_config, num_envs, device):
        super().__init__(sensor_config=sensor_config, num_envs=num_envs, device=device)
        self.world_frame = self.cfg.world_frame

        self.gravity_compensation = self.cfg.gravity_compensation

    def init_tensors(self, global_tensor_dict=None):
        self.global_tensor_dict = global_tensor_dict
        super().init_tensors(self.global_tensor_dict)

        self.force_sensor_tensor = self.global_tensor_dict["force_sensor_tensor"]

        # initialize the tensors for maximum values and randomized sampled values
        # first 3 values for acc bias/noise, next 3 for gyro bias/noise
        self.bias_std = torch.tensor(
            self.cfg.bias_std,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        ).expand(self.num_envs, -1)
        self.imu_noise_std = torch.tensor(
            self.cfg.imu_noise_std, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)
        self.max_measurement_value = torch.tensor(
            self.cfg.max_measurement_value, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)
        self.max_bias_init_value = torch.tensor(
            self.cfg.max_bias_init_value, device=self.device, requires_grad=False
        )

        self.accel_t = torch.zeros((self.num_envs, 3), device=self.device, requires_grad=False)

        # Nominal sensor orientation value that can be perturbed
        self.min_sensor_euler_rotation_rad = torch.deg2rad(
            torch.tensor(self.cfg.min_euler_rotation_deg, device=self.device, requires_grad=False)
        ).expand(self.num_envs, -1)
        self.max_sensor_euler_rotation_rad = torch.deg2rad(
            torch.tensor(self.cfg.max_euler_rotation_deg, device=self.device, requires_grad=False)
        ).expand(self.num_envs, -1)

        self.sensor_quats = quat_from_euler_xyz_tensor(
            torch_rand_float_tensor(
                self.min_sensor_euler_rotation_rad, self.max_sensor_euler_rotation_rad
            )
        )

        # gravity value initialization
        self.g_world = self.gravity * (
            1 - int(self.gravity_compensation)
        )  # if gravity compensation is enabled, then gravity is subtracted from the acceleration

        # tensors and variables related to noise and bias
        self.enable_noise = int(self.cfg.enable_noise)
        self.enable_bias = int(self.cfg.enable_bias)
        self.bias = torch.zeros((self.num_envs, 6), device=self.device, requires_grad=False)
        self.noise = torch.zeros((self.num_envs, 6), device=self.device, requires_grad=False)
        self.imu_meas = torch.zeros((self.num_envs, 6), device=self.device, requires_grad=False)

        self.global_tensor_dict["imu_measurement"] = self.imu_meas

    def sample_noise(self):
        self.noise = (
            torch.randn((self.num_envs, 6), device=self.device) * self.imu_noise_std / self.sqrt_dt
        )

    def update_bias(self):
        self.bias_update_step = (
            torch.randn((self.num_envs, 6), device=self.device) * self.bias_std * self.sqrt_dt
        )
        self.bias += self.bias_update_step  # check if this is correct

    def update(self):
        """
        world_frame: if accel_t and ang_rate_t are in world frame or not
        """
        self.accel_t = self.force_sensor_tensor[:, 0:3] / self.robot_masses.unsqueeze(1)
        if self.world_frame:
            acceleration = quat_rotate_inverse(
                quat_mul(self.robot_orientation, self.sensor_quats),
                (self.accel_t - self.g_world),
            )
            ang_rate = quat_rotate_inverse(
                quat_mul(self.robot_orientation, self.sensor_quats),
                self.robot_body_angvel,
            )
        else:
            # Rotate the acceleration and angular rate from true sensor frame to perturbed sensor frame
            acceleration = quat_rotate_inverse(
                self.sensor_quats, self.accel_t
            ) - quat_rotate_inverse(
                quat_mul(self.robot_orientation, self.sensor_quats), self.g_world
            )
            ang_rate = quat_rotate_inverse(self.sensor_quats, self.robot_body_angvel)

        self.sample_noise()
        self.update_bias()
        accel_meas = (
            acceleration
            + self.enable_bias * self.bias[:, :3]
            + self.enable_noise * self.noise[:, :3]
        )
        ang_rate_meas = (
            ang_rate + self.enable_bias * self.bias[:, 3:] + self.enable_noise * self.noise[:, 3:]
        )
        # clamp the measurements from accelerometer and gyro to max values
        accel_meas = tensor_clamp(
            accel_meas,
            -self.max_measurement_value[:, 0:3],
            self.max_measurement_value[:, 0:3],
        )
        ang_rate_meas = tensor_clamp(
            ang_rate_meas,
            -self.max_measurement_value[:, 3:],
            self.max_measurement_value[:, 3:],
        )
        self.imu_meas[:, :3] = accel_meas
        self.imu_meas[:, 3:] = ang_rate_meas
        return

    def reset(self):
        self.bias.zero_()
        self.bias[:] = self.max_bias_init_value * (2.0 * (torch.rand_like(self.bias) - 0.5))
        self.sensor_quats[:] = quat_from_euler_xyz_tensor(
            torch_rand_float_tensor(
                self.min_sensor_euler_rotation_rad, self.max_sensor_euler_rotation_rad
            )
        )

    def reset_idx(self, env_ids):
        self.bias[env_ids, :] = (
            self.max_bias_init_value * (2.0 * (torch.rand_like(self.bias) - 0.5))
        )[env_ids, :]
        self.sensor_quats[env_ids] = quat_from_euler_xyz_tensor(
            torch_rand_float_tensor(
                self.min_sensor_euler_rotation_rad, self.max_sensor_euler_rotation_rad
            )
        )[env_ids]

    def get_observation(self):
        pass
