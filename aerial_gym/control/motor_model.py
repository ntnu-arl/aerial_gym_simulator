import torch

from aerial_gym.utils.math import torch_rand_float_tensor


class MotorModel:
    def __init__(self, num_envs, motors_per_robot, dt, config, device="cuda:0"):
        self.num_envs = num_envs
        self.dt = dt
        self.cfg = config
        self.device = device
        self.num_motors_per_robot = motors_per_robot
        self.max_thrust = self.cfg.max_thrust
        self.min_thrust = self.cfg.min_thrust
        self.motor_time_constant_min = torch.tensor(
            self.cfg.motor_time_constant_min, device=self.device
        ).expand(self.num_envs, self.num_motors_per_robot)
        self.motor_time_constant_max = torch.tensor(
            self.cfg.motor_time_constant_max, device=self.device
        ).expand(self.num_envs, self.num_motors_per_robot)
        self.init_tensors()

    def init_tensors(self, global_tensor_dict=None):
        self.current_motor_thrust = torch.zeros(
            (self.num_envs, self.num_motors_per_robot), device=self.device
        )
        self.motor_time_constants = torch_rand_float_tensor(
            self.motor_time_constant_min, self.motor_time_constant_max
        )
        self.motor_thrust_rate = torch.zeros(
            (self.num_envs, self.num_motors_per_robot), device=self.device
        )

    def update_motor_thrusts(self, ref_thrust):
        ref_thrust = torch.clamp(ref_thrust, self.min_thrust, self.max_thrust)
        self.motor_thrust_rate[:] = (1.0 / self.motor_time_constants) * (
            ref_thrust - self.current_motor_thrust
        )
        self.motor_thrust_rate[:] = torch.clamp(
            self.motor_thrust_rate, -self.cfg.max_thrust_rate, self.cfg.max_thrust_rate
        )
        self.current_motor_thrust[:] = self.current_motor_thrust + self.dt * self.motor_thrust_rate
        return self.current_motor_thrust

    def reset_idx(self, env_ids):
        self.motor_time_constants[env_ids] = torch_rand_float_tensor(
            self.motor_time_constant_min, self.motor_time_constant_max
        )[env_ids]
        self.current_motor_thrust[env_ids] = torch_rand_float_tensor(
            self.min_thrust, self.max_thrust
        )[env_ids]

    def reset(self):
        self.motor_time_constants[:] = torch_rand_float_tensor(
            self.motor_time_constant_min, self.motor_time_constant_max
        )
        self.current_motor_thrust[:] = torch_rand_float_tensor(self.min_thrust, self.max_thrust)
