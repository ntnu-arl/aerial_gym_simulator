# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from isaacgym.torch_utils import tensor_clamp

class TensorPID:
    def __init__(self, num_envs, num_dims, Kp, Kd, Ki, dt, integral_min_limit, integral_max_limit, derivative_saturation_min_limit, derivative_saturation_max_limit, output_min_limit, output_max_limit, device = torch.device("cuda")):
        self.device = device
        self.Kp = torch.tensor(Kp, device = self.device)
        self.Kd = torch.tensor(Kd, device = self.device)
        self.Ki = torch.tensor(Ki, device = self.device)
        self.dt = dt
        self.integral_min_limit = torch.tensor(integral_min_limit, device = self.device)
        self.integral_max_limit = torch.tensor(integral_max_limit, device = self.device)
        self.derivative_saturation_min_limit = torch.tensor(derivative_saturation_min_limit, device = self.device)
        self.derivative_saturation_max_limit = torch.tensor(derivative_saturation_max_limit, device = self.device)
        self.output_min_limit = torch.tensor(output_min_limit, device = self.device)
        self.output_max_limit = torch.tensor(output_max_limit, device = self.device)
        self.integral = torch.zeros((num_envs, num_dims), device = self.device)
        self.prev_error = torch.zeros((num_envs, num_dims), device = self.device)
        self.reset_state = torch.ones((num_envs, num_dims), device = self.device)
    
    def update(self, error):
        self.integral += error * self.dt
        # calculate PID terms
        proportional_term = self.Kp * error
        derivative_term = self.Kd * (1 - self.reset_state)*(error - self.prev_error) / self.dt
        integral_term = self.Ki * self.integral
        # clamp the integral to avoid numerical instability
        integral_term = tensor_clamp(integral_term, self.integral_min_limit, self.integral_max_limit)
        derivative_term = tensor_clamp(derivative_term, self.derivative_saturation_min_limit, self.derivative_saturation_max_limit)
        # calculate PID output
        output = proportional_term + derivative_term + integral_term
        # clamp the output to avoid numerical instability
        output = tensor_clamp(output, self.output_min_limit, self.output_max_limit)
        self.prev_error = error
        self.reset_state[:,:] = 0.0
        return output
    
    def reset(self):
        self.integral[:, :] = 0
        self.prev_error[:,:] = 0
        self.reset_state[:,:] = 1.0
    
    def reset_idx(self, env_idx):
        self.integral[env_idx, :] = 0
        self.prev_error[env_idx, :] = 0.0
        self.reset_state[env_idx, :] = 1.0
