from aerial_gym.robots.base_reconfigurable import BaseReconfigurable
from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.math import pd_control

logger = CustomLogger("morphy_robot_class")
import torch


class Morphy(BaseReconfigurable):
    def __init__(self, robot_config, controller_name, env_config, device):
        super().__init__(
            robot_config=robot_config,
            controller_name=controller_name,
            env_config=env_config,
            device=device,
        )

    def init_joint_response_params(self, cfg):
        self.joint_stiffness = torch.tensor(
            cfg.reconfiguration_config.stiffness, device=self.device, dtype=torch.float32
        ).expand(self.num_envs, -1)
        self.joint_damping = torch.tensor(
            cfg.reconfiguration_config.damping, device=self.device, dtype=torch.float32
        ).expand(self.num_envs, -1)
        self.stiffness_param = cfg.reconfiguration_config.custom_nonlinear_stiffness
        self.damping_param = cfg.reconfiguration_config.custom_linear_damping

    def call_arm_controller(self):
        """
        Call the controller for the arm of the quadrotor. This function is called every simulation step.
        """
        if self.dof_control_mode == "effort":
            self.dof_effort_tensor[:] = (
                0.01625
                * (0.07 * 0.07)
                * arm_response_func(
                    (self.dof_states_position[:] - 7.2 * torch.pi / 180.0),
                    self.dof_states_velocity[:],
                    self.stiffness_param,
                    self.damping_param,
                )
            )
            self.dof_effort_tensor[:] -= (
                9.81 * 0.01625 * 0.07 * torch.cos(self.dof_states_position[:])
            )
        else:
            return


@torch.jit.script
def arm_response_func(pos_error, vel_error, lin_damper, nonlin_spring):
    # type: (Tensor, Tensor, float, float) -> Tensor
    return lin_damper * vel_error + nonlin_spring * torch.sign(pos_error) * pos_error**2
