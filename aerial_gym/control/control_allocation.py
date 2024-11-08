import torch
from aerial_gym.control.motor_model import MotorModel

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("control_allocation")


class ControlAllocator:
    def __init__(self, num_envs, dt, config, device):
        self.num_envs = num_envs
        self.dt = dt
        self.cfg = config
        self.device = device
        self.force_application_level = self.cfg.force_application_level
        self.motor_directions = torch.tensor(self.cfg.motor_directions, device=self.device)
        self.force_torque_allocation_matrix = torch.tensor(
            self.cfg.allocation_matrix, device=self.device, dtype=torch.float32
        )
        self.inv_force_torque_allocation_matrix = torch.linalg.pinv(
            self.force_torque_allocation_matrix
        )
        self.output_wrench = torch.zeros((self.num_envs, 6), device=self.device)

        assert (
            len(self.cfg.allocation_matrix[0]) == self.cfg.num_motors
        ), "Allocation matrix must have 6 rows and num_motors columns."

        self.force_torque_allocation_matrix = torch.tensor(
            self.cfg.allocation_matrix, device=self.device, dtype=torch.float32
        )
        alloc_matrix_rank = torch.linalg.matrix_rank(self.force_torque_allocation_matrix)
        if alloc_matrix_rank < 6:
            print("WARNING: allocation matrix is not full rank. Rank: {}".format(alloc_matrix_rank))
        self.force_torque_allocation_matrix = self.force_torque_allocation_matrix.expand(
            self.num_envs, -1, -1
        )
        self.inv_force_torque_allocation_matrix = torch.linalg.pinv(
            torch.tensor(self.cfg.allocation_matrix, device=self.device, dtype=torch.float32)
        ).expand(self.num_envs, -1, -1)
        self.motor_model = MotorModel(
            num_envs=self.num_envs,
            dt=self.dt,
            motors_per_robot=self.cfg.num_motors,
            config=self.cfg.motor_model_config,
            device=self.device,
        )
        logger.warning(
            f"Control allocation does not account for actuator limits. This leads to suboptimal allocation"
        )

    def allocate_output(self, command, output_mode):
        if self.force_application_level == "motor_link":
            if output_mode == "forces":
                motor_thrusts = self.update_motor_thrusts_with_forces(command)
            else:
                motor_thrusts = self.update_motor_thrusts_with_wrench(command)
            forces, torques = self.calc_motor_forces_torques_from_thrusts(motor_thrusts)

        else:
            output_wrench = self.update_wrench(command)
            forces = output_wrench[:, 0:3].unsqueeze(1)
            torques = output_wrench[:, 3:6].unsqueeze(1)

        return forces, torques

    def update_wrench(self, ref_wrench):

        ref_motor_thrusts = torch.bmm(
            self.inv_force_torque_allocation_matrix, ref_wrench.unsqueeze(-1)
        ).squeeze(-1)

        current_motor_thrust = self.motor_model.update_motor_thrusts(ref_motor_thrusts)

        self.output_wrench[:] = torch.bmm(
            self.force_torque_allocation_matrix, current_motor_thrust.unsqueeze(-1)
        ).squeeze(-1)

        return self.output_wrench

    def update_motor_thrusts_with_forces(self, ref_forces):
        current_motor_thrust = self.motor_model.update_motor_thrusts(ref_forces)
        return current_motor_thrust

    def update_motor_thrusts_with_wrench(self, ref_wrench):

        ref_motor_thrusts = torch.bmm(
            self.inv_force_torque_allocation_matrix, ref_wrench.unsqueeze(-1)
        ).squeeze(-1)

        current_motor_thrust = self.motor_model.update_motor_thrusts(ref_motor_thrusts)

        return current_motor_thrust

    def reset_idx(self, env_ids):
        self.motor_model.reset_idx(env_ids)
        # here we can randomize the allocation matrix if desired

    def reset(self):
        self.motor_model.reset()
        # here we can randomize the allocation matrix if desired

    def calc_motor_forces_torques_from_thrusts(self, motor_thrusts):
        motor_forces = torch.stack(
            [
                torch.zeros_like(motor_thrusts),
                torch.zeros_like(motor_thrusts),
                motor_thrusts,
            ],
            dim=2,
        )
        cq = self.cfg.motor_model_config.thrust_to_torque_ratio
        motor_torques = cq * motor_forces * (-self.motor_directions[None, :, None])
        return motor_forces, motor_torques

    def set_single_allocation_matrix(self, alloc_matrix):
        if alloc_matrix.shape != (6, self.cfg.num_motors):
            raise ValueError("Allocation matrix must have shape (6, num_motors)")
        self.force_torque_allocation_matrix[:] = torch.tensor(
            alloc_matrix, device=self.device, dtype=torch.float32
        ).expand(self.num_envs, -1, -1)
        self.inv_force_torque_allocation_matrix[:] = torch.linalg.pinv(
            self.force_torque_allocation_matrix
        ).expand(self.num_envs, -1, -1)
