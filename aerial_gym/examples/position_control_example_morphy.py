from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
import torch
from aerial_gym.utils.helpers import get_args

import matplotlib.pyplot as plt

if __name__ == "__main__":
    args = get_args()
    logger.warning(
        "This example demonstrates the use of geometric controllers with the Morphy robot in an empty environment."
    )

    env_manager = SimBuilder().build_env(
        sim_name="base_sim_2ms",
        env_name="empty_env_2ms",
        robot_name="morphy",
        controller_name="lee_position_control",
        args=None,
        device="cuda:0",
        num_envs=16,
        headless=args.headless,
        use_warp=args.use_warp,
    )
    actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")

    env_manager.reset()
    arm1_pitch_list = []
    arm1_yaw_list = []

    arm2_pitch_list = []
    arm2_yaw_list = []

    arm3_pitch_list = []
    arm3_yaw_list = []

    arm4_pitch_list = []
    arm4_yaw_list = []

    for i in range(10000):
        if i % 500 == 0:
            logger.info(f"Step {i}, changing target setpoint.")
            actions[:, 0:3] = 0 * 1.0 * (torch.rand_like(actions[:, 0:3]) * 2 - 1)
            actions[:, 3] = 0 * torch.pi * (torch.rand_like(actions[:, 3]) * 2 - 1)
            env_manager.reset()
            if i == 0:
                continue
            plt.plot(arm1_pitch_list, label="arm1_pitch")
            plt.plot(arm1_yaw_list, label="arm1_roll")

            plt.plot(arm2_pitch_list, label="arm2_pitch")
            plt.plot(arm2_yaw_list, label="arm2_roll")

            plt.plot(arm3_pitch_list, label="arm3_pitch")
            plt.plot(arm3_yaw_list, label="arm3_roll")

            plt.plot(arm4_pitch_list, label="arm4_pitch")
            plt.plot(arm4_yaw_list, label="arm4_roll")
            plt.legend()
            plt.show()
        env_manager.step(actions=actions)
        # get dof states and plot them
        dof_states = env_manager.global_tensor_dict["dof_state_tensor"]
        robot_0_dof_states = dof_states[0].cpu().numpy()
        # plot the states
        arm1_pitch_list.append(robot_0_dof_states[0, 0])
        arm1_yaw_list.append(robot_0_dof_states[1, 0])

        arm2_pitch_list.append(robot_0_dof_states[2, 0])
        arm2_yaw_list.append(robot_0_dof_states[3, 0])

        arm3_pitch_list.append(robot_0_dof_states[4, 0])
        arm3_yaw_list.append(robot_0_dof_states[5, 0])

        arm4_pitch_list.append(robot_0_dof_states[6, 0])
        arm4_yaw_list.append(robot_0_dof_states[7, 0])
