from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
import torch


import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    logger.debug("this is how a debug message looks like")
    logger.info("this is how an info message looks like")
    logger.warning("this is how a warning message looks like")
    logger.error("this is how an error message looks like")
    logger.critical("this is how a critical message looks like")
    logger.info(
        "\n\n\n\n\n\n This script provides an example of a robot with constant forward acceleration directly input to the environment. \n\n\n\n\n\n"
    )
    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="empty_env",  # empty_env
        robot_name="lmf2",  # "base_octarotor"
        controller_name="lee_velocity_control",
        args=None,
        num_envs=16,
        device="cuda:0",
        headless=False,
        use_warp=True,  # safer to use warp as it disables the camera when no object is in the environment
    )
    actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")
    actions[:, 0] = 10.0  # constant forward acceleration
    actions[:, 1] = 5.0  # no lateral acceleration
    actions[:, 2] = 2.0  # no vertical acceleration
    actions[:, 3] = -0.1  # no yaw acceleration
    env_manager.reset()
    obs_dict = env_manager.get_obs()

    position_array_list = np.zeros((2000, 3), dtype=np.float32)
    velocity_array_list = np.zeros((2000, 3), dtype=np.float32)


    for i in range(10000):
        if i % 1000 == 0 and i > 0:
            print("i", i)
            # env_manager.reset()
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.plot(position_array_list[:, 0], label="X Position")
            plt.plot(position_array_list[:, 1], label="Y Position")
            plt.plot(position_array_list[:, 2], label="Z Position")
            plt.legend()
            plt.title("Position")
            plt.subplot(1, 2, 2)
            plt.plot(velocity_array_list[:, 0], label="X Velocity")
            plt.plot(velocity_array_list[:, 1], label="Y Velocity")
            plt.plot(velocity_array_list[:, 2], label="Z Velocity")
            plt.legend()
            plt.title("Velocity")
            plt.tight_layout()
            plt.show()
            # position_array_list = np.zeros((1000, 3), dtype=np.float32)
            # velocity_array_list = np.zeros((1000, 3), dtype=np.float32)
            actions = -actions

        env_manager.step(actions=actions)
        env_manager.render()
        env_manager.reset_terminated_and_truncated_envs()

        position_array_list[i % 2000, 0:3] = obs_dict["robot_position"][0].cpu().numpy()
        velocity_array_list[i % 2000, 0:3] = obs_dict["robot_body_linvel"][0].cpu().numpy()

