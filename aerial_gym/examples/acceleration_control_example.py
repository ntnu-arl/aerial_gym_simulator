from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
import torch

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
        env_name="env_with_obstacles",  # empty_env
        robot_name="base_quadrotor",  # "base_octarotor"
        controller_name="lee_acceleration_control",
        args=None,
        num_envs=16,
        device="cuda:0",
        headless=False,
        use_warp=True,  # safer to use warp as it disables the camera when no object is in the environment
    )
    actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")
    actions[:, 0] = 0.25  # constant forward acceleration
    env_manager.reset()
    for i in range(1000):
        if i % 100 == 0:
            print("i", i)
            env_manager.reset()
        env_manager.step(actions=actions)
        env_manager.render()
        env_manager.reset_terminated_and_truncated_envs()
