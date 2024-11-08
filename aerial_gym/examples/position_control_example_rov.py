from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
import torch

from aerial_gym.utils.math import quat_from_euler_xyz_tensor

from aerial_gym.config.sim_config.base_sim_no_gravity_config import BaseSimNoGravityConfig
from aerial_gym.registry.sim_registry import sim_config_registry

if __name__ == "__main__":
    sim_config_registry.register("base_sim_no_gravity", BaseSimNoGravityConfig)
    logger.warning("This example demonstrates the use of geometric controllers for a rov.")
    env_manager = SimBuilder().build_env(
        sim_name="base_sim_no_gravity",
        env_name="empty_env",
        robot_name="base_rov",
        controller_name="fully_actuated_control",
        args=None,
        device="cuda:0",
        num_envs=64,
        headless=False,
        use_warp=False,  # since there is not supposed to be a camera in the robot for this example.
    )
    actions = torch.zeros((env_manager.num_envs, 7)).to("cuda:0")
    actions[:, 6] = 1.0
    env_manager.reset()
    logger.info(
        "\n\n\n\n\n\n This script provides an example of a custom geometric position controller for a custom BlueROV robot. \n\n\n\n\n\n"
    )
    for i in range(10000):
        if i % 500 == 0:
            logger.info(f"Step {i}, changing target setpoint.")
            actions[:, 0:3] = 2.0 * (torch.rand_like(actions[:, 0:3]) * 2 - 1)
            actions[:, 3:7] = quat_from_euler_xyz_tensor(
                torch.pi * (torch.rand_like(actions[:, 3:6]) * 2 - 1)
            )
            # actions[:, 3] = 0.0#torch.pi * (torch.rand_like(actions[:, 3]) * 2 - 1)
        env_manager.step(actions=actions)
