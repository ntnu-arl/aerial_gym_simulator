from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
import torch
from aerial_gym.utils.helpers import get_args

import matplotlib.pyplot as plt

from aerial_gym.config.sim_config.base_sim_no_gravity_config import BaseSimNoGravityConfig
from aerial_gym.registry.sim_registry import sim_config_registry

if __name__ == "__main__":
    args = get_args()
    logger.warning(
        "\n\n\nThis example demonstrates shape control of a reconfigurable robot with joint angle setpoints. Motor control for this robot is not implemented.\n\n\n"
    )
    BaseSimNoGravityConfig.sim.dt = 0.002
    print(BaseSimNoGravityConfig.sim.gravity)
    sim_config_registry.register("base_sim_no_gravity_2ms", BaseSimNoGravityConfig)
    env_manager = SimBuilder().build_env(
        sim_name="base_sim_no_gravity_2ms",
        env_name="empty_env_2ms",
        robot_name="snakey",
        controller_name="no_control",
        args=None,
        device="cuda:0",
        num_envs=16,
        headless=args.headless,
        use_warp=args.use_warp,
    )
    actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")
    actions[:, [0, 1, 2]] = 0.0
    env_manager.reset()

    for i in range(10000):
        if i % 200 == 0:
            logger.info(f"Step {i}, changing target shape.")
            env_manager.reset()
            dof_pos = 2*(torch.ones((env_manager.num_envs, 6)).to("cuda:0") - 0.5)
            env_manager.robot_manager.robot.set_dof_velocity_targets((3.14159 / 5.0) * dof_pos)
        env_manager.step(actions=actions)
