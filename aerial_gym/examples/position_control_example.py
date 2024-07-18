from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
import torch
from aerial_gym.utils.helpers import get_args

if __name__ == "__main__":
    args = get_args()
    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="empty_env",
        robot_name="base_quadrotor",  # base_fully_actuated
        controller_name="lee_position_control",
        args=None,
        device="cuda:0",
        num_envs=args.num_envs,
        headless=args.headless,
        use_warp=args.use_warp,
    )
    actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")
    env_manager.reset()
    logger.info(
        "\n\n\n\n\n\n This script provides an example of a geometric position controller for a standard quadrotor robot. \n\n\n\n\n\n"
    )
    for i in range(10000):
        if i % 500 == 0:
            logger.info(f"Step {i}, changing target setpoint.")
            actions[:, 0:3] = 2.0 * (torch.rand_like(actions[:, 0:3]) * 2 - 1)
            actions[:, 3] = torch.pi * (torch.rand_like(actions[:, 3]) * 2 - 1)
        env_manager.step(actions=actions)
