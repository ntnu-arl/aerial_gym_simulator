from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
import torch
from aerial_gym.utils.helpers import get_args

if __name__ == "__main__":
    logger.warning(
        "\n\n\nWhile possible, a dynamic environment will slow down the simulation by a lot. Use with caution. Native Isaac Gym cameras work faster than Warp in this case.\n\n\n"
    )
    args = get_args()
    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="dynamic_env",
        robot_name="lmf2",
        controller_name="lmf2_position_control",
        args=None,
        device="cuda:0",
        num_envs=args.num_envs,
        headless=args.headless,
        use_warp=args.use_warp,
    )
    actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")
    env_manager.reset()
    num_assets_in_env = (
        env_manager.IGE_env.num_assets_per_env - 1
    )  # subtract 1 because the robot is also an asset
    print(f"Number of assets in the environment: {num_assets_in_env}")
    num_envs = env_manager.num_envs

    asset_twist = torch.zeros(
        (num_envs, num_assets_in_env, 6), device="cuda:0", requires_grad=False
    )
    asset_twist[:, :, 0] = -1.0
    for i in range(10000):
        if i % 1000 == 0:
            logger.info(f"Step {i}, changing target setpoint.")
            actions[:, 0:3] = 2.0 * (torch.rand_like(actions[:, 0:3]) * 2 - 1)
            actions[:, 3] = torch.pi * (torch.rand_like(actions[:, 3]) * 2 - 1)
            # env_manager.reset()
        asset_twist[:, :, 0] = torch.sin(0.2 * i * torch.ones_like(asset_twist[:, :, 0]))
        asset_twist[:, :, 1] = torch.cos(0.2 * i * torch.ones_like(asset_twist[:, :, 1]))
        asset_twist[:, :, 2] = 0.0
        env_manager.step(actions=actions, env_actions=asset_twist)
