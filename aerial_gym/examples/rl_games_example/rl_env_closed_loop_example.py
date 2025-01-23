import time
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.registry.task_registry import task_registry
import torch

from aerial_gym.examples.rl_games_example.rl_games_inference import MLP

import time
import numpy as np

import matplotlib.pyplot as plt

if __name__ == "__main__":
    logger.print_example_message()
    start = time.time()
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    plt.style.use("seaborn-v0_8-colorblind")
    rl_task_env = task_registry.make_task(
        "position_setpoint_task",
        # "position_setpoint_task_acceleration_sim2real",
        # other params are not set here and default values from the task config file are used
        seed=seed,
        headless=False,
        num_envs=24,
        use_warp=True,
    )
    rl_task_env.reset()
    actions = torch.zeros(
        (
            rl_task_env.sim_env.num_envs,
            rl_task_env.task_config.action_space_dim,
        )
    ).to("cuda:0")
    model = (
        MLP(
            rl_task_env.task_config.observation_space_dim,
            rl_task_env.task_config.action_space_dim,
            # "networks/morphy_policy_for_rigid_airframe.pth"
            "networks/attitude_policy.pth"
            # "networks/morphy_policy_for_flexible_airframe_joint_aware.pth",
        )
        .to("cuda:0")
        .eval()
    )
    actions[:] = 0.0
    counter = 0
    action_list = []
    error_list = []
    joint_pos_list = []
    joint_vel_list = []
    with torch.no_grad():
        for i in range(10000):
            if i == 100:
                start = time.time()
            obs, reward, terminated, truncated, info = rl_task_env.step(actions=actions)
            start_time = time.time()
            actions[:] = model.forward(obs["observations"])

            end_time = time.time()
            
    end = time.time()
