import time
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.registry.task_registry import task_registry
import torch

if __name__ == "__main__":
    logger.print_example_message()
    start = time.time()
    rl_task_env = task_registry.make_task(
        "position_setpoint_task",
        # other params are not set here and default values from the task config file are used
    )
    rl_task_env.reset()
    actions = torch.zeros(
        (
            rl_task_env.sim_env.num_envs,
            rl_task_env.sim_env.robot_manager.robot.controller_config.num_actions,
        )
    ).to("cuda:0")
    actions[:] = 0.0
    logger.info("\n\n\n\n\n\n Example of a positon setpoint task interface. \n\n\n\n\n\n")
    with torch.no_grad():
        for i in range(10000):
            obs, reward, terminated, truncated, info = rl_task_env.step(actions=actions)
