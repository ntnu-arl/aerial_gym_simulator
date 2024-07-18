import time
import numpy as np

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)

from aerial_gym.sim.sim_builder import SimBuilder


from PIL import Image
import torch

if __name__ == "__main__":
    start = time.time()

    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="empty_env",
        robot_name="base_quadrotor",
        controller_name="lee_position_control",
        args=None,
        device="cuda:0",
        num_envs=int(524288 / 4),
        headless=True,
        use_warp=True,  # since there isn't supposed to be a camera in the robot for this example owing to lack of obstacles
    )
    actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")
    env_manager.reset()
    elapsed_steps = -100
    with torch.no_grad():
        for i in range(10000):
            # Allow the simulator to warm up a little bit before measuring the time
            if i == 100:
                start = time.time()
                elapsed_steps = 0
            env_manager.step(actions=actions)
            elapsed_steps += 1
            if i % 50 == 0:
                logger.critical(
                    f"i {elapsed_steps}, Current time: {time.time() - start}, FPS: {elapsed_steps * env_manager.num_envs / (time.time() - start)}, Real Time Speedup: {elapsed_steps * env_manager.num_envs * env_manager.sim_config.sim.dt / (time.time() - start)}"
                )
            if i % 1000 == 0:
                logger.error(f"i {i}")
    end = time.time()
