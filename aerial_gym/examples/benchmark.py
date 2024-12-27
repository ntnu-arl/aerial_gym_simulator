import time
import numpy as np

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)

from aerial_gym.sim.sim_builder import SimBuilder

import torch
import numpy as np

from PIL import Image
import torch

if __name__ == "__main__":
    start = time.time()
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    rendering_benchmark = False
    logger.warning(
        "This script provides an example of a rendering benchmark for the environment. The rendering benchmark will measure the FPS and the real-time speedup of the environment."
    )
    logger.warning(
        "\n\n\nThe rendering benchmark will run by default. Please set rendering_benchmark = False to run the physics benchmark. \n\n\n"
    )

    if rendering_benchmark == True:
        env_manager = SimBuilder().build_env(
            sim_name="base_sim",
            env_name="env_with_obstacles",
            robot_name="base_quadrotor_with_camera",
            controller_name="lee_velocity_control",
            args=None,
            device="cuda:0",
            num_envs=16,
            headless=True,
            use_warp=True,
        )
        if env_manager.robot_manager.robot.cfg.sensor_config.enable_camera == False:
            logger.error(
                "The camera is disabled for this environment. The rendering benchmark will not work."
            )
            exit(1)
    else:
        env_manager = SimBuilder().build_env(
            sim_name="base_sim",
            env_name="empty_env",
            robot_name="base_quadrotor",
            controller_name="no_control",
            args=None,
            device="cuda:0",
            num_envs=256,
            headless=True,
            use_warp=True,
        )
        if env_manager.robot_manager.robot.cfg.sensor_config.enable_camera == True:
            logger.critical(
                "The camera is enabled for this environment. The This will cause the benchmark to be slower than expected. Please disable the camera for a more accurate benchmark."
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
            if rendering_benchmark == True:
                env_manager.render(render_components="sensor")
            elapsed_steps += 1
            if i % 50 == 0:
                if i < 0:
                    logger.warning("Warming up....")
                else:
                    logger.critical(
                        f"i {elapsed_steps}, Current time: {time.time() - start}, FPS: {elapsed_steps * env_manager.num_envs / (time.time() - start)}, Real Time Speedup: {elapsed_steps * env_manager.num_envs * env_manager.sim_config.sim.dt / (time.time() - start)}"
                    )
    end = time.time()
