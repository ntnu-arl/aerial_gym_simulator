from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
import torch

from aerial_gym.sim import sim_config_registry
from aerial_gym.config.sim_config.base_sim_no_gravity_config import BaseSimConfig

from tqdm import tqdm
import time

if __name__ == "__main__":
    sim_config_registry.register("base_sim_no_gravity", BaseSimConfig)

    env_manager = SimBuilder().build_env(
        sim_name="base_sim_no_gravity",
        env_name="empty_env",
        robot_name="base_quadrotor_with_imu",
        controller_name="lee_velocity_control",
        args=None,
        device="cuda:0",
        num_envs=2,
        headless=True,
    )
    if env_manager.robot_manager.robot.cfg.sensor_config.enable_imu == False:
        logger.error(
            "The IMU is disabled for this environment. The IMU data collection will not work."
        )
        exit(1)
    tensor_dict = env_manager.global_tensor_dict
    actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")
    env_manager.reset()
    start_time = time.time()
    # f = open("simulated_imu_data.csv", "w")
    sim_dt = env_manager.sim_config.sim.dt
    for i in range(int(3.0 * 3600 / 0.005)):
        env_manager.step(actions=actions)
        imu_measurement = tensor_dict["imu_measurement"][0].cpu().numpy()
        print(imu_measurement)
        # f.write(
        #     f"{i*sim_dt},{imu_measurement[0]},{imu_measurement[1]},{imu_measurement[2]},\
        #         {imu_measurement[3]},{imu_measurement[4]},{imu_measurement[5]}\n"
        # )
