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
        robot_name="base_quadrotor",
        controller_name="no_control",
        args=None,
        device="cuda:0",
        num_envs=2,
        headless=True,
    )
    tensor_dict = env_manager.global_tensor_dict
    actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")
    env_manager.reset()
    start_time = time.time()
    f = open("imu_data_2.csv", "w")
    for i in tqdm(range(int(3.0 * 3600 / 0.005))):
        # env_manager.step(actions=actions)
        env_manager.post_physics_step(actions=actions)
        tensor_dict["imu_measurement"][:, 2] += 9.81
        imu_measurement = tensor_dict["imu_measurement"][0].cpu().numpy()
        f.write(
            f"{i*0.005},{imu_measurement[0]},{imu_measurement[1]},{imu_measurement[2]},\
                {imu_measurement[3]},{imu_measurement[4]},{imu_measurement[5]}\n"
        )
