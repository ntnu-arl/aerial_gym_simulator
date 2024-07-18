import time

import numpy as np

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)

from aerial_gym.sim.sim_builder import SimBuilder

from PIL import Image

if __name__ == "__main__":
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")
    start = time.time()

    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="env_with_obstacles",
        robot_name="base_quadrotor",
        args=None,
        device="cuda:0",
    )

    env_manager.reset()
    for i in range(1000):
        if i % 100 == 0:
            print("i", i)
            env_manager.reset()
        env_manager.step(actions=None)
        env_manager.IGE_env.viewer.render()
        image1 = (
            255.0 * env_manager.global_tensor_dict["depth_range_pixels"][0, 0].cpu().numpy()
        ).astype(np.uint8)
        seg_image1 = env_manager.global_tensor_dict["segmentation_pixels"][0, 0].cpu().numpy()
        seg_image1[seg_image1 <= 0] = seg_image1[seg_image1 > 0].min()
        seg_image1_normalized = (
            255.0 * (seg_image1 - seg_image1.min()) / (seg_image1.max() - seg_image1.min())
        )
        seg_image1_normalized = seg_image1_normalized.astype(np.uint8)

        Image.fromarray(image1).save(f"1/depth_image_{i}_1.png")
        Image.fromarray(seg_image1_normalized).save(f"1/seg_image_{i}_1.png")
        unique_seg = np.unique(seg_image1_normalized)
        logger.error(f"unique seg {unique_seg}")

        image2 = (
            255.0 * env_manager.global_tensor_dict["depth_range_pixels"][1, 0].cpu().numpy()
        ).astype(np.uint8)
        seg_image2 = env_manager.global_tensor_dict["segmentation_pixels"][1, 0].cpu().numpy()
        seg_image2[seg_image2 <= 0] = seg_image2[seg_image2 > 0].min()
        seg_image2_normalized = (
            255.0 * (seg_image2 - seg_image2.min()) / (seg_image2.max() - seg_image2.min())
        )
        seg_image2_normalized = seg_image2_normalized.astype(np.uint8)
        Image.fromarray(seg_image2_normalized).save(f"2/seg_image_{i}_2.png")
        Image.fromarray(image2).save(f"2/depth_image_{i}_2.png")
        unique_seg = np.unique(seg_image2_normalized)
        logger.error(f"unique seg {unique_seg}")

    # for i in range(1000):
    #     actions = torch.randn((env_config.env.num_envs, 4)).to("cuda:0")
    #     env_manager.step(actions)
    end = time.time()
    # print(f"Time taken for {env_config.env.num_envs} envs ", end-start)
