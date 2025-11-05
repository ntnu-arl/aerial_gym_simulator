"""
Test script for the procedural_forest environment.

This script tests:
1. Environment creation and initialization
2. Tree rendering and positioning
3. Environment bounds
4. Multiple resets to verify randomization
5. Camera rendering to generate segmentation and depth images

Usage:
    python test_procedural_forest.py
"""

import os

import numpy as np
import torch

from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Testing Procedural Forest Environment")
    logger.info("=" * 60)

    # Set seed for reproducibility during testing
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        logger.info("Creating procedural_forest environment...")
        env_manager = SimBuilder().build_env(
            sim_name="base_sim",
            env_name="procedural_forest",
            robot_name="base_quadrotor_with_camera",
            controller_name="lee_position_control",
            args=None,
            device="cuda:0",
            num_envs=1,
            headless=False,
            use_warp=True,
        )

        logger.info("Environment created successfully!")
        logger.info(f"  - Number of environments: {env_manager.num_envs}")
        logger.info(f"  - Device: {env_manager.device}")

        env_bounds_min = env_manager.global_tensor_dict["env_bounds_min"]
        env_bounds_max = env_manager.global_tensor_dict["env_bounds_max"]
        logger.info(f"  - Environment bounds (min): {env_bounds_min[0].cpu().numpy()}")
        logger.info(f"  - Environment bounds (max): {env_bounds_max[0].cpu().numpy()}")

        has_camera = "depth_range_pixels" in env_manager.global_tensor_dict
        if has_camera:
            logger.info("  - Camera sensors detected: Segmentation and depth images will be generated")
        else:
            logger.warning("  - No camera sensors detected. Images will not be generated.")

        actions = torch.zeros((env_manager.num_envs, 4)).to(env_manager.device)
        output_dir = "procedural_forest_images"
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"  - Images will be saved to: {output_dir}/")

        for i in range(1000000):
            if i % 2000 == 0:
                logger.info(f"Step {i}: Resetting environment...")
                env_manager.reset()

                if "env_asset_state_tensor" in env_manager.global_tensor_dict:
                    asset_positions = env_manager.global_tensor_dict["env_asset_state_tensor"][0, :, 0:3]
                    logger.info(f"  Tree positions in env 0: {asset_positions.cpu().numpy()[:5]}")

            env_manager.step(actions=actions)

            if has_camera:
                env_manager.render(render_components="sensors")

            env_manager.reset_terminated_and_truncated_envs()

        logger.info("\n" + "=" * 60)
        logger.info("Basic test completed successfully!")
        logger.info(f"Images saved to: {output_dir}/")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        raise
