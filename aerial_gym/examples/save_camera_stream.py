import matplotlib.image
import numpy as np
import random
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
from PIL import Image
import matplotlib
import torch
import random

if __name__ == "__main__":
    logger.warning("\n\n\nEnvironment to save a depth/range and segmentation image.\n\n\n")

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="env_with_obstacles",  # "forest_env", #"empty_env", # empty_env
        robot_name="base_quadrotor_with_stereo_camera",
        controller_name="lee_velocity_control",
        args=None,
        device="cuda:0",
        num_envs=2,
        headless=False,
        use_warp=True,
    )
    actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")
    actions[:, 3] = 0.1

    env_manager.reset()
    seg_frames = []
    depth_frames = []
    merged_image_frames = []
    for i in range(101):
        if i % 100 == 0 and i > 0:
            print("i", i)
            env_manager.reset()
            # save frames as a gif:
            seg_frames[0].save(
                f"seg_frames_{i}.gif",
                save_all=True,
                append_images=seg_frames[1:],
                duration=100,
                loop=0,
            )
            depth_frames[0].save(
                f"depth_frames_{i}.gif",
                save_all=True,
                append_images=depth_frames[1:],
                duration=100,
                loop=0,
            )
            merged_image_frames[0].save(
                f"merged_image_frames_{i}.gif",
                save_all=True,
                append_images=merged_image_frames[1:],
                duration=100,
                loop=0,
            )
            # save each image individually
            seg_frames[0].save(f"seg_frame_{i}.png")
            depth_frames[0].save(f"depth_frame_{i}.png")
            merged_image_frames[0].save(f"merged_image_frame_{i}.png")
            seg_frames = []
            depth_frames = []
            merged_image_frames = []
        env_manager.step(actions=actions)
        env_manager.render(render_components="sensors")
        # reset envs that have crashed
        env_manager.reset_terminated_and_truncated_envs()
        try:
            image1 = (
                255.0 * env_manager.global_tensor_dict["depth_range_pixels"][0, 0].cpu().numpy()
            ).astype(np.uint8)
            seg_image1 = env_manager.global_tensor_dict["segmentation_pixels"][0, 0].cpu().numpy()
        except Exception as e:
            logger.error("Error in getting images")
            logger.error("Seems like the image tensors have not been created yet.")
            logger.error("This is likely due to absence of a functional camera in the environment")
            raise e
        # seg_image1[seg_image1 <= 0] = seg_image1[seg_image1 > 0].min()
        seg_image1_normalized = (seg_image1 - seg_image1.min()) / (
            seg_image1.max() - seg_image1.min()
        )

        # for when Isaac Gym cameras are used for RGB images
        # image_frame1 = env_manager.global_tensor_dict["rgb_pixels"][0, 0].cpu().numpy().astype(np.uint8)
        # # save to file
        # im1 = Image.fromarray(image_frame1)
        # im1.save(f"image_frame_{i}.png")

        # set colormap to plasma in matplotlib
        seg_image1_normalized_plasma = matplotlib.cm.plasma(seg_image1_normalized)
        seg_image1 = Image.fromarray((seg_image1_normalized_plasma * 255.0).astype(np.uint8))
        depth_image1 = Image.fromarray(image1)
        image_4d = np.zeros((image1.shape[0], image1.shape[1], 4))
        image_4d[:, :, 0] = image1
        image_4d[:, :, 1] = image1
        image_4d[:, :, 2] = image1
        image_4d[:, :, 3] = 255.0
        merged_image = np.concatenate((image_4d, seg_image1_normalized_plasma * 255.0), axis=0)
        # save frames to array:
        seg_frames.append(seg_image1)
        depth_frames.append(depth_image1)
        merged_image_frames.append(Image.fromarray(merged_image.astype(np.uint8)))

        # # save frame as png
        # seg_image1.save(f"seg_image_{i}.png")
        # depth_image1.save(f"depth_image_{i}.png")
        # Image.fromarray(merged_image.astype(np.uint8)).save(f"merged_image_{i}.png")
