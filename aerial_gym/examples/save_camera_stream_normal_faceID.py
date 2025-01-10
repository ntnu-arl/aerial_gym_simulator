import numpy as np
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
from PIL import Image
import matplotlib
import torch
import os, random

seed = 0

if __name__ == "__main__":
    logger.warning("\n\n\nEnvironment to save a normal and faceid image.\n\n\n")

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    logger.critical("Setting seed: {}".format(seed))



    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="forest_env",  # "forest_env", #"empty_env", # empty_env
        robot_name="base_quadrotor_with_faceid_normal_camera",
        controller_name="lee_velocity_control",
        args=None,
        device="cuda:0",
        num_envs=2,
        headless=False,
        use_warp=True,
    )
    actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")

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

            # save the last frame as a PNG as well
            seg_frames[-1].save(f"faceid_frames_{i}.png")
            depth_frames[-1].save(f"normal_frames_{i}.png")



            seg_frames = []
            depth_frames = []
            merged_image_frames = []
        env_manager.step(actions=actions)
        env_manager.render(render_components="sensors")
        # reset envs that have crashed
        env_manager.reset_terminated_and_truncated_envs()
        try:
            # define a normal direction to take a dot product with
            one_vec = torch.zeros_like(env_manager.global_tensor_dict["depth_range_pixels"])
            one_vec[..., 0] = 1.0
            one_vec[..., 1] = 1 / 2.0
            one_vec[..., 2] = 1 / 3.0
            one_vec[:] = one_vec / torch.norm(one_vec, dim=-1, keepdim=True)
            cosine_vec = torch.abs(
                torch.sum(one_vec * env_manager.global_tensor_dict["depth_range_pixels"], dim=-1)
            )
            # max_dr = torch.max(cosine_vec)
            # min_dr = torch.min(cosine_vec)

            # print(torch.mean(cosine_vec), max_dr, min_dr)
            image1 = (255.0 * cosine_vec)[0, 0].cpu().numpy().astype(np.uint8)

            seg_image1 = env_manager.global_tensor_dict["segmentation_pixels"][0, 0].cpu().numpy()
        except Exception as e:
            logger.error("Error in getting images")
            logger.error("Seems like the image tensors have not been created yet.")
            logger.error("This is likely due to absence of a functional camera in the environment")
            raise e
        
        # discretize image for better visualization
        seg_image1[seg_image1 > 0] = (10*np.mod(seg_image1[seg_image1 > 0], 26) + 1).astype(np.uint8)
        seg_image1[seg_image1 <= 0] = 0
        # set colormap to plasma in matplotlib
        seg_image1_normalized_plasma = matplotlib.cm.plasma(seg_image1/255.0)
        mod_image = (255.0*seg_image1_normalized_plasma).astype(np.uint8)
        
        # set channel to opaque
        mod_image[:, :, 3] = 255
        seg_image1_discrete = Image.fromarray(mod_image)
        seg_image1 = Image.fromarray((seg_image1_normalized_plasma * 255.0).astype(np.uint8))

        depth_image1 = Image.fromarray(image1)
        image_4d = np.zeros((image1.shape[0], image1.shape[1], 4))
        image_4d[:, :, 0] = image1
        image_4d[:, :, 1] = image1
        image_4d[:, :, 2] = image1
        image_4d[:, :, 3] = 255.0
        merged_image = np.concatenate((image_4d, seg_image1_discrete), axis=0)
        # save frames to array:
        seg_frames.append(seg_image1_discrete)
        depth_frames.append(depth_image1)
        merged_image_frames.append(Image.fromarray(merged_image.astype(np.uint8)))
