import time
import isaacgym

# isort: on
import torch
from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym import (
    parse_aerialgym_cfg,
)
from aerial_gym.utils import get_args
from aerial_gym.registry.task_registry import task_registry


from aerial_gym.examples.dce_rl_navigation.dce_navigation_task import DCE_RL_Navigation_Task
from aerial_gym.examples.dce_rl_navigation.sf_inference_class import NN_Inference_Class

import matplotlib
import numpy as np
from PIL import Image


def sample_command(args):
    use_warp = True
    headless = args.headless
    # seg_frames = []
    # depth_frames = []
    # merged_image_frames = []

    rl_task = task_registry.make_task(
        "dce_navigation_task", seed=42, use_warp=use_warp, headless=headless
    )
    print("Number of environments", rl_task.num_envs)
    command_actions = torch.zeros((rl_task.num_envs, rl_task.task_config.action_space_dim))
    command_actions[:, 0] = 1.5
    command_actions[:, 1] = 0.0
    command_actions[:, 2] = 0.0
    nn_model = get_network(rl_task.num_envs)
    nn_model.eval()
    nn_model.reset(torch.arange(rl_task.num_envs))
    rl_task.reset()
    for i in range(0, 50000):
        start_time = time.time()
        obs, rewards, termination, truncation, infos = rl_task.step(command_actions)

        obs["obs"] = obs["observations"]
        # print(obs["observations"].shape)
        action = nn_model.get_action(obs)
        # print("Action", action, action.shape)
        action = torch.tensor(action).expand(rl_task.num_envs, -1)
        command_actions[:] = action

        reset_ids = (termination + truncation).nonzero(as_tuple=True)
        if torch.any(termination):
            terminated_envs = termination.nonzero(as_tuple=True)
            print(f"Resetting environments {terminated_envs} due to Termination")
        if torch.any(truncation):
            truncated_envs = truncation.nonzero(as_tuple=True)
            print(f"Resetting environments {truncated_envs} due to Timeout")
        nn_model.reset(reset_ids)

    # # Uncomment the below lines to save the frames from an episode as a GIF
    #     # save obs to file as a .gif
    #     image1 = (
    #         255.0 * rl_task.obs_dict["depth_range_pixels"][0, 0].cpu().numpy()
    #     ).astype(np.uint8)
    #     seg_image1 = rl_task.obs_dict["segmentation_pixels"][0, 0].cpu().numpy()
    #     seg_image1[seg_image1 <= 0] = seg_image1[seg_image1 > 0].min()
    #     seg_image1_normalized = (seg_image1 - seg_image1.min()) / (
    #         seg_image1.max() - seg_image1.min()
    #     )

    #     # set colormap to plasma in matplotlib
    #     seg_image1_normalized_plasma = matplotlib.cm.plasma(seg_image1_normalized)
    #     seg_image1 = Image.fromarray((seg_image1_normalized_plasma * 255.0).astype(np.uint8))

    #     depth_image1 = Image.fromarray(image1)
    #     image_4d = np.zeros((image1.shape[0], image1.shape[1], 4))
    #     image_4d[:, :, 0] = image1
    #     image_4d[:, :, 1] = image1
    #     image_4d[:, :, 2] = image1
    #     image_4d[:, :, 3] = 255.0
    #     merged_image = np.concatenate((image_4d, seg_image1_normalized_plasma * 255.0), axis=0)
    #     # save frames to array:
    #     seg_frames.append(seg_image1)
    #     depth_frames.append(depth_image1)
    #     merged_image_frames.append(Image.fromarray(merged_image.astype(np.uint8)))
    # if termination[0] or truncation[0]:
    #     print("i", i)
    #     rl_task.reset()
    #     # save frames as a gif:
    #     seg_frames[0].save(
    #         f"seg_frames_{i}.gif",
    #         save_all=True,
    #         append_images=seg_frames[1:],
    #         duration=100,
    #         loop=0,
    #     )
    #     depth_frames[0].save(
    #         f"depth_frames_{i}.gif",
    #         save_all=True,
    #         append_images=depth_frames[1:],
    #         duration=100,
    #         loop=0,
    #     )
    #     merged_image_frames[0].save(
    #         f"merged_image_frames_{i}.gif",
    #         save_all=True,
    #         append_images=merged_image_frames[1:],
    #         duration=100,
    #         loop=0,
    #     )
    #     seg_frames = []
    #     depth_frames = []
    #     merged_image_frames = []


def get_network(num_envs):
    """Script entry point."""
    # register_aerialgym_custom_components()
    cfg = parse_aerialgym_cfg(evaluation=True)
    print("CFG is:", cfg)
    nn_model = NN_Inference_Class(num_envs, 3, 81, cfg)
    return nn_model


if __name__ == "__main__":
    task_registry.register_task(
        task_name="dce_navigation_task",
        task_class=DCE_RL_Navigation_Task,
        task_config=task_registry.get_task_config(
            "navigation_task"
        ),  # same config as navigation task
    )
    args = get_args()
    sample_command(args)
