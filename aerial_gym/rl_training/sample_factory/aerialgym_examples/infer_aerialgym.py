import time
from collections import deque
from typing import Dict, Tuple

from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym import (
    parse_aerialgym_cfg,
)

import numpy as np
import torch
from torch import Tensor

from sample_factory.algo.utils.gymnasium_utils import convert_space
from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config, StatusCode
from gymnasium import spaces



from torch import nn

class NN_Inference_ROS(nn.Module):
    def __init__(self, cfg: Config, env_cfg) -> None:
        super().__init__()
        self.cfg = load_from_checkpoint(cfg)
        print("cfg: ", self.cfg)
        self.cfg.num_envs = 1
        self.num_actions = 4
        self.num_obs = 17 + 16*20 #15 #+ self.num_actions * 10
        self.num_agents = 1
        self.observation_space = spaces.Dict(dict(observations=convert_space(spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf))))
        self.action_space = convert_space(spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.))
        self.init_env_info()
        self.actor_critic = create_actor_critic(self.cfg, self.observation_space, self.action_space)
        self.actor_critic.eval()
        device = torch.device("cpu" if self.cfg.device == "cpu" else "cuda")
        self.actor_critic.model_to_device(device)
        print("Model:\n\n", self.actor_critic)
        # Load policy into model
        policy_id = 0 #self.cfg.policy_index
        name_prefix = dict(latest="checkpoint", best="best")[self.cfg.load_checkpoint_kind]
        checkpoints = Learner.get_checkpoints(
            Learner.checkpoint_dir(self.cfg, policy_id), f"{name_prefix}_*"
        )
        checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
        self.actor_critic.load_state_dict(checkpoint_dict["model"])
        self.rnn_states = torch.zeros(
            [self.num_agents, get_rnn_size(self.cfg)],
            dtype=torch.float32,
            device=device,
        )

    def init_env_info(self):
        self.env_info = EnvInfo(
            obs_space=self.observation_space,
            action_space=self.action_space,
            num_agents=self.num_agents,
            gpu_actions=self.cfg.env_gpu_actions,
            gpu_observations=self.cfg.env_gpu_observations,
            action_splits=None,
            all_discrete=None,
            frameskip=self.cfg.env_frameskip,
        )

    def reset(self):
        self.rnn_states[:] = 0.0

    def get_action(self, obs):
        start_time = time.time()
        with torch.no_grad():
            # put obs to device
            processed_obs = prepare_and_normalize_obs(self.actor_critic, obs)
            policy_outputs = self.actor_critic(processed_obs, self.rnn_states)
            # sample actions from the distribution by default
            actions = policy_outputs["actions"]
            action_distribution = self.actor_critic.action_distribution()
            actions = argmax_actions(action_distribution)

            # actions shape should be [num_agents, num_actions] even if it's [1, 1]
            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(self.env_info, actions)

            self.rnn_states = policy_outputs["new_rnn_states"]
        #actions_np = actions[0].cpu().numpy()
        #print("Time to get action:", time.time() - start_time)
        return actions

from sample_factory.model.encoder import *

class CustomEncoder(Encoder):
    """Just an example of how to use a custom model component."""

    def __init__(self, cfg, obs_space):
        super().__init__(cfg)
        # the observatiosn are in the following format:
        # 17 dim state + actions
        # 80 dim lidar readings
        
        # encode lidar readings using conv network and then combine with state vector and pass through MLP
        # all are flattened into "observations" key
        self.encoders = nn.ModuleDict()

        state_action_input_size = 17
        lidar_input_size = 16*20
        out_size = 0
        out_size_cnn = 0
        out_size += obs_space["observations"].shape[0] - lidar_input_size

        # self.encoders["obs_image"] = make_img_encoder(cfg, spaces.Box(low=-1, high=1.5, shape=(1, 16, 20)))
        # out_size += self.encoders["obs_image"].get_out_size()
        ###
        # input is 16 x 20 dims lidar readings
        self.encoders["obs_image"] = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # (B, 16, 16, 20)
            nn.ELU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (B, 16, 8, 10)
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # (B, 32, 8, 10)
            nn.ELU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (B, 32, 4, 5)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (B, 64, 4, 5)
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 64, 2, 2)
            nn.Flatten(),  # (B, 64*2*2)
        )
        out_size_cnn += 128
        out_size += out_size_cnn
        ###

        self.encoder_out_size = out_size
        mlp_layers = [256, 128, 64]
        mlp_input_size = out_size
        mlp = []
        for layer_size in mlp_layers:
            mlp.append(nn.Linear(mlp_input_size, layer_size))
            mlp.append(nn.ELU())
            mlp_input_size = layer_size
        self.mlp_head_custom = nn.Sequential(*mlp)
        if len(mlp_layers) > 0:
            self.encoder_out_size = mlp_layers[-1]
        else:
            self.encoder_out_size = out_size
            self.mlp_head_custom = nn.Identity()


    def forward(self, obs_dict):
        x_state_action = obs_dict["observations"][:, :17]
        x_lidar = obs_dict["observations"][:, 17:].unsqueeze(1).view(-1, 1, 16, 20)  # (B, 1, 8, 10)
        x_lidar_encoding = self.encoders["obs_image"](x_lidar)
        x = torch.cat([x_state_action, x_lidar_encoding], dim=-1)
        x = self.mlp_head_custom(x)
        return x

    def get_out_size(self) -> int:
        return self.encoder_out_size


from sample_factory.algo.utils.context import global_model_factory

def make_custom_encoder(cfg, obs_space):
    return CustomEncoder(cfg, obs_space)

def register_aerialgym_custom_components():
    global_model_factory().register_encoder_factory(CustomEncoder)

def main():
    """Script entry point."""
    print("Starting inference script")
    register_aerialgym_custom_components()
    cfg = parse_aerialgym_cfg(evaluation=True)
    model = NN_Inference_ROS(cfg, None)
    obs = {
        "observations": torch.zeros((1, model.num_obs), dtype=torch.float32)
    }
    action = model.get_action(obs)
    print("Sample action from the model:", action)
    from sample_factory.enjoy import enjoy

    status = enjoy(cfg)
    return


if __name__ == "__main__":
    status = main()
    # exit(status.status.value)