import sys
from aerial_gym.registry.task_registry import task_registry
from sample_factory.utils.utils import str2bool
from sample_factory.algo.utils.gymnasium_utils import convert_space
from sample_factory.envs.env_utils import register_env
from sample_factory.utils.typing import Config, Env
from sample_factory.train import run_rl
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args

from typing import Dict, List, Optional, Tuple

import gymnasium as gym

import torch
from torch import Tensor

class AerialGymVecEnv(gym.Env):
    """
    Wrapper for isaacgym environments to make them compatible with the sample factory.
    """

    def __init__(self, aerialgym_env, obs_key):
        self.env = aerialgym_env
        self.num_agents = self.env.num_envs
        self.action_space = convert_space(self.env.action_space)

        # isaacgym_examples environments actually return dicts
        if obs_key == "obs" or obs_key == "observations":
            self.observation_space = gym.spaces.Dict(convert_space(self.env.observation_space))
        else:
            raise ValueError(f"Unknown observation key: {obs_key}")

        self._truncated: Tensor = torch.zeros(self.num_agents, dtype=torch.bool)

    def reset(self, *args, **kwargs) -> Tuple[Dict[str, Tensor], Dict]:
        # some IGE envs return all zeros on the first timestep, but this is probably okay
        obs, rew, terminated, truncated, infos = self.env.reset()
        return obs, infos

    def step(self, action) -> Tuple[Dict[str, Tensor], Tensor, Tensor, Tensor, Dict]:
        obs, rew, terminated, truncated, infos = self.env.step(action)
        return obs, rew, terminated, truncated, infos

    def render(self):
        pass

def make_aerialgym_env(
    full_task_name: str,
    cfg: Config,
    _env_config=None,
    render_mode: Optional[str] = None,
) -> Env:
    

    return AerialGymVecEnv(task_registry.make_task(task_name=full_task_name), "obs")

def add_extra_params_func(parser):
    """
    Specify any additional command line arguments for this family of custom environments.
    """
    p = parser
    p.add_argument(
        "--env_agents",
        default=-1,
        type=int,
        help="Num agents in each env (default: -1, means use default value from isaacgymenvs env yaml config file)",
    )
    p.add_argument(
        "--obs_key",
        default="obs",
        type=str,
        help='IsaacGym envs return dicts, some envs return just "obs", and some return "obs" and "states".'
        "States key denotes the full state of the environment, and obs key corresponds to limited observations "
        'available in real world deployment. If we use "states" here we can train will full information '
        "(although the original idea was to use asymmetric training - critic sees full state and policy only sees obs).",
    )
    p.add_argument(
        "--subtask",
        default=None,
        type=str,
        help="Subtask for envs that support it (i.e. AllegroKuka regrasping or manipulation or throw).",
    )
    p.add_argument(
        "--ige_api_version",
        default="preview4",
        type=str,
        choices=["preview3", "preview4"],
        help="We can switch between different versions of IsaacGymEnvs API using this parameter.",
    )
    p.add_argument(
        "--eval_stats",
        default=False,
        type=str2bool,
        help="Whether to collect env stats during evaluation.",
    )


def override_default_params_func(env, parser, env_configs):
    """Most of these parameters are taken from IsaacGymEnvs default config files."""

    parser.set_defaults(
        # we're using a single very vectorized env, no need to parallelize it further
        batched_sampling=True,
        num_workers=1,
        num_envs_per_worker=1,
        worker_num_splits=1,
        actor_worker_gpus=[0],  # obviously need a GPU
        train_for_env_steps= 10_000_000_000, #1_109_245_952
        train_for_seconds=1, # training time parameter. Stop training after train_for_seconds seconds.
        use_rnn=False,
        adaptive_stddev=False,
        policy_initialization="torch_default",
        env_gpu_actions=True,
        reward_scale=0.1,
        rollout=16,
        max_grad_norm=0.0,
        batch_size=8192*16, #2048,#32768
        num_batches_per_epoch=4,
        num_epochs=1,
        ppo_clip_ratio=0.05,
        value_loss_coeff=2.0,
        exploration_loss_coeff=0.003, #0.003
        nonlinearity="tanh",
        learning_rate=5e-3,
        lr_schedule="kl_adaptive_epoch",
        lr_schedule_kl_threshold=0.008,
        lr_adaptive_min = 1e-6,
        lr_adaptive_max = 1e-2,
        shuffle_minibatches=False,
        gamma=0.98,
        gae_lambda=0.95,
        with_vtrace=False,
        value_bootstrap=True,  # assuming reward from the last step in the episode can generally be ignored
        normalize_input=False,
        normalize_returns=True,  # does not improve results on all envs, but with return normalization we don't need to tune reward scale
        serial_mode=True,  # it makes sense to run isaacgym envs in serial mode since most of the parallelism comes from the env itself (although async mode works!)
        async_rl=False,
        use_env_info_cache=False, # speeds up startup
        kl_loss_coeff=0.1,
        restart_behavior="overwrite",
        load_checkpoint_kind = "best",
        save_every_sec = 20,
        save_best_after = 30_000_001, # make sure to save only after curriculum learning phase is over.
        save_best_every_sec = 20,
        continuous_tanh_scale = 1.0,
        initial_stddev = 0.4,
        actor_critic_share_weights = False,
        seed = 42,
    )

    # override default config parameters for specific envs
    if env in env_configs:
        parser.set_defaults(**env_configs[env])

env_configs = dict(
            position_setpoint_task_sim2real_end_to_end=dict(
            train_for_env_steps=160_000_000,
            #encoder_mlp_layers=[256, 128, 64],
            encoder_mlp_layers=[32, 24],#encoder_mlp_layers=[32, 24], #64, 52, 32
            gamma=0.99,
            rollout=32,
            learning_rate=3e-4,
            lr_schedule_kl_threshold=0.016,
            batch_size=4096*16,
            num_epochs=5,
            max_grad_norm=1.0,
            num_batches_per_epoch=2,
            exploration_loss_coeff=1e-2,
            with_wandb=False,
            wandb_project="gen_aerial_robot",
            wandb_user="welfrehberg",
            adaptive_stddev=False,
            continuous_tanh_scale=1.0,
            seed=42,
        ),
    )

def create_new_task():
    register_aerialgym_custom_components(env_configs)
    cfg = parse_aerialgym_cfg(env_configs)
    return cfg

def register_aerialgym_custom_components(env_learning_configs):
    for env_name in env_learning_configs:
        register_env(env_name, make_aerialgym_env)

def parse_aerialgym_cfg(evaluation=False):
    parser, partial_cfg = parse_sf_args(evaluation=evaluation)
    add_extra_params_func(parser)
    override_default_params_func(partial_cfg.env, parser, env_configs)
    final_cfg = parse_full_cfg(parser)
    return final_cfg


