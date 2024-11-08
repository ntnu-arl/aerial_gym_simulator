from .nn_inference_class import Sim2RealInferenceClass
import torch


SAMPLE_FROM_LATENT = False
device = "cuda:0"
TASK = "navigation_task"

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args

# custom default configuration parameters for specific envs
# add more envs here analogously (env names should match config file names in IGE)
env_configs = dict(
    navigation_task=dict(
        train_for_env_steps=131000000000,
        encoder_mlp_layers=[256, 128, 64],
        gamma=0.98,
        rollout=32,
        learning_rate=1e-4,
        lr_schedule_kl_threshold=0.016,
        batch_size=2048,
        num_epochs=4,
        max_grad_norm=1.0,
        num_batches_per_epoch=4,
        exploration_loss_coeff=0.0,
        with_wandb=False,
        wandb_project="quad",
        wandb_user="mihirkulkarni",
    ),
)


def override_default_params_func(env, parser):
    """Most of these parameters are taken from IsaacGymEnvs default config files."""

    parser.set_defaults(
        # we're using a single very vectorized env, no need to parallelize it further
        batched_sampling=True,
        num_workers=1,
        num_envs_per_worker=1,
        worker_num_splits=1,
        actor_worker_gpus=[0],  # obviously need a GPU
        train_for_env_steps=10000000,
        use_rnn=False,
        adaptive_stddev=True,
        policy_initialization="torch_default",
        env_gpu_actions=True,
        reward_scale=0.1,
        rollout=24,
        max_grad_norm=0.0,
        batch_size=2048,
        num_batches_per_epoch=2,
        num_epochs=4,
        ppo_clip_ratio=0.2,
        value_loss_coeff=2.0,
        exploration_loss_coeff=0.0,
        nonlinearity="elu",
        learning_rate=3e-4,
        lr_schedule="kl_adaptive_epoch",
        lr_schedule_kl_threshold=0.016,
        shuffle_minibatches=True,
        gamma=0.98,
        gae_lambda=0.95,
        with_vtrace=False,
        value_bootstrap=True,  # assuming reward from the last step in the episode can generally be ignored
        normalize_input=True,
        normalize_returns=True,  # does not improve results on all envs, but with return normalization we don't need to tune reward scale
        save_best_after=int(5e6),
        serial_mode=True,  # it makes sense to run isaacgym envs in serial mode since most of the parallelism comes from the env itself (although async mode works!)
        async_rl=True,
        use_env_info_cache=False,  # speeds up startup
        kl_loss_coeff=0.1,
        restart_behavior="overwrite",
        eval_deterministic=True,
    )

    # override default config parameters for specific envs
    if env in env_configs:
        parser.set_defaults(**env_configs[env])


def parse_aerialgym_cfg(evaluation=True):
    parser, partial_cfg = parse_sf_args(evaluation=evaluation)
    override_default_params_func(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser)
    return final_cfg


def get_network(num_envs):
    """Script entry point."""
    # register_aerialgym_custom_components()
    cfg = parse_aerialgym_cfg(evaluation=True)
    print("CFG is:", cfg)
    nn_model = Sim2RealInferenceClass(num_envs, 4, 81, cfg)
    return nn_model


class RL_Nav_Interface:
    def __init__(self, num_envs=1):
        # Initialize networks
        self.device = device
        self.RL_net_interface = get_network(num_envs)

    def step(self, obs):
        # Get action from RL network
        return self.RL_net_interface.get_action(obs)

    def reset(self, env_ids=[0]):
        self.RL_net_interface.reset(env_ids)


if __name__ == "__main__":
    nav_interface = RL_Nav_Interface()
    print("Loaded weights. Ready for inference")
    obs = torch.zeros(1, 81)
    obs_dict = {"observations": obs}
    action = nav_interface.step(obs_dict)
    print("Action: ", action)
