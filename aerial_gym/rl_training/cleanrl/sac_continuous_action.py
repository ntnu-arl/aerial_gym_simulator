# SAC algorithm integrated with Isaac Gym for aerial robots
# This script includes:
# 1. Argument parsing.
# 2. Replay buffer implementation.
# 3. Neural network (Actor, Critic, and Value) implementation with weight initialization.
# 4. SAC algorithm logic.
# 5. Environment setup and main training loop.

# The script can be run using the following command: 
# docs and experiment results can be found at  https://arxiv.org/abs/1801.01290
import os
import random
import time
import copy

import gym
import isaacgym  # noqa
from isaacgym import gymutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from aerial_gym.envs import *
from aerial_gym.utils import task_registry


def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "quad", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--experiment_name", "type": str, "default": os.path.basename(__file__).rstrip(".py"), "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--checkpoint", "type": str, "default": None, "help": "Saved model checkpoint number."},        
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "default": 512, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "default": 1, "help": "Random seed. Overrides config file if provided."},
        {"name": "--play", "required": False, "help": "only run network", "action": 'store_true'},

        {"name": "--torch-deterministic-off", "action": "store_true", "default": True, "help": "if toggled, `torch.backends.cudnn.deterministic=False`"}, # it was False for the ppo implimentation 

        {"name": "--track", "action": "store_true", "default": True,"help": "if toggled, this experiment will be tracked with Weights and Biases"},
        {"name": "--wandb-project-name", "type":str, "default": "cleanRL", "help": "the wandb's project name"},
        {"name": "--wandb-entity", "type":str, "default": None, "help": "the entity (team) of wandb's project"},

        # Algorithm specific arguments
        {"name": "--total-timesteps", "type":int, "default": 30000000,
            "help": "total timesteps of the experiments"},
        {"name": "--learning-rate", "type":float, "default": 0.0026,
            "help": "the learning rate of the optimizer"},
        {"name": "--num-steps", "type":int, "default": 16,
            "help": "the number of steps to run in each environment per policy rollout"},
        {"name": "--gamma", "type":float, "default": 0.99,
            "help": "the discount factor gamma"},
        {"name": "--tau", "type":float, "default": 0.005,
            "help": "the tau for soft update of the target network"},
        {"name": "--batch-size", "type":int, "default": 256,
            "help": "the batch size for training"},
        {"name": "--alpha", "type":float, "default": 0.2,
            "help": "Entropy regularization coefficient."},
        {"name": "--adaptive-alpha", "action": "store_true", "default": False,
            "help": "Toggle adaptive entropy regularization."},   
        ]

    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)
    
    args.batch_size = int(args.num_envs * args.num_steps)

    args.torch_deterministic = not args.torch_deterministic_off

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size, device):
        self.max_size = max_size
        self.device = device
        self.ptr = 0
        self.size = 0

        self.s = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
        self.a = torch.zeros((max_size, action_dim), dtype=torch.float32, device=device)
        self.r = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
        self.s_next = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
        self.dw = torch.zeros((max_size, 1), dtype=torch.bool, device=device)

    def add(self, s, a, r, s_next, dw):
        self.s[self.ptr] = torch.tensor(s, dtype=torch.float32, device=self.device)
        self.a[self.ptr] = torch.tensor(a, dtype=torch.float32, device=self.device)
        self.r[self.ptr] = torch.tensor(r, dtype=torch.float32, device=self.device)
        self.s_next[self.ptr] = torch.tensor(s_next, dtype=torch.float32, device=self.device)
        self.dw[self.ptr] = torch.tensor(dw, dtype=torch.bool, device=self.device)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, size=(batch_size,), device=self.device)
        return (
            self.s[ind],
            self.a[ind],
            self.r[ind],
            self.s_next[ind],
            self.dw[ind],
        )

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.num_obs).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, np.prod(envs.num_actions)), std=1.0),#0.01
        )

        self.critic_1 = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.num_obs).prod() + np.prod(envs.num_actions), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.critic_2 = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.num_obs).prod() + np.prod(envs.num_actions), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def get_action(self, x, deterministic=False,reparametrization =True):
        mean = self.actor(x)
        if deterministic:
            return mean
        log_std = -0.5 * torch.ones_like(mean)  # Assuming fixed log_std
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std) # we learn log_std rather than std, so that exp(log_std) is always > 0
        normal = Normal(mean, std)
        if reparametrization:
            return normal.rsample()# for determnstic case with reparammetrization 
        else:
            return normal.sample()

    def get_action_and_value(self, x, action=None):
        mean = self.actor(x)
        log_std = -0.5 * torch.ones_like(mean)  # Assuming fixed log_std
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        normal = Normal(mean, std)
        if action is None:
            action = normal.rsample() # it was sample()
        log_prob = normal.log_prob(action).sum(-1)
        q_value_1 = self.critic_1(torch.cat([x, action], -1))
        q_value_2 = self.critic_2(torch.cat([x, action], -1))

        return action, log_prob, q_value_1 ,q_value_2


if __name__ == "__main__":
    args = get_args()

    run_name = f"{args.task}__{args.experiment_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = args.rl_device
    print("using device:", device)

    # env setup
    envs, env_cfg = task_registry.make_env(name="quad", args=args)

    #envs = RecordEpisodeStatisticsTorch(envs, device)  # a wrapper for recording episodic returns

    print("num actions: ", envs.num_actions)
    print("num obs: ", envs.num_obs)

    agent = Agent(envs).to(device)
    replay_buffer = ReplayBuffer(envs.num_obs, envs.num_actions, max_size=int(1e6), device=device)
    actor_optimizer = optim.Adam(agent.actor.parameters(), lr=args.learning_rate)
    critic_1_optimizer = optim.Adam(agent.critic_1.parameters(), lr=args.learning_rate)
    critic_2_optimizer = optim.Adam(agent.critic_2.parameters(), lr=args.learning_rate)

    # Adaptive alpha (temprature) setup 
    if args.adaptive_alpha:
        target_entopy = -np.prod(envs.num_actions)
        log_alpha = torch.tensor(np.log(args.alpha),dtype=torch.float32,device=device ,requires_grad=True)
        alpha_optimizer =optim.Adam([log_alpha],lr = args.learning_rate)
    else:
        log_alpha = torch.tensor(np.log(args.alpha), dtype=torch.float32 , device = device)
    alpha = log_alpha.exp()

    if args.play and args.checkpoint is None:
        raise ValueError("No checkpoint provided for testing.")

    # load checkpoint if needed
    if args.checkpoint is not None:
        print("Loading checkpoint...")
        checkpoint = torch.load(args.checkpoint)
        agent.load_state_dict(checkpoint)
        print("Loaded checkpoint")

    global_step = 0
    start_time = time.time()
    next_obs, _info = envs.reset()
    next_done = torch.zeros(args.num_envs, dtype=torch.float32).to(device)

    if not args.play:
        for update in range(1, args.total_timesteps // args.batch_size + 1):
            for step in range(args.num_steps):
                global_step += 1 * args.num_envs
                obs_tensor = torch.tensor(next_obs, dtype=torch.float32).to(device)

                with torch.no_grad():
                    action = agent.get_action(obs_tensor)
                #action_np = action.cpu().numpy()
                action_np = action

                next_obs, rewards, next_done, infos ,next_info= envs.step(action_np)
                replay_buffer.add(next_obs, action_np, rewards, next_obs, next_done)

                if global_step % args.batch_size == 0:
                    for _ in range(args.num_steps):
                        s, a, r, s_next, dw = replay_buffer.sample(args.batch_size)

                        with torch.no_grad():
                            next_action, next_log_prob, next_q_value_1 ,next_q_value_2= agent.get_action_and_value(s_next)
                            min_next_value = torch.min(next_q_value_1, next_q_value_2)
                            target_value = r + args.gamma * (1 - dw) * (min_next_value - alpha * next_log_prob)

                        current_value_1 = agent.critic_1(torch.cat([s, a], -1))
                        current_value_2 = agent.critic_2(torch.cat([s, a], -1))
                        critic_1_loss = F.mse_loss(current_value_1, target_value) 
                        critic_2_loss = F.mse_loss(current_value_2, target_value)

                        critic_1_optimizer.zero_grad()
                        critic_1_loss.backward()
                        critic_1_optimizer.step()

                        critic_2_optimizer.zero_grad()
                        critic_2_loss.backward()
                        critic_2_optimizer.step()

                        # Actor Update (Update policy by one step of gradient ascent)
                        actions,log_prob, q_value_1,q_value_2 = agent.get_action_and_value(s)
                        min_q_value= torch.min(q_value_1 ,q_value_2)
                        actor_loss = (alpha * next_log_prob - min_q_value).mean()

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()

                        # Alpha (temprature) update 
                        if args.adaptive_alpha:
                            alpha_loss = - (log_alpha * (log_prob + target_entopy).detach()).mean()
                            alpha_optimizer.zero_grad()
                            alpha_loss.backward()
                            alpha_optimizer.step()
                            alpa = log_alpha.exp()

                        # Sof update of target networks

                        soft_update(agent.target_critic_1, agent.critic_1, args.tau)
                        soft_update(agent.target_critic_2, agent.critic_2, args.tau)

            writer.add_scalar("charts/learning_rate", actor_optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/critic_1_loss", critic_1_loss.item(), global_step)
            writer.add_scalar("losses/critic_2_loss", critic_2_loss.item(), global_step)
            writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)

            if args.adaptive_alpha:
                writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
                writer.add_scalar("charts/alpha", alpha.item(), global_step)

            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            if update % 50 == 0:
                print("Saving model.")
                torch.save(agent.state_dict(), f"runs/{run_name}/latest_model.pth")

    else:
        for step in range(0, 5000000):
            with torch.no_grad():
                action, logprob, _, q_value_1, q_value_2 = agent.get_action_and_value(next_obs)
            next_obs, rewards, next_done, info = envs.step(action)


    writer.close()
