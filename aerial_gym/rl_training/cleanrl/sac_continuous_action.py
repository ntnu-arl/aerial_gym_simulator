# SAC algorithm integrated with Isaac Gym for aerial robots
# This script includes:
# 1. Argument parsing.
# 2. Replay buffer implementation.
# 3. Neural network (Actor, Critic, and Value) implementation with weight initialization.
# 4. SAC algorithm logic.
# 5. Environment setup and main training loop.

import os
import random
import time

import gym
import isaacgym  # noqa
from isaacgym import gymutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
        {"name": "--torch-deterministic-off", "action": "store_true", "default": False, "help": "if toggled, `torch.backends.cudnn.deterministic=False`"},
        {"name": "--track", "action": "store_true", "default": False, "help": "if toggled, this experiment will be tracked with Weights and Biases"},
        {"name": "--wandb-project-name", "type": str, "default": "cleanRL", "help": "the wandb's project name"},
        {"name": "--wandb-entity", "type": str, "default": None, "help": "the entity (team) of wandb's project"},
        # Algorithm specific arguments
        {"name": "--total-timesteps", "type": int, "default": 30000000, "help": "total timesteps of the experiments"},
        {"name": "--learning-rate", "type": float, "default": 0.0026, "help": "the learning rate of the optimizer"},
        {"name": "--num-steps", "type": int, "default": 16, "help": "the number of steps to run in each environment per policy rollout"},
        {"name": "--anneal-lr", "action": "store_true", "default": False, "help": "Toggle learning rate annealing for policy and value networks"},
        {"name": "--gamma", "type": float, "default": 0.99, "help": "the discount factor gamma"},
        {"name": "--gae-lambda", "type": float, "default": 0.95, "help": "the lambda for the general advantage estimation"},
        {"name": "--num-minibatches", "type": int, "default": 2, "help": "the number of mini-batches"},
        {"name": "--update-epochs", "type": int, "default": 4, "help": "the K epochs to update the policy"},
        {"name": "--norm-adv-off", "action": "store_true", "default": False, "help": "Toggles advantages normalization"},
        {"name": "--clip-coef", "type": float, "default": 0.2, "help": "the surrogate clipping coefficient"},
        {"name": "--clip-vloss", "action": "store_true", "default": False, "help": "Toggles whether or not to use a clipped loss for the value function, as per the paper."},
        {"name": "--ent-coef", "type": float, "default": 0.0, "help": "coefficient of the entropy"},
        {"name": "--vf-coef", "type": float, "default": 2, "help": "coefficient of the value function"},
        {"name": "--max-grad-norm", "type": float, "default": 1, "help": "the maximum norm for the gradient clipping"},
        {"name": "--target-kl", "type": float, "default": None, "help": "the target KL divergence threshold"},
    ]

    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    args.torch_deterministic = not args.torch_deterministic_off
    args.norm_adv = not args.norm_adv_off

    # name alignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == 'cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.done[ind]).to(device)
        )


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = layer_init(nn.Linear(state_dim, 256))
        self.l2 = layer_init(nn.Linear(256, 256))
        self.mean = layer_init(nn.Linear(256, action_dim), std=0.01)
        self.log_std = layer_init(nn.Linear(256, action_dim), std=0.01)
        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        mean = self.mean(a)
        log_std = self.log_std(a).clamp(-20, 2)
        std = log_std.exp()
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        normal = Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.l1 = layer_init(nn.Linear(state_dim + action_dim, 256))
        self.l2 = layer_init(nn.Linear(256, 256))
        self.l3 = layer_init(nn.Linear(256, 1))

        # Q2 architecture
        self.l4 = layer_init(nn.Linear(state_dim + action_dim, 256))
        self.l5 = layer_init(nn.Linear(256, 256))
        self.l6 = layer_init(nn.Linear(256, 1))

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = torch.relu(self.l1(sa))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = torch.relu(self.l4(sa))
        q2 = torch.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


class Value(nn.Module):
    def __init__(self, state_dim):
        super(Value, self).__init__()
        self.l1 = layer_init(nn.Linear(state_dim, 256))
        self.l2 = layer_init(nn.Linear(256, 256))
        self.l3 = layer_init(nn.Linear(256, 1))

    def forward(self, state):
        v = torch.relu(self.l1(state))
        v = torch.relu(self.l2(v))
        return self.l3(v)


class SAC:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        alpha=0.2,
        policy_freq=2
    ):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.learning_rate)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.learning_rate)

        self.value = Value(state_dim).to(device)
        self.value_target = Value(state_dim).to(device)
        self.value_target.load_state_dict(self.value.state_dict())
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=args.learning_rate)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.alpha = alpha
        self.policy_freq = policy_freq
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action, _, _ = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Compute the target Q value
            next_action, log_prob, _ = self.actor.sample(next_state)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_V = self.value_target(next_state)
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha * log_prob
            target_Q = reward + ((1 - done) * self.discount * target_V)

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute value loss
        value = self.value(state)
        value_loss = nn.MSELoss()(value, target_Q)

        # Optimize the value
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            action, log_prob, _ = self.actor.sample(state)
            q1, q2 = self.critic(state, action)
            actor_loss = (self.alpha * log_prob - torch.min(q1, q2)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.value.parameters(), self.value_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.value.state_dict(), filename + "_value")
        torch.save(self.value_optimizer.state_dict(), filename + "_value_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.value.load_state_dict(torch.load(filename + "_value"))
        self.value_optimizer.load_state_dict(torch.load(filename + "_value_optimizer"))


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

    device = torch.device(args.rl_device)
    # env setup
    envs = gym.vector.AsyncVectorEnv(
        [task_registry.make_env(args.task, sim_device=args.sim_device, graphics_device_id=args.graphics_device_id, headless=args.headless)] * args.num_envs
    )

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    assert isinstance(envs.single_observation_space, gym.spaces.Box), "only continuous observation space is supported"

    max_action = float(envs.single_action_space.high[0])
    action_shape = envs.single_action_space.shape
    obs_shape = envs.single_observation_space.shape

    sac = SAC(obs_shape[0], action_shape[0], max_action)

    replay_buffer = ReplayBuffer(obs_shape[0], action_shape[0])

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.batch_size:
            action = np.array([envs.single_action_space.sample() for _ in range(args.num_envs)])
        else:
            action = np.array([sac.select_action(obs[i]) for i in range(args.num_envs)])

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, done, info = envs.step(action)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in info:
            for item in info["final_info"]:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    break

        # ALGO LOGIC: training.
        # `next_obs`, `reward`, `done` should be converted to float32 arrays for sac's replay buffer
        replay_buffer.add(obs, action, next_obs, reward, done)
        obs = next_obs

        if global_step >= args.batch_size:
            sac.train(replay_buffer, args.batch_size)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        if done:
            obs, _ = envs.reset()

    envs.close()
    writer.close()
