import time
from aerial_gym.utils.logging import CustomLogger

from aerial_gym.sim2real.sample_factory_inference import RL_Nav_Interface

logger = CustomLogger(__name__)
from aerial_gym.registry.task_registry import task_registry
import torch


class EMA:
    def __init__(self, beta):
        self.beta = beta
        self.average = None

    def update(self, value):
        if self.average is None:
            self.average = value
        else:
            self.average = (1 - self.beta) * self.average + self.beta * value
        return self.average


num_envs = 16

if __name__ == "__main__":
    start = time.time()
    rl_task_env = task_registry.make_task("navigation_task", headless=False, num_envs=num_envs)


    logger.warning(
        "\n\nExample file simulating a Sample Factory trained policy in cluttered environments using a Task Definition for navigation."
    )
    logger.warning(
        "Usage: python3 inference_example.py --env=navigation_task --experiment=lmf2_sim2real_241024 --train_dir=../sim2real/weights --load_checkpoint_kind=best"
    )
    logger.warning(
        "Please make sure a camera sensor is enabled on the robot as per specifications of the task.\n\n"
    )
    rl_model = RL_Nav_Interface(num_envs=num_envs)
    action_filter = EMA(0.8)

    rl_task_env.reset()
    actions = torch.zeros(
        (rl_task_env.task_config.num_envs, rl_task_env.task_config.action_space_dim)
    ).to("cuda:0")
    actions[:, 0] = -1.0
    with torch.no_grad():
        for i in range(10000):
            if i == 100:
                start = time.time()
            for j in range(5):
                obs, reward, terminated, truncated, info = rl_task_env.step(actions=actions)
            obs["observations"] = action_filter.update(obs["observations"])
            reset_list = (terminated + truncated).nonzero().squeeze().tolist()
            if ((type(reset_list) is int) and (reset_list > 0)) or len(reset_list) > 0:
                rl_model.reset(reset_list)
            actions[:] = rl_model.step(obs=obs)
    end = time.time()
