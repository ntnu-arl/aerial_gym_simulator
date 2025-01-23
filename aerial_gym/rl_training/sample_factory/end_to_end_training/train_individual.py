# this is here just to guarantee that isaacgym is imported before PyTorch
# isort: off
# noinspection PyUnresolvedReferences
import isaacgym

# isort: on

import os
import sys
from os.path import join

from sample_factory.train import run_rl
from aerial_gym.rl_training.sample_factory.end_to_end_training.helper import create_new_task

import torch

print(torch.cuda.is_available())
print(torch.__version__)


def train_individual():
    
    cfg = create_new_task()
        
    status = run_rl(cfg)
    return status

if __name__ == "__main__":
    sys.exit(train_individual())
