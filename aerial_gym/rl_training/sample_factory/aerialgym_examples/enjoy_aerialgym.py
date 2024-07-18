# this is here just to guarantee that isaacgym is imported before PyTorch
# isort: off
# noinspection PyUnresolvedReferences
import isaacgym

# isort: on

import sys

from sample_factory.enjoy import enjoy
from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym import (
    parse_aerialgym_cfg,
    register_aerialgym_custom_components,
)


def main():
    """Script entry point."""
    register_aerialgym_custom_components()
    cfg = parse_aerialgym_cfg(evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
