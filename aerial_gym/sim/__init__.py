from aerial_gym.registry.sim_registry import sim_config_registry

from aerial_gym.config.sim_config.base_sim_config import BaseSimConfig

from aerial_gym.config.sim_config.base_sim_headless_config import (
    BaseSimHeadlessConfig,
)

from aerial_gym.config.sim_config.sim_config_2ms import SimCfg2Ms
from aerial_gym.config.sim_config.sim_config_4ms import SimCfg4Ms

sim_config_registry.register("base_sim", BaseSimConfig)
sim_config_registry.register("base_sim_headless", BaseSimHeadlessConfig)
sim_config_registry.register("base_sim_2ms", SimCfg2Ms)
sim_config_registry.register("base_sim_4ms", SimCfg4Ms)

# Uncomment the following lines to register your custom sim config

# from aerial_gym.config.sim_config.custom_sim_config import CustomSimConfig
# sim_config_registry.register("custom_sim", CustomSimConfig)
