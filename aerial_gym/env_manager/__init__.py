from aerial_gym.config.env_config.env_with_obstacles import EnvWithObstaclesCfg
from aerial_gym.config.env_config.empty_env import EmptyEnvCfg
from aerial_gym.config.env_config.forest_env import ForestEnvCfg

from aerial_gym.registry.env_registry import env_config_registry

env_config_registry.register("env_with_obstacles", EnvWithObstaclesCfg)
env_config_registry.register("empty_env", EmptyEnvCfg)
env_config_registry.register("forest_env", ForestEnvCfg)
