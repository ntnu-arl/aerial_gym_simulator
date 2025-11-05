import isaacgym

from aerial_gym.config.env_config.dynamic_environment import DynamicEnvironmentCfg
from aerial_gym.config.env_config.empty_env import EmptyEnvCfg
from aerial_gym.config.env_config.env_config_2ms import EnvCfg2Ms
from aerial_gym.config.env_config.env_with_obstacles import EnvWithObstaclesCfg
from aerial_gym.config.env_config.forest_env import ForestEnvCfg
from aerial_gym.config.env_config.procedural_forest_env import ProceduralForestEnvCfg
from aerial_gym.registry.env_registry import env_config_registry

env_config_registry.register("env_with_obstacles", EnvWithObstaclesCfg)
env_config_registry.register("empty_env", EmptyEnvCfg)
env_config_registry.register("forest_env", ForestEnvCfg)
env_config_registry.register("empty_env_2ms", EnvCfg2Ms)
env_config_registry.register("dynamic_env", DynamicEnvironmentCfg)
env_config_registry.register("procedural_forest", ProceduralForestEnvCfg)
