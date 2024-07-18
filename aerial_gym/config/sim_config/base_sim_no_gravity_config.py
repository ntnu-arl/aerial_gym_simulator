from aerial_gym.config.sim_config.base_sim_config import BaseSimConfig


class BaseSimNoGravityConfig(BaseSimConfig):
    class sim(BaseSimConfig.sim):
        gravity = [0.0, 0.0, 0.0]
