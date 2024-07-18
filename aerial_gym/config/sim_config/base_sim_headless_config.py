from .base_sim_config import BaseSimConfig


class BaseSimHeadlessConfig(BaseSimConfig):
    class viewer(BaseSimConfig.viewer):
        headless = True
