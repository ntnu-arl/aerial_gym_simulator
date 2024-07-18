from abc import ABC


class BaseManager(ABC):
    def __init__(self, config, device):
        self.cfg = config
        self.device = device

    def reset(self):
        raise NotImplementedError("reset not implemented")

    def reset_idx(self, env_ids):
        raise NotImplementedError("reset_idx not implemented")

    def pre_physics_step(self, actions):
        pass

    def step(self):
        raise NotImplementedError("step not implemented")

    def post_physics_step(self):
        pass

    def init_tensors(self, global_tensor_dict):
        pass
