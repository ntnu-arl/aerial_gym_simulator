from aerial_gym.utils.logging import CustomLogger
from aerial_gym.control.control_allocation import ControlAllocator

logger = CustomLogger("no_control")


class NoControl:
    def __init__(self, config, num_envs, device):
        pass

    def init_tensors(self, global_tensor_dict=None):
        pass

    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)

    def reset_commands(self):
        pass

    def reset(self):
        self.reset_idx(env_ids=None)

    def reset_idx(self, env_ids):
        pass

    def randomize_params(self, env_ids):
        pass

    def update(self, command_actions):
        return command_actions
