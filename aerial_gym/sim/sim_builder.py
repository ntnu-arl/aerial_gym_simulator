from aerial_gym.env_manager.env_manager import EnvManager

import torch


class SimBuilder:
    def __init__(self):
        self.sim_name = None
        self.env_name = None
        self.robot_name = None
        self.env = None
        pass

    def delete_env(self):
        # garbage cleanup for the environment
        del self.env
        # make sure all cuda memory is freed
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.env = None

    def build_env(
        self,
        sim_name,
        env_name,
        robot_name,
        controller_name,
        device,
        args=None,
        num_envs=None,
        use_warp=None,
        headless=None,
    ):
        self.sim_name = sim_name
        self.env_name = env_name
        self.robot_name = robot_name
        self.env = EnvManager(
            sim_name=sim_name,
            env_name=env_name,
            robot_name=robot_name,
            controller_name=controller_name,
            args=args,
            device=device,
            num_envs=num_envs,
            use_warp=use_warp,
            headless=headless,
        )
        return self.env
