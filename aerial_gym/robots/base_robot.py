from abc import ABC, abstractmethod


from aerial_gym.registry.controller_registry import controller_registry
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("BaseRobot")


class BaseRobot(ABC):
    """
    Base class for the aerial robot. This class should be inherited by the specific robot class.

    """

    def __init__(self, robot_config, controller_name, env_config, device):
        self.cfg = robot_config
        self.num_envs = env_config.env.num_envs
        self.device = device

        self.controller, self.controller_config = controller_registry.make_controller(
            controller_name,
            self.num_envs,
            self.device,
        )
        logger.info("[DONE] Initializing controller")

        # Initialize the controller
        logger.info(f"Initializing controller {controller_name}")
        self.controller_config = controller_registry.get_controller_config(controller_name)
        if controller_name == "no_control":
            self.controller_config.num_actions = self.cfg.control_allocator_config.num_motors

        self.num_actions = self.controller_config.num_actions

    @abstractmethod
    def init_tensors(self, global_tensor_dict):
        self.dt = global_tensor_dict["dt"]
        self.gravity = global_tensor_dict["gravity"]
        self.robot_state = global_tensor_dict["robot_state_tensor"]
        self.robot_position = global_tensor_dict["robot_position"]
        self.robot_orientation = global_tensor_dict["robot_orientation"]
        self.robot_linvel = global_tensor_dict["robot_linvel"]
        self.robot_angvel = global_tensor_dict["robot_angvel"]

        # tensors for robot forces and torques
        self.robot_force_tensors = global_tensor_dict["robot_force_tensor"]
        self.robot_torque_tensors = global_tensor_dict["robot_torque_tensor"]

        self.env_bounds_min = global_tensor_dict["env_bounds_min"]
        self.env_bounds_max = global_tensor_dict["env_bounds_max"]

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def reset_idx(self, env_ids):
        pass

    @abstractmethod
    def step(self):
        pass

    # @abstractmethod
    # def apply_noise(self):
    #     pass

    # @abstractmethod
    # def get_state(self):
    #     pass

    # @abstractmethod
    # def set_state(self, state):
    #     pass
