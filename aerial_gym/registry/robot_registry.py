from aerial_gym.registry.controller_registry import controller_registry


class RobotRegistry:
    """
    This class is used to keep track of the robots that are created.
    New robots can be added to the registry and can be accessed by other classes.
    This will allow the environment manager to create the robots and the robot manager
    to access the robots.
    """

    def __init__(self) -> None:
        self.robot_classes = {}
        self.robot_configs = {}

    def register(self, robot_name, robot_class, robot_config):
        """
        Add a robot to the robot dictionary.
        """
        self.robot_classes[robot_name] = robot_class
        self.robot_configs[robot_name] = robot_config

    def get_robot_class(self, robot_name):
        """
        Get a robot from the robot dictionary.
        """
        return self.robot_classes[robot_name]

    def get_robot_config(self, robot_name):
        """
        Get a robot config from the robot dictionary.
        """
        return self.robot_configs[robot_name]

    def get_robot_names(self):
        """
        Get the robot names from the robot dictionary.
        """
        return self.robot_classes.keys()

    def make_robot(self, robot_name, controller_name, env_config, device):
        """
        Make a robot from the robot dictionary.
        """
        if robot_name not in self.robot_classes:
            raise ValueError(f"Robot {robot_name} not found in robot registry")
        return (
            self.robot_classes[robot_name](
                self.robot_configs[robot_name], controller_name, env_config, device
            ),
            self.robot_configs[robot_name],
        )


# create a global robot registry
robot_registry = RobotRegistry()
