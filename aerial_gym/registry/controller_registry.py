class ControllerRegistry:
    """
    This class is used to register and get controllers for the environment.

    """

    def __init__(self) -> None:
        self.controller_classes = {}
        self.controller_configs = {}

    def register_controller(self, controller_name, controller_class, controller_config):
        """
        Add a controller to the controller dictionary.
        """
        self.controller_classes[controller_name] = controller_class
        self.controller_configs[controller_name] = controller_config

    def get_controller_class(self, controller_name):
        """
        Get a controller from the controller dictionary.
        """
        return self.controller_classes[controller_name]

    def get_controller_names(self):
        """
        Get the controller names from the controller dictionary.
        """
        return self.controller_classes.keys()

    def get_controller_config(self, controller_name):
        """
        Get the controller config from the controller dictionary.
        """
        return self.controller_configs[controller_name]

    def make_controller(self, controller_name, num_envs, device, mode="robot"):
        """
        Make a controller from the controller dictionary.
        """
        # check if it is in the registry
        if controller_name not in self.controller_classes:
            raise ValueError(
                f"Controller {controller_name} not found in controller registry. Available controllers are {self.controller_classes.keys()}"
            )
        return (
            self.controller_classes[controller_name](
                self.controller_configs[controller_name],
                num_envs,
                device,
            ),
            self.controller_configs[controller_name],
        )


controller_registry = ControllerRegistry()
