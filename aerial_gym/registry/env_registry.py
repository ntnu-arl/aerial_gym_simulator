class EnvConfigRegistry:
    """
    This class is used to keep track of the environment classes that are registered.
    New environment configurations can be added to the registry and can be accessed by other classes.
    """

    def __init__(self) -> None:
        self.env_configs = {}

    def register(self, env_name, env_config):
        """
        Add a env to the env dictionary.
        """
        self.env_configs[env_name] = env_config

    def get_env_config(self, env_name):
        """
        Get a env from the env dictionary.
        """
        return self.env_configs[env_name]

    def get_env_names(self):
        """
        Get the env names from the env dictionary.
        """
        return self.env_configs.keys()

    def make_env(self, env_name):
        """
        Make a env from the env dictionary.
        """
        if env_name not in self.env_configs:
            raise ValueError(f"env {env_name} not found in env registry")
        return self.env_configs[env_name]


# create a global env registry
env_config_registry = EnvConfigRegistry()
