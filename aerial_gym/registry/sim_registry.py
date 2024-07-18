class SimConfigRegistry:
    """
    This class is used to keep track of the sim config classes that are registered.
    New sim configurations can be added to the registry and can be accessed by other classes.
    """

    def __init__(self) -> None:
        self.sim_configs = {}

    def register(self, sim_name, sim_config):
        """
        Add a sim to the sim dictionary.
        """
        self.sim_configs[sim_name] = sim_config

    def get_sim_config(self, sim_name):
        """
        Get a sim from the sim dictionary.
        """
        return self.sim_configs[sim_name]

    def get_sim_names(self):
        """
        Get the sim names from the sim dictionary.
        """
        return self.sim_configs.keys()

    def make_sim(self, sim_name):
        """
        Make a sim from the sim dictionary.
        """
        if sim_name not in self.sim_configs:
            raise ValueError(f"sim {sim_name} not found in sim registry")
        return self.sim_configs[sim_name]


# create a global sim registry
sim_config_registry = SimConfigRegistry()
