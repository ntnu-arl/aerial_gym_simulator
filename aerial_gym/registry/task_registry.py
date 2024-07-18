class TaskRegistry:
    def __init__(self):
        self.task_class_registry = {}
        self.task_config_registry = {}

    def register_task(self, task_name, task_class, task_config):
        self.task_class_registry[task_name] = task_class
        self.task_config_registry[task_name] = task_config

    def get_task_class(self, task_name):
        return self.task_class_registry[task_name]

    def get_task_config(self, task_name):
        return self.task_config_registry[task_name]

    def get_task_names(self):
        return list(self.task_class_registry.keys())

    def get_task_classes(self):
        return list(self.task_class_registry.values())

    def get_task_configs(self):
        return list(self.task_config_registry.values())

    def make_task(self, task_name, seed=None, num_envs=None, headless=None, use_warp=None):
        task_class = self.get_task_class(task_name)
        task_config = self.get_task_config(task_name)
        return task_class(
            task_config, seed=seed, num_envs=num_envs, headless=headless, use_warp=use_warp
        )


task_registry = TaskRegistry()
