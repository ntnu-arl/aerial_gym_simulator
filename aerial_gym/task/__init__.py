from aerial_gym.task.position_setpoint_task.position_setpoint_task import (
    PositionSetpointTask,
)

from aerial_gym.task.navigation_task.navigation_task import NavigationTask

from aerial_gym.config.task_config.position_setpoint_task_config import (
    task_config as position_setpoint_task_config,
)

from aerial_gym.config.task_config.navigation_task_config import (
    task_config as navigation_task_config,
)
from aerial_gym.registry.task_registry import task_registry


task_registry.register_task(
    "position_setpoint_task", PositionSetpointTask, position_setpoint_task_config
)

task_registry.register_task("navigation_task", NavigationTask, navigation_task_config)


## Uncomment this to use custom tasks

# from aerial_gym.task.custom_task.custom_task import CustomTask
# task_registry.register_task("custom_task", CustomTask, custom_task.task_config)
