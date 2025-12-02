# import classes for controllers
from aerial_gym.control.controllers.acceleration_control import (
    LeeAccelerationController,
)
from aerial_gym.control.controllers.attitude_control import LeeAttitudeController
from aerial_gym.control.controllers.velocity_control import LeeVelocityController
from aerial_gym.control.controllers.position_control import LeePositionController
from aerial_gym.control.controllers.velocity_steeing_angle_controller import (
    LeeVelocitySteeringAngleController,
)
from aerial_gym.control.controllers.rates_control import LeeRatesController
from aerial_gym.control.controllers.no_control import NoControl


# import configs for controllers
from aerial_gym.config.controller_config.lee_controller_config import (
    control as lee_controller_config,
)
from aerial_gym.config.controller_config.no_control_config import (
    control as no_control_config,
)

from aerial_gym.config.controller_config.lee_controller_config_octarotor import (
    control as lee_controller_config_octarotor,
)

from aerial_gym.control.controllers.fully_actuated_control import FullyActuatedController
from aerial_gym.config.controller_config.fully_actuated_controller_rov import (
    control as fully_actuated_controller_config,
)

from aerial_gym.config.controller_config.lmf2_controller_config import (
    control as lmf2_controller_config,
)

from aerial_gym.config.controller_config.magpie_controller_config import (
    control as magpie_controller_config,
)

from aerial_gym.registry.controller_registry import controller_registry

controller_registry.register_controller("no_control", NoControl, no_control_config)


controller_registry.register_controller(
    "lee_position_control", LeePositionController, lee_controller_config
)
controller_registry.register_controller(
    "lee_velocity_control", LeeVelocityController, lee_controller_config
)
controller_registry.register_controller(
    "lee_attitude_control", LeeAttitudeController, lee_controller_config
)
controller_registry.register_controller(
    "lee_rates_control", LeeRatesController, lee_controller_config
)
controller_registry.register_controller(
    "lee_acceleration_control", LeeAccelerationController, lee_controller_config
)

def register_robot_controllers(robot_name=None, controller_config=None):
    controller_registry.register_controller(
        f"{robot_name}_position_control",
        LeePositionController,
        controller_config,
    )
    controller_registry.register_controller(
        f"{robot_name}_velocity_control",
        LeeVelocityController,
        controller_config,
    )
    controller_registry.register_controller(
        f"{robot_name}_attitude_control",
        LeeAttitudeController,
        controller_config,
    )
    controller_registry.register_controller(
        f"{robot_name}_rates_control",
        LeeRatesController,
        controller_config,
    )
    controller_registry.register_controller(
        f"{robot_name}_acceleration_control",
        LeeAccelerationController,
        controller_config,
    )

register_robot_controllers(
    "magpie", magpie_controller_config
)
register_robot_controllers(
    "lmf2", lmf2_controller_config
)
register_robot_controllers(
    "octarotor", lee_controller_config_octarotor
)

controller_registry.register_controller(
    "rov_fully_actuated_control", FullyActuatedController, fully_actuated_controller_config
)
