# import configs here
from aerial_gym.config.robot_config.base_quad_config import BaseQuadCfg
from aerial_gym.config.robot_config.base_octarotor_config import BaseOctarotorCfg
from aerial_gym.config.robot_config.base_random_config import BaseRandCfg
from aerial_gym.config.robot_config.base_rov_config import BaseROVCfg
from aerial_gym.config.robot_config.base_quad_root_link_control_config import (
    BaseQuadRootLinkControlCfg,
)


# import robot classes here
from aerial_gym.robots.base_multirotor import BaseMultirotor
from aerial_gym.robots.base_rov import BaseROV

# get robot registry
from aerial_gym.registry.robot_registry import robot_registry

# register the robot classes here
robot_registry.register("base_quadrotor", BaseMultirotor, BaseQuadCfg)
robot_registry.register("base_octarotor", BaseMultirotor, BaseOctarotorCfg)
robot_registry.register("base_random", BaseMultirotor, BaseRandCfg)
robot_registry.register("base_quad_root_link_control", BaseMultirotor, BaseQuadRootLinkControlCfg)

robot_registry.register("base_rov", BaseROV, BaseROVCfg)
