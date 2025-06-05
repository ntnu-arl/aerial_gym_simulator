# import configs here
from aerial_gym.config.robot_config.base_quad_config import BaseQuadCfg
from aerial_gym.config.robot_config.base_octarotor_config import BaseOctarotorCfg
from aerial_gym.config.robot_config.base_random_config import BaseRandCfg
from aerial_gym.config.robot_config.base_rov_config import BaseROVCfg
from aerial_gym.config.robot_config.base_quad_root_link_control_config import (
    BaseQuadRootLinkControlCfg,
)

from aerial_gym.config.robot_config.lmf1_config import LMF1Cfg
from aerial_gym.config.robot_config.lmf2_config import LMF2Cfg
from aerial_gym.config.robot_config.x500_config import X500Cfg
from aerial_gym.config.robot_config.morphy_config import MorphyCfg, MorphyFixedBaseCfg
from aerial_gym.config.robot_config.morphy_stiff_config import MorphyStiffCfg
from aerial_gym.config.robot_config.snakey_config import SnakeyCfg
from aerial_gym.config.robot_config.snakey5_config import Snakey5Cfg
from aerial_gym.config.robot_config.snakey6_config import Snakey6Cfg
from aerial_gym.config.robot_config.tinyprop_config import TinyPropCfg

from aerial_gym.config.robot_config.lmf2_config import LMF2Cfg

# import robot classes here
from aerial_gym.robots.base_multirotor import BaseMultirotor
from aerial_gym.robots.base_rov import BaseROV
from aerial_gym.robots.base_reconfigurable import BaseReconfigurable
from aerial_gym.robots.morphy import Morphy

# get robot registry
from aerial_gym.registry.robot_registry import robot_registry

from aerial_gym.config.robot_config.base_quad_config import *


# register the robot classes here
robot_registry.register("base_quadrotor", BaseMultirotor, BaseQuadCfg)
robot_registry.register("base_octarotor", BaseMultirotor, BaseOctarotorCfg)
robot_registry.register("base_random", BaseMultirotor, BaseRandCfg)
robot_registry.register("base_quad_root_link_control", BaseMultirotor, BaseQuadRootLinkControlCfg)
robot_registry.register("morphy_stiff", BaseMultirotor, MorphyStiffCfg)
robot_registry.register("morphy", Morphy, MorphyCfg)
robot_registry.register("morphy_fixed_base", Morphy, MorphyFixedBaseCfg)

robot_registry.register("snakey", BaseReconfigurable, SnakeyCfg)
robot_registry.register("snakey5", BaseReconfigurable, Snakey5Cfg)
robot_registry.register("snakey6", BaseReconfigurable, Snakey6Cfg)
robot_registry.register("base_rov", BaseROV, BaseROVCfg)
robot_registry.register("lmf1", BaseMultirotor, LMF1Cfg)
robot_registry.register("lmf2", BaseMultirotor, LMF2Cfg)
robot_registry.register("x500", BaseMultirotor, X500Cfg)

robot_registry.register("tinyprop", BaseMultirotor, TinyPropCfg)

# register the special robot classes here for working with the examples
robot_registry.register("base_quadrotor_with_imu", BaseMultirotor, BaseQuadWithImuCfg)
robot_registry.register("base_quadrotor_with_camera", BaseMultirotor, BaseQuadWithCameraCfg)
robot_registry.register("base_quadrotor_with_camera_imu", BaseMultirotor, BaseQuadWithCameraImuCfg)
robot_registry.register("base_quadrotor_with_lidar", BaseMultirotor, BaseQuadWithLidarCfg)
robot_registry.register("base_quadrotor_with_faceid_normal_camera", BaseMultirotor, BaseQuadWithFaceIDNormalCameraCfg)
robot_registry.register("base_quadrotor_with_stereo_camera", BaseMultirotor, BaseQuadWithStereoCameraCfg)

robot_registry.register("tinyprop", BaseMultirotor, TinyPropCfg)

