# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from aerial_gym.envs.base.aerial_robot_config import AerialRobotCfg
from aerial_gym.envs.base.aerial_robot_with_obstacles_config import AerialRobotWithObstaclesCfg
from .base.aerial_robot  import AerialRobot
from .base.aerial_robot_with_obstacles import AerialRobotWithObstacles

import os

from aerial_gym.utils.task_registry import task_registry

task_registry.register( "quad", AerialRobot, AerialRobotCfg())
task_registry.register("quad_with_obstacles", AerialRobotWithObstacles, AerialRobotWithObstaclesCfg())

"""
        This file is used to import the environments and register them with the task registry.
        The task registry is used to keep track of the different environments and their configurations.

        The environments are imported from the base folder and registered with the task registry using the 
        register function.The register function takes the name of the environment and the environment class
        as arguments. 
        
        The environment class is initialized with the configuration of the environment, 
        which is defined in the corresponding configuration file. 

        The configuration file contains the parameters and settings for the environment, such as 
        the number of environments, the controller type, and the maximum episode length. The task
        registry is used to create instances of the environments based on their names, which are
        provided as arguments to the make_env function.

        The make_env function returns an instance of the environment class with the specified configuration.
        This allows for easy creation and management of different environments in the codebase.
    
"""
