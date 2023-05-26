# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .base_config import BaseConfig

import numpy as np
from aerial_gym import AERIAL_GYM_ROOT_DIR

class AerialRobotCfg(BaseConfig):
    seed = 1
    class env:
        num_envs = 65536
        num_observations = 13
        get_privileged_obs = False # if True the states of all entitites in the environment will be returned as privileged observations, otherwise None will be returned
        num_actions = 4
        env_spacing = 1
        episode_length_s = 8 # episode length in seconds
        num_control_steps_per_env_step = 1 # number of physics steps per env step

    class control:
        """
        Control parameters
        controller:
            lee_position_control: command_actions = [x, y, z, yaw] in environment frame scaled between -1 and 1
            lee_velocity_control: command_actions = [vx, vy, vz, yaw_rate] in vehicle frame scaled between -1 and 1
            lee_attitude_control: command_actions = [thrust, roll, pitch, yaw_rate] in vehicle frame scaled between -1 and 1
        kP: gains for position
        kV: gains for velocity
        kR: gains for attitude
        kOmega: gains for angular velocity
        """
        controller = "lee_attitude_control" # or "lee_velocity_control" or "lee_attitude_control"
        kP = [0.8, 0.8, 1.0] # used for lee_position_control only
        kV = [0.5, 0.5, 0.4] # used for lee_position_control, lee_velocity_control only
        kR = [3.0, 3.0, 1.0] # used for lee_position_control, lee_velocity_control and lee_attitude_control
        kOmega = [0.5, 0.5, 1.20] # used for lee_position_control, lee_velocity_control and lee_attitude_control
        scale_input = [1.0, 1.0, 1.0, 1.0] # scale the input to the controller from -1 to 1 for each dimension, scale from -np.pi to np.pi for yaw in the case of position control

    class robot_asset:
        file = "{AERIAL_GYM_ROOT_DIR}/resources/robots/quad/model.urdf"
        name = "aerial_robot"  # actor name
        base_link_name = "base_link"
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fix the base of the robot
        collision_mask = 1 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 100.
        max_linear_velocity = 100.
        armature = 0.001

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [-5, -5, 4]  # [m]
        lookat = [0, 0, 0]  # [m]

    class sim:
        dt =  0.01
        substeps = 1
        gravity = [0., 0. , -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 0 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
