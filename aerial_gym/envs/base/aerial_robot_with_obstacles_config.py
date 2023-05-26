# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .base_config import BaseConfig

import numpy as np
from aerial_gym import AERIAL_GYM_ROOT_DIR

THIN_SEMANTIC_ID = 1
TREE_SEMANTIC_ID = 2
OBJECT_SEMANTIC_ID = 3
WALL_SEMANTIC_ID = 8

class AerialRobotWithObstaclesCfg(BaseConfig):
    seed = 1
    class env:
        num_envs = 64
        num_observations = 13
        get_privileged_obs = True # if True the states of all entitites in the environment will be returned as privileged observations, otherwise None will be returned
        num_actions = 4
        env_spacing = 5.0  # not used with heightfields/trimeshes
        episode_length_s = 10 # episode length in seconds
        num_control_steps_per_env_step = 10 # number of control & physics steps between camera renders
        enable_onboard_cameras = False # enable onboard cameras
        reset_on_collision = True # reset environment when contact force on quadrotor is above a threshold
        create_ground_plane = True # create a ground plane

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
            bounce_threshold_velocity = 0.5 # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 1 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

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
        controller = "lee_velocity_control" # or "lee_velocity_control" or "lee_attitude_control"
        kP = [0.8, 0.8, 1.0] # used for lee_position_control only
        kV = [0.5, 0.5, 0.4] # used for lee_position_control, lee_velocity_control only
        kR = [3.0, 3.0, 1.0] # used for lee_position_control, lee_velocity_control and lee_attitude_control
        kOmega = [0.5, 0.5, 1.20] # used for lee_position_control, lee_velocity_control and lee_attitude_control
        scale_input =[2.0, 1.0, 1.0, np.pi/4.0] # scale the input to the controller from -1 to 1 for each dimension

    class robot_asset:
        file = "{AERIAL_GYM_ROOT_DIR}/resources/robots/quad/model.urdf"
        name = "aerial_robot"  # actor name
        base_link_name = "base_link"
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints.
        fix_base_link = False # fix the base of the robot
        collision_mask = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 100.
        max_linear_velocity = 100.
        armature = 0.001


    class asset_state_params(robot_asset):
        num_assets = 1                  # number of assets to include

        min_position_ratio = [0.5, 0.5, 0.5] # min position as a ratio of the bounds
        max_position_ratio = [0.5, 0.5, 0.5] # max position as a ratio of the bounds

        collision_mask = 1

        collapse_fixed_joints = True
        fix_base_link = True
        links_per_asset = 1
        set_whole_body_semantic_mask = False
        set_semantic_mask_per_link = False
        semantic_mask_link_list = [] # For empty list, all links are labeled
        specific_filepath = None # if not None, use this folder instead randomizing
        color = None


    class thin_asset_params(asset_state_params):
        num_assets = 10

        collision_mask = 1 # objects with the same collision mask will not collide

        max_position_ratio = [0.95, 0.95, 0.95] # min position as a ratio of the bounds
        min_position_ratio = [0.05, 0.05, 0.05] # max position as a ratio of the bounds

        specified_position = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing   the ratios

        min_euler_angles = [-np.pi, -np.pi, -np.pi] # min euler angles
        max_euler_angles = [np.pi, np.pi, np.pi] # max euler angles

        specified_euler_angle = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing
                
        collapse_fixed_joints = True
        links_per_asset = 1
        set_whole_body_semantic_mask = True
        semantic_id = THIN_SEMANTIC_ID
        set_semantic_mask_per_link = False
        semantic_mask_link_list = [] ## If nothing is specified, all links are labeled
        color = [170,66,66]
      

    class tree_asset_params(asset_state_params):
        num_assets = 10

        collision_mask = 1 # objects with the same collision mask will not collide

        max_position_ratio = [0.95, 0.95, 0.1] # min position as a ratio of the bounds
        min_position_ratio = [0.05, 0.05, 0.0] # max position as a ratio of the bounds

        specified_position = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing   the ratios

        min_euler_angles = [0, -np.pi/6.0, -np.pi] # min euler angles
        max_euler_angles = [0, np.pi/6.0, np.pi] # max euler angles

        specified_euler_angle = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing

        collapse_fixed_joints = True
        links_per_asset = 1
        set_whole_body_semantic_mask = False
        set_semantic_mask_per_link = True
        semantic_mask_link_list = [] ## If nothing is specified, all links are labeled
        semantic_id = TREE_SEMANTIC_ID
        color = [70,200,100]

    class object_asset_params(asset_state_params):
        num_assets = 50
        
        max_position_ratio = [0.95, 0.95, 0.95] # min position as a ratio of the bounds
        min_position_ratio = [0.05, 0.05, 0.05] # max position as a ratio of the bounds

        specified_position = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0, -np.pi/6, -np.pi] # min euler angles
        max_euler_angles = [0, np.pi/6, np.pi] # max euler angles

        specified_euler_angle = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing

        links_per_asset = 1
        set_whole_body_semantic_mask = False
        set_semantic_mask_per_link = False
        semantic_id = OBJECT_SEMANTIC_ID

        # color = [80,255,100]

    class left_wall(asset_state_params):
        num_assets = 1

        collision_mask = 1 # objects with the same collision mask will not collide
        
        min_position_ratio = [0.5, 1.0, 0.5] # min position as a ratio of the bounds
        max_position_ratio = [0.5, 1.0, 0.5] # max position as a ratio of the bounds

        specified_position = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, 0.0] # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0] # max euler angles

        specified_euler_angle = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing
                
        collapse_fixed_joints = False
        links_per_asset = 1
        specific_filepath = "cube.urdf"
        semantic_id = WALL_SEMANTIC_ID
        color = [100,200,210]
    
    class right_wall(asset_state_params):
        num_assets = 1

        collision_mask = 1 # objects with the same collision mask will not collide

        min_position_ratio = [0.5, 0.0, 0.5] # min position as a ratio of the bounds
        max_position_ratio = [0.5, 0.0, 0.5] # max position as a ratio of the bounds

        specified_position = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, 0.0] # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0] # max euler angles

        specified_euler_angle = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing
        
        collapse_fixed_joints = False
        links_per_asset = 1
        specific_filepath = "cube.urdf"
        semantic_id = WALL_SEMANTIC_ID
        color = [100,200,210]
    
    class top_wall(asset_state_params):
        num_assets = 1

        collision_mask = 1 # objects with the same collision mask will not collide

        min_position_ratio = [0.5, 0.5, 1.0] # min position as a ratio of the bounds
        max_position_ratio = [0.5, 0.5, 1.0] # max position as a ratio of the bounds

        specified_position = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, 0.0] # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0] # max euler angles

        specified_euler_angle = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing

        collapse_fixed_joints = False
        links_per_asset = 1
        specific_filepath = "cube.urdf"
        semantic_id = WALL_SEMANTIC_ID
        color = [100,200,210]
    
    class bottom_wall(asset_state_params):
        num_assets = 1

        collision_mask = 1 # objects with the same collision mask will not collide

        min_position_ratio = [0.5, 0.5, 0.0] # min position as a ratio of the bounds
        max_position_ratio = [0.5, 0.5, 0.0] # max position as a ratio of the bounds

        specified_position = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, 0.0] # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0] # max euler angles

        specified_euler_angle = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing

        collapse_fixed_joints = False
        links_per_asset = 1
        specific_filepath = "cube.urdf"
        semantic_id = WALL_SEMANTIC_ID
        color = [100,150,150]
    

    class front_wall(asset_state_params):
        num_assets = 1

        collision_mask = 1 # objects with the same collision mask will not collide


        min_position_ratio = [1.0, 0.5, 0.5] # min position as a ratio of the bounds
        max_position_ratio = [1.0, 0.5, 0.5] # max position as a ratio of the bounds

        specified_position = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, 0.0] # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0] # max euler angles

        specified_euler_angle = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing

        collapse_fixed_joints = False
        links_per_asset = 1
        specific_filepath = "cube.urdf"
        semantic_id = WALL_SEMANTIC_ID
        color = [100,200,210]
    
    class back_wall(asset_state_params):
        num_assets = 1

        collision_mask = 1 # objects with the same collision mask will not collide
        
        min_position_ratio = [0.0, 0.5, 0.5] # min position as a ratio of the bounds
        max_position_ratio = [0.0, 0.5, 0.5] # max position as a ratio of the bounds

        specified_position = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, 0.0] # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0] # max euler angles

        specified_euler_angle = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing
        
        collapse_fixed_joints = False
        links_per_asset = 1
        specific_filepath = "cube.urdf"
        semantic_id = WALL_SEMANTIC_ID
        color = [100,200,210]


    class asset_config:
        folder_path = f"{AERIAL_GYM_ROOT_DIR}/resources/models/environment_assets"
        
        include_asset_type = {
            "thin": True,
            "trees": False,
            "objects": True
            }
            
        include_env_bound_type = {
            "front_wall": False, 
            "left_wall": False, 
            "top_wall": False, 
            "back_wall": False,
            "right_wall": False, 
            "bottom_wall": False}

        env_lower_bound_min = [-5.0, -5.0, 0.0] # lower bound for the environment space
        env_lower_bound_max = [-5.0, -5.0, 0.0] # lower bound for the environment space
        env_upper_bound_min = [5.0, 5.0, 5.0] # upper bound for the environment space
        env_upper_bound_max = [5.0, 5.0, 5.0] # upper bound for the environment space
 