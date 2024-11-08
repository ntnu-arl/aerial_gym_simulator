from aerial_gym import AERIAL_GYM_DIRECTORY

import numpy as np

THIN_SEMANTIC_ID = 1
TREE_SEMANTIC_ID = 2
OBJECT_SEMANTIC_ID = 3
PANEL_SEMANTIC_ID = 20
FRONT_WALL_SEMANTIC_ID = 9
BACK_WALL_SEMANTIC_ID = 10
LEFT_WALL_SEMANTIC_ID = 11
RIGHT_WALL_SEMANTIC_ID = 12
BOTTOM_WALL_SEMANTIC_ID = 13
TOP_WALL_SEMANTIC_ID = 14


class asset_state_params:
    num_assets = 1  # number of assets to include

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets"
    file = None  # if file=None, random assets will be selected. If not None, this file will be used

    min_position_ratio = [0.5, 0.5, 0.5]  # min position as a ratio of the bounds
    max_position_ratio = [0.5, 0.5, 0.5]  # max position as a ratio of the bounds

    collision_mask = 1

    disable_gravity = True
    replace_cylinder_with_capsule = (
        True  # replace collision cylinders with capsules, leads to faster/more stable simulation
    )
    flip_visual_attachments = True  # Some .obj meshes must be flipped from y-up to z-up
    density = 0.001
    angular_damping = 0.1
    linear_damping = 0.1
    max_angular_velocity = 100.0
    max_linear_velocity = 100.0
    armature = 0.001

    collapse_fixed_joints = True
    fix_base_link = False
    specific_filepath = None  # if not None, use this folder instead randomizing
    color = None
    keep_in_env = False

    body_semantic_label = 0
    link_semantic_label = 0
    per_link_semantic = False
    semantic_masked_links = {}
    place_force_sensor = False
    force_sensor_parent_link = "base_link"
    force_sensor_transform = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]  # position, quat x, y, z, w

    use_collision_mesh_instead_of_visual = False


class panel_asset_params(asset_state_params):
    num_assets = 3

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/panels"

    collision_mask = 1  # objects with the same collision mask will not collide

    min_position_ratio = [0.3, 0.05, 0.05]  # max position as a ratio of the bounds
    max_position_ratio = [0.85, 0.95, 0.95]  # min position as a ratio of the bounds

    specified_position = [
        -1000.0,
        -1000.0,
        -1000.0,
    ]  # if > -900, use this value instead of randomizing   the ratios

    min_euler_angles = [0.0, 0.0, -np.pi / 3.0]  # min euler angles
    max_euler_angles = [0.0, 0.0, np.pi / 3.0]  # max euler angles

    min_state_ratio = [
        0.3,
        0.05,
        0.05,
        0.0,
        0.0,
        -np.pi / 3.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        0.85,
        0.95,
        0.95,
        0.0,
        0.0,
        np.pi / 3.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    keep_in_env = True

    collapse_fixed_joints = True
    per_link_semantic = False
    semantic_id = -1  # will be assigned incrementally per instance
    color = [170, 66, 66]


class thin_asset_params(asset_state_params):
    num_assets = 0

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/thin"

    collision_mask = 1  # objects with the same collision mask will not collide

    min_state_ratio = [
        0.3,
        0.05,
        0.05,
        -np.pi,
        -np.pi,
        -np.pi,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        0.85,
        0.95,
        0.95,
        np.pi,
        np.pi,
        np.pi,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    collapse_fixed_joints = True
    per_link_semantic = False
    semantic_id = -1  # will be assigned incrementally per instance
    color = [170, 66, 66]


class tree_asset_params(asset_state_params):
    num_assets = 6

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/trees"

    collision_mask = 1  # objects with the same collision mask will not collide

    min_state_ratio = [
        0.1,
        0.1,
        0.0,
        0,
        -np.pi / 6.0,
        -np.pi,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        0.9,
        0.9,
        0.0,
        0,
        np.pi / 6.0,
        np.pi,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    collapse_fixed_joints = True
    per_link_semantic = True
    keep_in_env = True

    semantic_id = -1  # TREE_SEMANTIC_ID
    color = [70, 200, 100]

    semantic_masked_links = {}


class object_asset_params(asset_state_params):
    num_assets = 40

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/objects"

    min_state_ratio = [
        0.30,
        0.05,
        0.05,
        -np.pi,
        -np.pi,
        -np.pi,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        0.85,
        0.9,
        0.9,
        np.pi,
        np.pi,
        np.pi,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    keep_in_env = False
    per_link_semantic = False
    semantic_id = -1  # will be assigned incrementally per instance

    # color = [80,255,100]


class left_wall(asset_state_params):
    num_assets = 1

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/walls"
    file = "left_wall.urdf"

    collision_mask = 1  # objects with the same collision mask will not collide

    min_state_ratio = [
        0.5,
        1.0,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        0.5,
        1.0,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    keep_in_env = True

    collapse_fixed_joints = True
    specific_filepath = "cube.urdf"
    per_link_semantic = False
    semantic_id = LEFT_WALL_SEMANTIC_ID
    color = [100, 200, 210]


class right_wall(asset_state_params):
    num_assets = 1

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/walls"
    file = "right_wall.urdf"

    min_state_ratio = [
        0.5,
        0.0,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        0.5,
        0.0,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    keep_in_env = True

    collapse_fixed_joints = True
    per_link_semantic = False
    specific_filepath = "cube.urdf"
    semantic_id = RIGHT_WALL_SEMANTIC_ID
    color = [100, 200, 210]


class top_wall(asset_state_params):
    num_assets = 1

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/walls"
    file = "top_wall.urdf"

    collision_mask = 1  # objects with the same collision mask will not collide

    min_state_ratio = [
        0.5,
        0.5,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        0.5,
        0.5,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    keep_in_env = True

    collapse_fixed_joints = True
    specific_filepath = "cube.urdf"
    per_link_semantic = False
    semantic_id = TOP_WALL_SEMANTIC_ID
    color = [100, 200, 210]


class bottom_wall(asset_state_params):
    num_assets = 1
    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/walls"
    file = "bottom_wall.urdf"

    collision_mask = 1  # objects with the same collision mask will not collide

    min_state_ratio = [
        0.5,
        0.5,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        0.5,
        0.5,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    keep_in_env = True

    collapse_fixed_joints = True
    specific_filepath = "cube.urdf"
    per_link_semantic = False
    semantic_id = BOTTOM_WALL_SEMANTIC_ID
    color = [100, 150, 150]


class front_wall(asset_state_params):
    num_assets = 1

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/walls"
    file = "front_wall.urdf"

    collision_mask = 1  # objects with the same collision mask will not collide

    min_state_ratio = [
        1.0,
        0.5,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        1.0,
        0.5,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    keep_in_env = True

    collapse_fixed_joints = True
    specific_filepath = "cube.urdf"
    per_link_semantic = False
    semantic_id = FRONT_WALL_SEMANTIC_ID
    color = [100, 200, 210]


class back_wall(asset_state_params):
    num_assets = 1

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/walls"
    file = "back_wall.urdf"

    collision_mask = 1  # objects with the same collision mask will not collide

    min_state_ratio = [
        0.0,
        0.5,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        0.0,
        0.5,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    keep_in_env = True

    collapse_fixed_joints = True
    specific_filepath = "cube.urdf"
    per_link_semantic = False
    semantic_id = BACK_WALL_SEMANTIC_ID
    color = [100, 200, 210]
