from aerial_gym.config.asset_config.base_asset import *

import numpy as np

THIN_SEMANTIC_ID = 1
TREE_SEMANTIC_ID = 2
OBJECT_SEMANTIC_ID = 3
WALL_SEMANTIC_ID = 8


class EnvObjectConfig:
    class panel_asset_params(BaseAssetParams):
        num_assets = 6

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
        per_link_semantic = True
        semantic_id = -1
        color = [170, 66, 66]

    class thin_asset_params(BaseAssetParams):
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
        semantic_id = THIN_SEMANTIC_ID
        color = [170, 66, 66]

    class tree_asset_params(BaseAssetParams):
        num_assets = 1

        asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/trees"

        collision_mask = 1  # objects with the same collision mask will not collide

        min_state_ratio = [
            0.2,
            0.05,
            0.05,
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
            0.9,
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
        per_link_semantic = False

        semantic_id = TREE_SEMANTIC_ID
        color = [70, 200, 100]

        semantic_masked_links = {}

    class object_asset_params(BaseAssetParams):
        num_assets = 2

        asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/objects"

        min_state_ratio = [
            0.25,
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

        semantic_id = OBJECT_SEMANTIC_ID

        # color = [80,255,100]

    class left_wall(BaseAssetParams):
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
        per_link_semantic = True
        semantic_id = -1  # semantic_id = WALL_SEMANTIC_ID
        color = [100, 200, 210]

    class right_wall(BaseAssetParams):
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
        semantic_id = -1  # semantic_id = WALL_SEMANTIC_ID
        color = [100, 200, 210]

    class top_wall(BaseAssetParams):
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
        per_link_semantic = True
        semantic_id = -1  # semantic_id = WALL_SEMANTIC_ID
        color = [100, 200, 210]

    class bottom_wall(BaseAssetParams):
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
        per_link_semantic = True
        semantic_id = -1  # semantic_id = WALL_SEMANTIC_ID
        color = [100, 150, 150]

    class front_wall(BaseAssetParams):
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
        per_link_semantic = True
        semantic_id = -1  # semantic_id = WALL_SEMANTIC_ID
        color = [100, 200, 210]

    class back_wall(BaseAssetParams):
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
        semantic_id = -1  # semantic_id = WALL_SEMANTIC_ID
        color = [100, 200, 210]
