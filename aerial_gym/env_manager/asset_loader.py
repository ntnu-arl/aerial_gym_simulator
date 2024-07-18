import os
import random

from isaacgym import gymapi
from aerial_gym.assets.warp_asset import WarpAsset
from aerial_gym.assets.isaacgym_asset import IsaacGymAsset

from collections import deque


from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("asset_loader")


def asset_class_to_AssetOptions(asset_class):
    asset_options = gymapi.AssetOptions()
    asset_options.collapse_fixed_joints = asset_class.collapse_fixed_joints
    asset_options.replace_cylinder_with_capsule = asset_class.replace_cylinder_with_capsule
    asset_options.flip_visual_attachments = asset_class.flip_visual_attachments
    asset_options.fix_base_link = asset_class.fix_base_link
    asset_options.density = asset_class.density
    asset_options.angular_damping = asset_class.angular_damping
    asset_options.linear_damping = asset_class.linear_damping
    asset_options.max_angular_velocity = asset_class.max_angular_velocity
    asset_options.max_linear_velocity = asset_class.max_linear_velocity
    asset_options.disable_gravity = asset_class.disable_gravity
    return asset_options


class AssetLoader:
    def __init__(self, global_sim_dict, device):
        self.global_sim_dict = global_sim_dict
        self.gym = self.global_sim_dict["gym"]
        self.sim = self.global_sim_dict["sim"]
        self.cfg = self.global_sim_dict["env_cfg"]
        self.device = device
        self.env_config = self.cfg.env_config
        self.num_envs = self.cfg.env.num_envs

        self.asset_buffer = {}
        self.global_asset_counter = 0

        self.max_loaded_semantic_id = 0

    def randomly_pick_assets_from_folder(self, folder, num_assets=0):
        # pick files that are URDF files from the folder
        available_assets = []
        for file in os.listdir(folder):
            if file.endswith(".urdf"):
                available_assets.append(file)

        if num_assets == 0:
            return []

        selected_files = random.choices(available_assets, k=num_assets)
        return selected_files

    def load_selected_file_from_config(
        self, asset_type, asset_class_config, selected_file, is_robot=False
    ):

        asset_options_for_class = asset_class_to_AssetOptions(asset_class_config)
        filepath = os.path.join(asset_class_config.asset_folder, selected_file)

        # check if  it exists in the buffer

        if filepath in self.asset_buffer:
            return self.asset_buffer[filepath]

        logger.info(
            f"Loading asset: {selected_file} for the first time. Next use of this asset will be via the asset buffer."
        )

        asset_class_dict = {
            "asset_type": asset_type,
            "asset_options": asset_options_for_class,
            "semantic_id": asset_class_config.semantic_id,
            "collision_mask": asset_class_config.collision_mask,
            "color": asset_class_config.color,
            "semantic_masked_links": asset_class_config.semantic_masked_links,
            "keep_in_env": asset_class_config.keep_in_env,
            "filename": filepath,
            "asset_folder": asset_class_config.asset_folder,
            "per_link_semantic": asset_class_config.per_link_semantic,
            "min_state_ratio": asset_class_config.min_state_ratio,
            "max_state_ratio": asset_class_config.max_state_ratio,
            "place_force_sensor": asset_class_config.place_force_sensor,
            "force_sensor_parent_link": asset_class_config.force_sensor_parent_link,
            "force_sensor_transform": asset_class_config.force_sensor_transform,
            "use_collision_mesh_instead_of_visual": asset_class_config.use_collision_mesh_instead_of_visual,
            # do stuff with position, randomization, etc
        }
        max_list_vals = 0
        if len(list(asset_class_config.semantic_masked_links.values())) > 0:
            max_list_vals = max(list(asset_class_config.semantic_masked_links.values()))

        self.max_loaded_semantic_id = max(
            self.max_loaded_semantic_id, asset_class_config.semantic_id, max_list_vals
        )

        asset_name = asset_type

        # get robot sensor config
        robot_sensor_config = self.global_sim_dict["robot_config"].sensor_config
        use_camera_collision_mesh = (
            robot_sensor_config.camera_config.use_collision_geometry
            if robot_sensor_config.enable_camera
            else False
        )

        if is_robot == False:
            if self.cfg.env.use_warp:
                warp_asset = WarpAsset(asset_name, filepath, asset_class_dict)
                asset_class_dict["warp_asset"] = warp_asset
                if use_camera_collision_mesh:
                    msg_str = (
                        "Warp cameras will render the mesh per each asset that is "
                        + "specified by the asset configuration file."
                        + "The parameter from the sensor configuration file"
                        + " to render collision meshes will be ignored."
                        + "This message is generated because you have set the"
                        + "use_collision_geometry parameter to True in the sensor"
                        + "configuration file."
                    )
                    logger.warning(msg_str)
            elif (
                use_camera_collision_mesh != asset_class_config.use_collision_mesh_instead_of_visual
            ):
                msg_str = (
                    "Choosing between collision and visual meshes per asset is not supported"
                    + "for Isaac Gym rendering pipeline. If the Isaac Gym rendering pipeline is selected, "
                    + "you can render only visual or only collision meshes for all assets."
                    + " Please make ensure that the appropriate option is set in the sensor configuration file."
                    + " Current simulation will run but will render only the mesh you set for the sensor "
                    + "configuration and the setting from the asset configuration will be ignored."
                    + "This message is generated because the use_collision_geometry parameter in the sensor"
                    + "configuration file is different from the use_collision_mesh_instead_of_visual parameter in the asset"
                    + "configuration file. \n\n\nThe simulation will still run but the rendering will be as per the sensor configuration file."
                )
                logger.warning(msg_str)
        IGE_asset = IsaacGymAsset(self.gym, self.sim, asset_name, filepath, asset_class_dict)
        asset_class_dict["isaacgym_asset"] = IGE_asset
        self.asset_buffer[filepath] = asset_class_dict
        return asset_class_dict

    def select_and_order_assets(self):
        ordered_asset_list = deque()
        keep_in_env_num = 0
        for (
            asset_type,
            asset_class_config,
        ) in self.env_config.asset_type_to_dict_map.items():
            num_assets = asset_class_config.num_assets
            if (
                asset_type in self.env_config.include_asset_type
                and self.env_config.include_asset_type[asset_type] == False
            ):
                continue
            if num_assets > 0:
                if asset_class_config.file is None:
                    selected_files = self.randomly_pick_assets_from_folder(
                        asset_class_config.asset_folder, num_assets
                    )
                else:
                    selected_files = [asset_class_config.file] * num_assets

                for selected_file in selected_files:
                    asset_info_dict = self.load_selected_file_from_config(
                        asset_type, asset_class_config, selected_file
                    )
                    if asset_info_dict["keep_in_env"]:
                        ordered_asset_list.appendleft(asset_info_dict)
                        logger.debug(f"Asset {asset_type} kept in env")
                        keep_in_env_num += 1
                    else:
                        ordered_asset_list.append(asset_info_dict)

        # shuffle assets that are not necessarily kept in the environment
        ordered_asset_list = list(ordered_asset_list)
        shuffle_subset = ordered_asset_list[keep_in_env_num:]
        random.shuffle(shuffle_subset)
        ordered_asset_list[keep_in_env_num:] = shuffle_subset
        return ordered_asset_list, keep_in_env_num

    def select_assets_for_sim(self):
        self.global_env_asset_dicts = []
        for i in range(self.num_envs):
            logger.debug(f"Loading assets for env: {i}")
            ordered_asset_list, num_assets_kept_in_env = self.select_and_order_assets()
            self.global_env_asset_dicts.append(ordered_asset_list)
            logger.debug(f"Loaded assets for env {i}")
        return self.global_env_asset_dicts, num_assets_kept_in_env
