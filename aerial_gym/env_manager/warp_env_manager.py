from aerial_gym.env_manager.base_env_manager import BaseManager

import warp as wp
import numpy as np
import torch

import trimesh as tm

from aerial_gym.utils.math import tf_apply

# intialize warp
wp.init()

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("warp_env_manager")


class WarpEnv(BaseManager):
    def __init__(self, global_sim_dict, device):
        logger.debug("Initializing WarpEnv")
        super().__init__(global_sim_dict["env_cfg"], device)
        self.num_envs = global_sim_dict["num_envs"]
        self.env_meshes = []
        self.warp_mesh_id_list = []
        self.warp_mesh_per_env = []
        self.global_vertex_to_asset_index_tensor = None
        self.vertex_maps_per_env_original = None
        self.global_env_mesh_list = []
        self.global_vertex_counter = 0
        self.global_vertex_segmentation_list = []
        self.global_vertex_to_asset_index_map = []

        self.CONST_WARP_MESH_ID_LIST = None
        self.CONST_WARP_MESH_PER_ENV = None
        self.CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR = None
        self.VERTEX_MAPS_PER_ENV_ORIGINAL = None
        logger.debug("[DONE] Initializing WarpEnv")

    def reset_idx(self, env_ids):
        if self.global_vertex_counter == 0:
            return
        # logger.debug("Updating vertex maps per env")
        self.vertex_maps_per_env_updated[:] = tf_apply(
            self.unfolded_env_vec_root_tensor[self.CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR, 3:7],
            self.unfolded_env_vec_root_tensor[self.CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR, 0:3],
            self.VERTEX_MAPS_PER_ENV_ORIGINAL[:],
        )
        # logger.debug("[DONE] Updating vertex maps per env")

        # logger.debug("Refitting warp meshes")
        for i in env_ids:
            self.warp_mesh_per_env[i].refit()
        # logger.debug("[DONE] Refitting warp meshes")

    def pre_physics_step(self, action):
        pass

    def post_physics_step(self):
        pass

    def step(self, action):
        pass

    def reset(self):
        return self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_env(self, env_id):
        if len(self.env_meshes) <= env_id:
            self.env_meshes.append([])
        else:
            raise ValueError("Environment already exists")

    def add_asset_to_env(self, asset_info_dict, env_id, global_asset_counter, segmentation_counter):
        warp_asset = asset_info_dict["warp_asset"]
        # use the variable segmentation mask to set the segmentation id for each vertex
        updated_vertex_segmentation = (
            warp_asset.asset_vertex_segmentation_value
            + segmentation_counter * warp_asset.variable_segmentation_mask
        )
        logger.debug(
            f"Asset {asset_info_dict['filename']} has {len(warp_asset.asset_unified_mesh.vertices)} vertices. Segmentation mask: {warp_asset.variable_segmentation_mask} and updated segmentation: {updated_vertex_segmentation}"
        )
        self.env_meshes[env_id].append(warp_asset.asset_unified_mesh)

        self.global_vertex_to_asset_index_map += [global_asset_counter] * len(
            warp_asset.asset_unified_mesh.vertices
        )
        self.global_vertex_counter += len(warp_asset.asset_unified_mesh.vertices)
        self.global_vertex_segmentation_list += updated_vertex_segmentation.tolist()
        return None, len(
            np.unique(
                warp_asset.asset_vertex_segmentation_value * warp_asset.variable_segmentation_mask
            )
        )

    def prepare_for_simulation(self, global_tensor_dict):
        logger.debug("Preparing for simulation")
        self.global_tensor_dict = global_tensor_dict
        if self.global_vertex_counter == 0:
            logger.warning(
                "No assets have been added to the environment. Skipping preparation for simulation"
            )
            self.global_tensor_dict["CONST_WARP_MESH_ID_LIST"] = None
            self.global_tensor_dict["CONST_WARP_MESH_PER_ENV"] = None
            self.global_tensor_dict["CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR"] = None
            self.global_tensor_dict["VERTEX_MAPS_PER_ENV_ORIGINAL"] = None
            return 1

        self.global_vertex_to_asset_index_tensor = torch.tensor(
            self.global_vertex_to_asset_index_map,
            device=self.device,
            requires_grad=False,
        )
        self.vertex_maps_per_env_original = torch.zeros(
            (self.global_vertex_counter, 3), device=self.device, requires_grad=False
        )
        # updated vertex maps are used for the warp environment
        self.vertex_maps_per_env_updated = self.vertex_maps_per_env_original.clone()

        ## unify env meshes
        logger.debug("Unifying environment meshes")
        for i in range(len(self.env_meshes)):
            self.global_env_mesh_list.append(tm.util.concatenate(self.env_meshes[i]))
        logger.debug("[DONE] Unifying environment meshes")

        # prepare warp meshes
        logger.debug("Creating warp meshes")
        vertex_iterator = 0
        for env_mesh in self.global_env_mesh_list:
            self.vertex_maps_per_env_original[
                vertex_iterator : vertex_iterator + len(env_mesh.vertices)
            ] = torch.tensor(env_mesh.vertices, device=self.device, requires_grad=False)
            faces_tensor = torch.tensor(
                env_mesh.faces,
                device=self.device,
                requires_grad=False,
                dtype=torch.int32,
            )
            vertex_velocities = torch.zeros(
                len(env_mesh.vertices), 3, device=self.device, requires_grad=False
            )
            segmentation_tensor = torch.tensor(
                self.global_vertex_segmentation_list[
                    vertex_iterator : vertex_iterator + len(env_mesh.vertices)
                ],
                device=self.device,
                requires_grad=False,
            )
            # we hijack this field and use it for segmentation
            vertex_velocities[:, 0] = segmentation_tensor

            vertex_vec3_array = wp.from_torch(
                self.vertex_maps_per_env_updated[
                    vertex_iterator : vertex_iterator + len(env_mesh.vertices)
                ],
                dtype=wp.vec3,
            )
            faces_wp_int32_array = wp.from_torch(faces_tensor.flatten(), dtype=wp.int32)
            velocities_vec3_array = wp.from_torch(vertex_velocities, dtype=wp.vec3)

            wp_mesh = wp.Mesh(
                points=vertex_vec3_array,
                indices=faces_wp_int32_array,
                velocities=velocities_vec3_array,
            )

            self.warp_mesh_per_env.append(wp_mesh)
            self.warp_mesh_id_list.append(wp_mesh.id)
            vertex_iterator += len(env_mesh.vertices)

        logger.debug("[DONE] Creating warp meshes")
        # define consts so that they can be accessed only after the environment has been prepared
        self.CONST_WARP_MESH_ID_LIST = self.warp_mesh_id_list
        self.CONST_WARP_MESH_PER_ENV = self.warp_mesh_per_env
        self.CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR = self.global_vertex_to_asset_index_tensor
        self.VERTEX_MAPS_PER_ENV_ORIGINAL = self.vertex_maps_per_env_original

        self.global_tensor_dict["CONST_WARP_MESH_ID_LIST"] = self.CONST_WARP_MESH_ID_LIST
        self.global_tensor_dict["CONST_WARP_MESH_PER_ENV"] = self.CONST_WARP_MESH_PER_ENV
        self.global_tensor_dict["CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR"] = (
            self.CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR
        )
        self.global_tensor_dict["VERTEX_MAPS_PER_ENV_ORIGINAL"] = self.VERTEX_MAPS_PER_ENV_ORIGINAL

        self.unfolded_env_vec_root_tensor = self.global_tensor_dict[
            "unfolded_env_asset_state_tensor"
        ]
        return 1
