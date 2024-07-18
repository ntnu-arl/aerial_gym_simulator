from urdfpy import URDF
import numpy as np

import trimesh as tm

from aerial_gym.assets.base_asset import BaseAsset


from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)


class WarpAsset(BaseAsset):
    def __init__(self, asset_name, asset_file, loading_options):
        super().__init__(asset_name, asset_file, loading_options)
        self.load_from_file(self.file)

    def load_from_file(self, asset_file):
        self.file = asset_file
        # get trimesh collision and visual meshes
        self.urdf_asset = URDF.load(asset_file)
        self.visual_mesh_items = self.urdf_asset.visual_trimesh_fk().items()
        self.urdf_named_links = [key.name for key in self.urdf_asset.link_fk().keys()]

        self.asset_meshes = []
        self.asset_vertex_segmentation_value = []
        self.variable_segmentation_mask = []

        mesh_items = self.visual_mesh_items

        # normally would not recommend this, but still have functionality for it
        if self.options.use_collision_mesh_instead_of_visual:
            self.collision_mesh_items = self.urdf_asset.collision_trimesh_fk().items()
            mesh_items = self.collision_mesh_items

            # Rename named links to only include links with collision meshes because the
            # collision_mesh_fk function does not include the links that do not have collision meshes.
            temp_named_links_with_collision_meshes = []
            for linkname in self.urdf_named_links:
                if self.urdf_asset.link_map[linkname].collision_mesh is not None:
                    temp_named_links_with_collision_meshes.append(linkname)
            self.urdf_named_links = temp_named_links_with_collision_meshes

        mesh_index = 0
        self.segmentation_id = self.options.semantic_id
        self.segmentation_counter = 0

        if self.segmentation_id < 0:
            self.segmentation_id = self.segmentation_counter

        for mesh, mesh_tf in mesh_items:
            # in this context, mesh refers to the mesh of a link
            generalized_mesh_vertices = np.c_[mesh.vertices, np.ones(len(mesh.vertices))]
            # transform vertices to the correct frame
            generalized_mesh_vertices_tf = np.matmul(mesh_tf, generalized_mesh_vertices.T).T
            mesh.vertices[:] = generalized_mesh_vertices_tf[:, 0:3]

            # If the configuration specifies that the asset should have per_link_segmentation,
            # then we allot increasing segmentation values to each link, unless the link is
            # allotted specific segmentation values in the configuration.
            # The configuration file needs to have the segmentation values for each link indexed.
            # At this time we provide the capability for this, and it is up to the user to provide
            # the correct mesh index values in the configuration file.

            # skip the values that are already in the dictionary which are predefined for objects
            while self.segmentation_counter in self.options.semantic_masked_links.values():
                self.segmentation_counter += 1

            links_to_segment = self.options.semantic_masked_links.keys()
            if len(links_to_segment) == 0:
                links_to_segment = self.urdf_named_links

            if self.options.per_link_semantic:
                # if the list is predefined, check if the link is in the list
                if self.urdf_named_links[mesh_index] in links_to_segment:
                    # if the name is in the predefined list, use the value, if not, use it and increment
                    if self.urdf_named_links[mesh_index] in self.options.semantic_masked_links:
                        object_segmentation_id = self.options.semantic_masked_links[
                            self.urdf_named_links[mesh_index]
                        ]
                        variable_segmentation_mask_value = 0
                    # if the list is predefined, but the link is not in the list, then use the counter
                    else:
                        object_segmentation_id = self.segmentation_counter
                        self.segmentation_counter += 1
                        variable_segmentation_mask_value = 1
                # if the list is not predefined, then use the counter
                else:
                    object_segmentation_id = self.segmentation_counter
                    variable_segmentation_mask_value = 1
                logger.debug(
                    f"Mesh name {self.urdf_named_links[mesh_index]} has segmentation id {object_segmentation_id}"
                )
            else:
                if self.options.semantic_id < 0:
                    logger.debug("Segmentation id is negative. Using the counter.")
                    object_segmentation_id = self.segmentation_counter
                    variable_segmentation_mask_value = 1
                else:
                    object_segmentation_id = self.segmentation_id
                    variable_segmentation_mask_value = 0
                logger.debug(
                    f"Mesh name {self.urdf_named_links[mesh_index]} has segmentation id {object_segmentation_id}"
                    + f" and variable_segmentation_mask_value {variable_segmentation_mask_value}"
                )

            self.asset_meshes.append(mesh)
            self.asset_vertex_segmentation_value += [object_segmentation_id] * len(mesh.vertices)
            self.variable_segmentation_mask += [variable_segmentation_mask_value] * len(
                mesh.vertices
            )
            mesh_index += 1

        self.asset_unified_mesh = tm.util.concatenate(self.asset_meshes)
        self.asset_vertex_segmentation_value = np.array(self.asset_vertex_segmentation_value)
        logger.debug(
            f"Asset {asset_file} has {len(self.asset_unified_mesh.vertices)} vertices. Segmentation mask: {self.variable_segmentation_mask}"
        )
        logger.debug(f"Asset vertex segmentation value: {self.asset_vertex_segmentation_value}")
        self.variable_segmentation_mask = np.array(self.variable_segmentation_mask)

        # assert the above but print an error message if the assertion fails
        assert len(self.asset_vertex_segmentation_value) == len(
            self.asset_unified_mesh.vertices
        ), f"len(self.asset_vertex_segmentation_value) = {len(self.asset_vertex_segmentation_value)}, len(self.asset_unified_mesh.vertices) = {len(self.asset_unified_mesh.vertices)}"

        # also assert that the segmentation mask is the same length as the vertices
        assert len(self.variable_segmentation_mask) == len(
            self.asset_unified_mesh.vertices
        ), f"len(self.variable_segmentation_mask) = {len(self.variable_segmentation_mask)}, len(self.asset_unified_mesh.vertices) = {len(self.asset_unified_mesh.vertices)}"

        # also assert that the length of the segmentation mask is the same as the length of the segmentation values
        assert len(self.variable_segmentation_mask) == len(
            self.asset_vertex_segmentation_value
        ), f"len(self.variable_segmentation_mask) = {len(self.variable_segmentation_mask)}, len(self.asset_vertex_segmentation_value) = {len(self.asset_vertex_segmentation_value)}"
