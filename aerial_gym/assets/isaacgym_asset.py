from urdfpy import URDF
import numpy as np

import trimesh as tm

from isaacgym import gymapi

from aerial_gym.assets.base_asset import BaseAsset


class IsaacGymAsset(BaseAsset):
    def __init__(self, gym, sim, asset_name, asset_file, loading_options):
        super().__init__(asset_name, asset_file, loading_options)
        self.gym = gym
        self.sim = sim
        self.load_from_file(self.file)

    def load_from_file(self, asset_file):
        file = asset_file.split("/")[-1]
        self.asset = self.gym.load_asset(
            self.sim, self.options.asset_folder, file, self.options.asset_options
        )

        if self.options.place_force_sensor:
            parent_link_idx = self.gym.find_asset_rigid_body_index(
                self.asset, self.options.force_sensor_parent_link
            )
            self.force_sensor_transform = gymapi.Transform()
            self.force_sensor_transform.p = gymapi.Vec3(
                self.options.force_sensor_transform[0],
                self.options.force_sensor_transform[1],
                self.options.force_sensor_transform[2],
            )
            self.force_sensor_transform.r = gymapi.Quat(
                self.options.force_sensor_transform[3],
                self.options.force_sensor_transform[4],
                self.options.force_sensor_transform[5],
                self.options.force_sensor_transform[6],
            )
            sensor_props = gymapi.ForceSensorProperties()
            sensor_props.enable_forward_dynamics_forces = True
            sensor_props.enable_constraint_solver_forces = True
            sensor_props.use_world_frame = False
            self.force_sensor_handle = self.gym.create_asset_force_sensor(
                self.asset, parent_link_idx, self.force_sensor_transform, sensor_props
            )
