from isaacgym import gymapi
import numpy as np

from aerial_gym.utils.math import quat_from_euler_xyz

import sys
import torch
import time

from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.math import quat_rotate_inverse, quat_rotate

logger = CustomLogger("IGE_viewer_control")


class IGEViewerControl:
    """
    This class is used to control the viewer of the environment.
    The class instantates the viewer with the following parameters:
    - ref_env: reference environment
    - pos: position of the camera
    - lookat: point the camera is looking at (object or body)

    The class also provides methods to control the viewer:
    - set_camera_pos: sets the position of the camera
    - set_camera_lookat: sets the point the camera is looking at (object or body)
    - set_camera_ref_env: sets the reference environment

    """

    def __init__(self, gym, sim, env_manager, config, device):
        self.sim = sim
        self.gym = gym
        self.config = config
        self.env_manager = env_manager
        self.headless = config.headless
        self.device = device

        # Set Camera Properties
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        camera_props.width = self.config.width
        camera_props.height = self.config.height
        camera_props.far_plane = self.config.max_range
        camera_props.near_plane = self.config.min_range
        camera_props.horizontal_fov = self.config.horizontal_fov_deg
        camera_props.use_collision_geometry = self.config.use_collision_geometry
        self.camera_follow_transform_local_offset = torch.tensor(
            self.config.camera_follow_transform_local_offset, device=self.device
        )
        self.camera_follow_position_global_offset = torch.tensor(
            self.config.camera_follow_position_global_offset, device=self.device
        )

        # camera supersampling is unchanged for now
        self.camera_properties = camera_props

        # local transform for camera
        self.local_transform = gymapi.Transform()
        self.local_transform.p = gymapi.Vec3(
            self.config.camera_position[0],
            self.config.camera_position[1],
            self.config.camera_position[2],
        )
        angle_euler = torch.deg2rad(torch.tensor(self.config.camera_orientation_euler_deg))
        angle_quat = quat_from_euler_xyz(angle_euler[0], angle_euler[1], angle_euler[2])
        self.local_transform.r = gymapi.Quat(
            angle_quat[0], angle_quat[1], angle_quat[2], angle_quat[3]
        )

        self.lookat = gymapi.Vec3(
            self.config.lookat[0], self.config.lookat[1], self.config.lookat[2]
        )

        if self.config.camera_follow_type == "FOLLOW_TRANSFORM":
            self.camera_follow_type = gymapi.FOLLOW_TRANSFORM
        elif self.config.camera_follow_type == "FOLLOW_POSITION":
            self.camera_follow_type = gymapi.FOLLOW_POSITION

        self.cam_handle = None

        self.enable_viewer_sync = True
        self.viewer = None
        self.camera_follow = False

        self.camera_image_tensor = None

        self.current_target_env = 0

        self.sync_frame_time = True

        self.pause_sim = False

        self.create_viewer()

    def set_actor_and_env_handles(self, actor_handles, env_handles):
        self.actor_handles = actor_handles
        self.env_handles = env_handles

    def init_tensors(self, global_tensor_dict):
        self.robot_positions = global_tensor_dict["robot_position"]
        self.robot_vehicle_orientations = global_tensor_dict["robot_orientation"]

    def create_viewer(self):
        """
        Create the camera sensor for the viewer. Set the camera properties and attach it to the reference environment.
        """
        logger.debug("Creating viewer")
        if self.headless == True:
            logger.warn("Headless mode enabled. Not creating viewer.")
            return
        # subscribe to keyboard shortcuts
        self.viewer = self.gym.create_viewer(self.sim, self.camera_properties)

        # Set keybindings to events for the viewer.
        # Allow the user to quit the viewer using the ESC key.
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")

        # Allow the user to toggle viewer sync using the V key.
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

        # Sync frame time
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "sync_frame_time")

        # Toggle between follow and unattached camera modes using the F key.
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_F, "toggle_camera_follow")

        # Toggle between follow position and transform modes using the P key.
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_P, "toggle_camera_follow_type"
        )
        # Reset all environments using the R key.
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset_all_envs")
        # Switch target environment up or down using the UP and DOWN keys.
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_UP, "switch_target_env_up")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_DOWN, "switch_target_env_down"
        )
        # Pause simulation using the SPACE key.
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "pause_simulation")
        str_instructions = (
            "Instructions for using the viewer with the keyboard:\n"
            + "ESC: Quit\n"
            + "V: Toggle Viewer Sync\n"
            + "S: Sync Frame Time\n"
            + "F: Toggle Camera Follow\n"
            + "P: Toggle Camera Follow Type\n"
            + "R: Reset All Environments\n"
            + "UP: Switch Target Environment Up\n"
            + "DOWN: Switch Target Environment Down\n"
            + "SPACE: Pause Simulation\n"
        )
        logger.warning(str_instructions)

        return self.viewer

    def handle_keyboard_events(self):
        # check for keyboard events
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.action == "QUIT" and evt.value > 0:
                sys.exit()
            elif evt.action == "reset_all_envs" and evt.value > 0:
                self.reset_all_envs()
            elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                self.toggle_viewer_sync()
            elif evt.action == "toggle_camera_follow" and evt.value > 0:
                self.toggle_camera_follow()
            elif evt.action == "toggle_camera_follow_type" and evt.value > 0:
                self.toggle_camera_follow_type()
            elif evt.action == "switch_target_env_up" and evt.value > 0:
                self.switch_target_env_up()
            elif evt.action == "switch_target_env_down" and evt.value > 0:
                self.switch_target_env_down()
            elif evt.action == "pause_simulation" and evt.value > 0:
                self.pause_simulation()
            elif evt.action == "sync_frame_time" and evt.value > 0:
                self.toggle_sync_frame_time()

    def reset_all_envs(self):
        logger.warning("Resetting all environments.")
        self.env_manager.reset()
        self.env_manager.global_tensor_dict["truncations"][:] = 1

    def toggle_sync_frame_time(self):
        """
        Toggle the sync frame time.
        """
        self.sync_frame_time = not self.sync_frame_time
        logger.warning("Sync frame time: {}".format(self.sync_frame_time))

    def get_viewer_image(self):
        """
        Get the image from the viewer.
        """
        return self.camera_image_tensor

    def toggle_viewer_sync(self):
        """
        Toggle the viewer sync.
        """
        self.enable_viewer_sync = not self.enable_viewer_sync
        logger.warning("Viewer sync: {}".format(self.enable_viewer_sync))

    def toggle_camera_follow_type(self):
        """
        Toggle the camera follow mode.
        """
        self.camera_follow_type = (
            gymapi.FOLLOW_TRANSFORM
            if self.camera_follow_type == gymapi.FOLLOW_POSITION
            else gymapi.FOLLOW_POSITION
        )
        logger.warning("Camera follow type: {}".format(self.camera_follow_type))

    def toggle_camera_follow(self):
        self.camera_follow = not self.camera_follow
        logger.warning("Camera follow: {}".format(self.camera_follow))
        self.set_camera_lookat()

    def switch_target_env_up(self):
        self.current_target_env = (self.current_target_env + 1) % len(self.actor_handles)
        logger.warning("Switching target environment to: {}".format(self.current_target_env))
        self.set_camera_lookat()

    def switch_target_env_down(self):
        self.current_target_env = (self.current_target_env - 1) % len(self.actor_handles)
        logger.warning("Switching target environment to: {}".format(self.current_target_env))
        self.set_camera_lookat()

    def pause_simulation(self):
        self.pause_sim = not self.pause_sim
        logger.warning(
            "Simulation Paused. You can control the viewer at a reduced rate with full keyboard control."
        )
        while self.pause_sim:
            self.render()
            time.sleep(0.1)
            # self.gym.poll_viewer_events(self.viewer)
        return

    def set_camera_lookat(self, pos=None, quat_or_target=None):
        """
        Set the camera position.
        """
        if pos is None:
            pos = self.config.camera_position
        if quat_or_target is None:
            quat_or_target = self.config.lookat
        self.local_transform.p = gymapi.Vec3(pos[0], pos[1], pos[2])
        if self.camera_follow:
            robot_position = self.robot_positions[self.current_target_env]
            robot_vehicle_orientation = self.robot_vehicle_orientations[self.current_target_env]
            self.lookat = gymapi.Vec3(robot_position[0], robot_position[1], robot_position[2])
            if self.camera_follow_type == gymapi.FOLLOW_TRANSFORM:
                viewer_position = robot_position + quat_rotate(
                    robot_vehicle_orientation.unsqueeze(0),
                    self.camera_follow_transform_local_offset.unsqueeze(0),
                ).squeeze(0)
            else:
                viewer_position = robot_position + self.camera_follow_position_global_offset

            self.local_transform.p = gymapi.Vec3(
                viewer_position[0], viewer_position[1], viewer_position[2]
            )
            self.gym.viewer_camera_look_at(
                self.viewer,
                self.env_handles[self.current_target_env],
                self.local_transform.p,
                self.lookat,
            )
        if self.camera_follow == False:
            target_pos = quat_or_target
            self.lookat = gymapi.Vec3(target_pos[0], target_pos[1], target_pos[2])
            self.gym.viewer_camera_look_at(
                self.viewer,
                self.env_handles[self.current_target_env],
                self.local_transform.p,
                self.lookat,
            )

    def render(self):
        """
        Draw the viewer.
        """
        if self.gym.query_viewer_has_closed(self.viewer):
            logger.critical("Viewer has been closed. Exiting simulation.")
            sys.exit()
        self.handle_keyboard_events()
        if self.enable_viewer_sync:
            if self.camera_follow:
                self.set_camera_lookat()
            self.gym.draw_viewer(self.viewer, self.sim, False)
            if self.sync_frame_time:
                self.gym.sync_frame_time(self.sim)
        else:
            self.gym.poll_viewer_events(self.viewer)
