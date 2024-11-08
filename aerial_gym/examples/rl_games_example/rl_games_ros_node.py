#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TwistStamped, Twist
from mavros_msgs.msg import PositionTarget

from rl_games_inference import MLP
import torch
import time

# WEIGHTS_PATH = "gen_ppo.pth"
# WEIGHTS_PATH = "vel_control_lmf2.pth"

COMMAND_MODE = "acceleration"  # "velocity" or "acceleration"

if COMMAND_MODE == "velocity":
    WEIGHTS_PATH = "networks/vel_control_lmf2_direct.pth"
    CLIP_VALUE = 1.0
    VELOCITY_ACTION_MAGNITUDE = 1.0  # 0.5
    YAW_RATE_ACTION_MAGNITUDE = 1.0  # 0.5

elif COMMAND_MODE == "acceleration":
    WEIGHTS_PATH = "networks/acc_command_2_multiplier_disturbance.pth"
    # WEIGHTS_PATH = "acc_control_lmf2_direct.pth"
    CLIP_VALUE = 1.0
    VELOCITY_ACTION_MAGNITUDE = 1.5  # 0.5
    YAW_RATE_ACTION_MAGNITUDE = 0.8  # 0.5


ROBOT_BASE_LINK_ID = "base_link"


ODOMETRY_TOPIC = "/mavros/local_position/odom"
GOAL_TOPIC = "/target_position"
COMMAND_TOPIC = "/mavros/setpoint_raw/local"
COMMAND_TOPIC_VIZ = "/mavros/setpoint_raw/local_viz"

STATE_TENSOR_BUFFER_DEVICE = "cpu"  # generally should be cpu
NN_INFERENCE_DEVICE = "cpu"


class RobotPositionControlNode:
    def __init__(self):
        rospy.init_node("robot_position_control_node")

        # Parameters
        self.update_rate = rospy.get_param("~update_rate", 50)  # Hz
        self.max_velocity = rospy.get_param("~max_velocity", 1.0)  # m/s
        self.max_acceleration = rospy.get_param("~max_acceleration", 0.5)  # m/s^2

        # State variables
        self.current_position = torch.zeros(
            3, device=STATE_TENSOR_BUFFER_DEVICE, requires_grad=False
        )
        self.current_orientation = torch.zeros(
            4, device=STATE_TENSOR_BUFFER_DEVICE, requires_grad=False
        )
        self.current_orientation[3] = 1.0
        self.current_body_velocity = torch.zeros(
            3, device=STATE_TENSOR_BUFFER_DEVICE, requires_grad=False
        )
        self.current_body_angular_velocity = torch.zeros(
            3, device=STATE_TENSOR_BUFFER_DEVICE, requires_grad=False
        )
        self.current_state = torch.zeros(13, device=STATE_TENSOR_BUFFER_DEVICE, requires_grad=False)
        self.target_position = None
        self.weights_path = WEIGHTS_PATH
        self.actions = torch.zeros(4, device=STATE_TENSOR_BUFFER_DEVICE, requires_grad=False)
        self.previous_actions = torch.zeros(
            4, device=STATE_TENSOR_BUFFER_DEVICE, requires_grad=False
        )

        self.obs_tensor = torch.zeros(1, 17, device=NN_INFERENCE_DEVICE, requires_grad=False)

        self.controller = (
            MLP(input_dim=17, output_dim=4, path=self.weights_path).to(NN_INFERENCE_DEVICE).eval()
        )

        # Publishers
        self.cmd_pub = rospy.Publisher(COMMAND_TOPIC, PositionTarget, queue_size=1)
        self.cmd_pub_viz = rospy.Publisher(COMMAND_TOPIC_VIZ, TwistStamped, queue_size=1)

        # Subscribers
        rospy.Subscriber(ODOMETRY_TOPIC, Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber(GOAL_TOPIC, PoseStamped, self.goal_callback, queue_size=1)

    def odom_callback(self, msg):
        msgpose = msg.pose.pose
        msgtwist = msg.twist.twist
        # Update current position and velocity
        self.current_position[0] = msgpose.position.x
        self.current_position[1] = msgpose.position.y
        self.current_position[2] = msgpose.position.z

        self.current_body_velocity[0] = msgtwist.linear.x
        self.current_body_velocity[1] = msgtwist.linear.y
        self.current_body_velocity[2] = msgtwist.linear.z

        quat_sign = 1.0 if msgpose.orientation.w >= 0 else -1.0
        self.current_orientation[0] = msgpose.orientation.x
        self.current_orientation[1] = msgpose.orientation.y
        self.current_orientation[2] = msgpose.orientation.z
        self.current_orientation[3] = msgpose.orientation.w

        self.current_orientation[:] = quat_sign * self.current_orientation

        self.current_body_angular_velocity[0] = msgtwist.angular.x
        self.current_body_angular_velocity[1] = msgtwist.angular.y
        self.current_body_angular_velocity[2] = msgtwist.angular.z

        # Update current state
        self.current_state[:] = torch.concatenate(
            [
                self.current_position,
                self.current_orientation,
                self.current_body_velocity,
                self.current_body_angular_velocity,
            ]
        )

    def goal_callback(self, msg):
        # Update target position
        self.target_position = torch.tensor(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            device=STATE_TENSOR_BUFFER_DEVICE,
            requires_grad=False,
        )

    def send_position_target_command(
        self, x_command, y_command, z_command, yaw_rate_command, mode="velocity"
    ):
        msg = PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.coordinate_frame = PositionTarget.FRAME_BODY_NED

        # Ignore position and use velocity
        msg.type_mask = (
            PositionTarget.IGNORE_PX
            + PositionTarget.IGNORE_PY
            + PositionTarget.IGNORE_PZ
            + PositionTarget.IGNORE_YAW
        )
        if mode == "velocity":
            msg.type_mask += (
                PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ
            )
        elif mode == "acceleration":
            msg.type_mask += (
                PositionTarget.IGNORE_VX + PositionTarget.IGNORE_VY + PositionTarget.IGNORE_VZ
            )

        if mode == "velocity":
            # Set velocity
            msg.velocity.x = x_command
            msg.velocity.y = y_command
            msg.velocity.z = z_command
        elif mode == "acceleration":
            # Set acceleration
            msg.acceleration_or_force.x = x_command
            msg.acceleration_or_force.y = y_command
            msg.acceleration_or_force.z = z_command

        # Set yaw rate
        msg.yaw_rate = yaw_rate_command
        self.cmd_pub.publish(msg)

        viz_msg = TwistStamped()
        viz_msg.header.stamp = rospy.Time.now()
        viz_msg.header.frame_id = ROBOT_BASE_LINK_ID
        viz_msg.twist.linear.x = x_command
        viz_msg.twist.linear.y = y_command
        viz_msg.twist.linear.z = z_command
        viz_msg.twist.angular.z = yaw_rate_command
        self.cmd_pub_viz.publish(viz_msg)

    def get_observations_tensor(self, current_state, previous_actions, target_position):
        self.obs_tensor[0, 0:3] = (target_position - current_state[:3]).to(NN_INFERENCE_DEVICE)
        self.obs_tensor[0, 3:7] = current_state[3:7].to(NN_INFERENCE_DEVICE)
        self.obs_tensor[0, 7:10] = current_state[7:10].to(NN_INFERENCE_DEVICE)
        self.obs_tensor[0, 10:13] = current_state[10:13].to(NN_INFERENCE_DEVICE)
        self.obs_tensor[0, 13:17] = previous_actions.to(NN_INFERENCE_DEVICE)
        return self.obs_tensor

    def compute_command(self):
        if self.target_position is None:
            return None
        obs_tensor = self.get_observations_tensor(
            self.current_state, self.previous_actions, self.target_position
        )
        actions = self.controller(obs_tensor)
        return actions

    def filter_actions(self, actions):
        clipped_actions = torch.clip(actions, -CLIP_VALUE, CLIP_VALUE)
        clipped_actions[0] *= VELOCITY_ACTION_MAGNITUDE
        clipped_actions[1] *= VELOCITY_ACTION_MAGNITUDE
        clipped_actions[2] *= VELOCITY_ACTION_MAGNITUDE
        clipped_actions[3] *= YAW_RATE_ACTION_MAGNITUDE
        return clipped_actions

    def run(self):
        rate = rospy.Rate(self.update_rate)

        while not rospy.is_shutdown():
            try:
                start_time = time.time()
                command = self.compute_command()
                if command is None:
                    self.send_position_target_command(0, 0, 0, 0, mode="velocity")
                else:
                    self.actions[:] = command
                    end_time = time.time()
                    print(f"Control loop time: {end_time - start_time}")
                    self.previous_actions[:] = self.actions
                    filtered_actions = self.filter_actions(self.actions).to(
                        STATE_TENSOR_BUFFER_DEVICE
                    )
                    self.send_position_target_command(
                        filtered_actions[0],
                        filtered_actions[1],
                        filtered_actions[2],
                        filtered_actions[3],
                        mode=COMMAND_MODE,
                    )
            except Exception as e:
                rospy.logerr(f"Error in control loop: {e}")
            rate.sleep()


if __name__ == "__main__":
    try:
        node = RobotPositionControlNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
