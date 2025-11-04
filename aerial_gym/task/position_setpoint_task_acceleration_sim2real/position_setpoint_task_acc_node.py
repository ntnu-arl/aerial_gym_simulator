#!/usr/bin/env python3

import rospy
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import math

# ROS message types
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import PositionTarget, State
from std_msgs.msg import Empty

# TF for coordinate transformations
from tf.transformations import euler_from_quaternion, quaternion_from_euler


class AccelerationControlNetwork(nn.Module):
    """
    Neural network for acceleration-based quadrotor control.
    Based on the network architecture from the rl_games examples.
    """
    def __init__(self, input_dim=17, output_dim=4, hidden_dims=[256, 128, 64]):
        super(AccelerationControlNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ELU()
            ])
            prev_dim = hidden_dim
        
        # Output layer for acceleration commands (x, y, z, yaw_rate)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def load_network(self, path):
        """Load network weights from checkpoint file"""
        try:
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
            
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
            
            # Clean the state dict to match our network structure
            cleaned_state_dict = OrderedDict()
            
            for key, value in state_dict.items():
                # Skip non-actor network parameters
                if "value" in key or "sigma" in key or "critic" in key:
                    continue
                    
                # Map rl_games network keys to our network structure
                if "a2c_network.actor_mlp" in key:
                    new_key = key.replace("a2c_network.actor_mlp.", "network.")
                elif "a2c_network.mu" in key:
                    new_key = key.replace("a2c_network.mu", f"network.{len(self.network)-1}")
                elif "actor_mlp" in key:
                    new_key = key.replace("actor_mlp.", "network.")
                elif "mu." in key:
                    new_key = key.replace("mu.", f"network.{len(self.network)-1}.")
                else:
                    continue
                    
                cleaned_state_dict[new_key] = value
            
            self.load_state_dict(cleaned_state_dict, strict=False)
            print(f"Successfully loaded network from {path}")
            
        except Exception as e:
            print(f"Failed to load network from {path}: {str(e)}")
            raise
    
    def forward(self, x):
        return self.network(x)


class QuadrotorAccelerationControlNode:
    """
    Simplified ROS node for quadrotor control using neural network acceleration commands.
    Similar to sample_factory_ros_node but for acceleration control with MAVROS setpoint_raw/local.
    """
    
    def __init__(self, network_path):
        # Initialize state variables
        self.current_position = np.zeros(3)
        self.current_velocity = np.zeros(3)
        self.current_orientation = np.array([0, 0, 0, 1])  # quaternion [x,y,z,w]
        self.current_angular_velocity = np.zeros(3)
        self.target_position = np.zeros(3)
        self.prev_action = np.zeros(4)
        
        # MAVROS state
        self.mavros_state = State()
        self.enable = False
        
        # Load neural network
        self.network = AccelerationControlNetwork()
        self.network.load_network(network_path)
        self.network.eval()
        
        # ROS Publishers
        self.command_pub = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size=1)
        print("Publishing acceleration commands to: /mavros/setpoint_raw/local")
        
        # ROS Subscribers
        self.odom_sub = rospy.Subscriber('/mavros/local_position/odom', Odometry, self.odom_callback, queue_size=1)
        print("Subscribed to odometry: /mavros/local_position/odom")
        
        self.target_sub = rospy.Subscriber('/target_position', PoseStamped, self.target_callback, queue_size=1)
        print("Subscribed to target: /target_position")
        
        self.mavros_state_sub = rospy.Subscriber('/mavros/state', State, self.mavros_state_callback, queue_size=1)
        print("Subscribed to MAVROS state: /mavros/state")
        
        self.reset_sub = rospy.Subscriber("/reset", Empty, self.reset_callback, queue_size=1)
        
        # Control timer (50 Hz)
        self.control_timer = rospy.Timer(rospy.Duration(0.02), self.control_callback)
        
        print("Quadrotor Acceleration Control Node initialized")
        print("FRAME CONVENTIONS (matching training script):")
        print("- Position errors: Transformed to BODY frame")
        print("- Velocities: In BODY frame")
        print("- Euler angles: Roll and pitch only")
        print("- Angular velocities: In BODY frame")
        print("- Actions: 4D [acc_x, acc_y, acc_z, yaw_rate] in BODY_NED frame")
    
    def reset_callback(self, msg):
        """Reset network state and previous actions"""
        self.prev_action = np.zeros(4)
        print("Network state reset")
    
    def odom_callback(self, odom):
        """Process odometry data"""
        self.current_position[0] = odom.pose.pose.position.x
        self.current_position[1] = odom.pose.pose.position.y
        self.current_position[2] = odom.pose.pose.position.z
        
        # Extract orientation (quaternion)
        self.current_orientation = np.array([
            odom.pose.pose.orientation.x,
            odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z,
            odom.pose.pose.orientation.w
        ])
        
        # Extract linear velocity (assuming body frame from MAVROS)
        self.current_velocity[0] = odom.twist.twist.linear.x
        self.current_velocity[1] = odom.twist.twist.linear.y
        self.current_velocity[2] = odom.twist.twist.linear.z
        
        # Extract angular velocity (assuming body frame from MAVROS)
        self.current_angular_velocity[0] = odom.twist.twist.angular.x
        self.current_angular_velocity[1] = odom.twist.twist.angular.y
        self.current_angular_velocity[2] = odom.twist.twist.angular.z
    
    def target_callback(self, target):
        """Process target position"""
        self.target_position[0] = target.pose.position.x
        self.target_position[1] = target.pose.position.y
        self.target_position[2] = target.pose.position.z
        # Reset network state when new target is received
        self.prev_action = np.zeros(4)
    
    def mavros_state_callback(self, data):
        """Process MAVROS state"""
        self.mavros_state = data
        # Enable control in OFFBOARD or GUIDED mode
        if data.mode == "OFFBOARD" or data.mode == "GUIDED":
            if not self.enable:
                self.prev_action = np.zeros(4)
                print("Control enabled - MAVROS in OFFBOARD/GUIDED mode")
            self.enable = True
        else:
            self.enable = False
    
    def transform_to_body_frame(self, vector):
        """Transform a vector from world frame to body frame using current orientation"""
        # Quaternion format: [x, y, z, w]
        qx, qy, qz, qw = self.current_orientation
        
        # Quaternion rotation (inverse rotation from world to body frame)
        # v_body = q_inv * v_world * q where q_inv is quaternion conjugate
        q_w_sq = qw * qw
        q_vec = np.array([qx, qy, qz])
        
        v_rotated = vector * (2.0 * q_w_sq - 1.0) - 2.0 * qw * np.cross(q_vec, vector) + 2.0 * q_vec * np.dot(q_vec, vector)
        
        return v_rotated
    
    def prepare_observation(self):
        """
        Prepare observation tensor for the neural network.
        Based on the observation structure from position_setpoint_task_acceleration_sim2real.py
        
        Observation structure (17 elements):
        [0:3]   - Position error in body frame 
        [3:5]   - Roll and pitch (Euler angles)
        [5]     - Yaw error (target_yaw - current_yaw) 
        [6]     - Reserved (set to 0.0)
        [7:10]  - Linear velocity in body frame
        [10:13] - Angular velocity in body frame  
        [13:17] - Previous actions (4 elements)
        """
        # Position error (target - current) transformed to body frame
        pos_error_world = self.target_position - self.current_position
        pos_error_body = self.transform_to_body_frame(pos_error_world)
        
        # Current roll, pitch, yaw
        roll, pitch, yaw = euler_from_quaternion(self.current_orientation)
        
        # Yaw error (assuming target yaw = 0 for now)
        yaw_error = -yaw  # Target yaw = 0, so error is -current_yaw
        # Normalize yaw error to [-pi, pi]
        while yaw_error > np.pi:
            yaw_error -= 2 * np.pi
        while yaw_error < -np.pi:
            yaw_error += 2 * np.pi
        
        # Velocities are already in body frame from MAVROS odometry
        velocity_body = self.current_velocity
        angular_vel_body = self.current_angular_velocity
        
        # Combine all observations according to training script format
        obs = np.concatenate([
            pos_error_body,           # [0:3] - Position error in body frame
            np.array([roll, pitch]),  # [3:5] - Roll and pitch only
            np.array([yaw_error]),    # [5] - Yaw error
            np.array([0.0]),          # [6] - Reserved
            velocity_body,            # [7:10] - Linear velocity in body frame
            angular_vel_body,         # [10:13] - Angular velocity in body frame  
            self.prev_action          # [13:17] - Previous actions
        ])
        
        return torch.from_numpy(obs).float().unsqueeze(0)  # Add batch dimension
    
    def publish_acceleration_command(self, action):
        """Create and publish PositionTarget message for MAVROS acceleration control"""
        msg = PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"
        
        # Set coordinate frame to body_NED for body frame control
        msg.coordinate_frame = PositionTarget.FRAME_BODY_NED
        
        # Set type mask for acceleration control (ignore position, velocity, yaw)
        msg.type_mask = (
            PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_PZ |
            PositionTarget.IGNORE_VX | PositionTarget.IGNORE_VY | PositionTarget.IGNORE_VZ |
            PositionTarget.IGNORE_YAW
        )
        
        # Set acceleration commands
        msg.acceleration_or_force.x = action[0]
        msg.acceleration_or_force.y = action[1] 
        msg.acceleration_or_force.z = action[2]
        
        # Set yaw rate
        msg.yaw_rate = action[3]
        
        self.command_pub.publish(msg)
    
    def publish_velocity_command(self, action):
        """Create and publish PositionTarget message for MAVROS velocity control"""
        msg = PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"
        
        # Set coordinate frame to body_NED for body frame control
        msg.coordinate_frame = PositionTarget.FRAME_BODY_NED
        
        # Set type mask for velocity control (ignore position, acceleration, yaw)
        msg.type_mask = (
            PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_PZ |
            PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
            PositionTarget.IGNORE_YAW
        )
        
        # Set velocity commands
        msg.velocity.x = action[0]
        msg.velocity.y = action[1] 
        msg.velocity.z = action[2]
        
        # Set yaw rate
        msg.yaw_rate = action[3]
        
        self.command_pub.publish(msg)
    
    def transform_action(self, action):
        acc_scale = 2.0
        acc_clip = 1.0
        yaw_clip = 1.0
        yaw_scale = 1.0

        # Scale and clip acceleration commands
        acc_cmd = np.clip(action[0:3] * acc_scale, -acc_clip, acc_clip)
        yaw_cmd = np.clip(action[3] * yaw_scale, -yaw_clip, yaw_clip)
        return np.concatenate([acc_cmd, [yaw_cmd]])

    def control_callback(self, event):
        """Main control loop callback"""
        if not self.enable:
            # Send zero commands when not enabled
            self.publish_acceleration_command(np.zeros(4))
            return
        
        try:
            # Prepare observation for neural network
            obs = self.prepare_observation()
            
            # Run neural network inference
            with torch.no_grad():
                action = self.network(obs)
                action = self.transform_action(action.numpy().flatten())
            
            # Convert to numpy and flatten
            action = action.detach()
            
            # Store previous action for next observation
            
            # Publish acceleration command
            self.publish_acceleration_command(action)
            self.prev_action = action.copy()
            
        except Exception as e:
            print(f"Control callback error: {str(e)}")
            # Send zero commands on error
            self.publish_velocity_command(np.zeros(4))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--network_path', type=str, required=True, 
                       help='Path to the neural network checkpoint file')
    args = parser.parse_args()
    
    # Initialize ROS node
    rospy.init_node('quadrotor_acceleration_control_node')
    
    print("Loading network from:", args.network_path)
    node = QuadrotorAccelerationControlNode(args.network_path)
    
    print("Node started. Waiting for MAVROS state and odometry...")
    print("Put MAVROS in OFFBOARD mode to enable control")
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Node stopped")


