#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty
from mavros_msgs.msg import State
import cv2
import numpy as np
import torch
import struct
from scipy.spatial.transform import Rotation as R

# Sample Factory inference (your standalone file)
from sample_factory_inference import RL_Nav_Interface

# ============================================================================
# Configuration matching lidar_navigation_task training
# ============================================================================
class Config:
    # Image settings (match training)
    IMAGE_HEIGHT = 48  # After (3,6) max pooling from 135
    IMAGE_WIDTH = 80   # After (3,6) max pooling from 480
    IMAGE_MAX_DEPTH = 10.0
    IMAGE_MIN_DEPTH = 0.02
    
    # Observation dimensions
    STATE_DIM = 17
    LIDAR_DIM = 16 * 20  # 320 (downsampled lidar grid)
    TOTAL_OBS_DIM = STATE_DIM + LIDAR_DIM  # 337
    
    # Action dimensions
    ACTION_DIM = 4
    
    # ROS topics
    IMAGE_TOPIC = "/m100/front/depth_image"
    ODOM_TOPIC = "/m100/odometry"
    ACTION_TOPIC = "/m100/command/velocity"
    TARGET_TOPIC = "/target"
    MAVROS_STATE_TOPIC = "/m100/mavros/state"
    
    # Action transformation (match your training config)
    # These should match the action_transformation_function in your task config
    SPEED_SCALE = 1.0
    YAW_RATE_SCALE = 1.0
    
    # Frame IDs
    BODY_FRAME_ID = "m100/base_link"
    
    # Control
    USE_MAVROS_STATE = False
    ACTION_FILTER_ALPHA = 0.8  # EMA filter
    
    # Device
    DEVICE = "cuda:0"  # Default device, can be overridden by command line arg

cfg = Config()

# ============================================================================
# EMA Filter (same as before)
# ============================================================================
class EMA:
    def __init__(self, alpha):
        self.alpha = alpha
        self.value = None
    
    def reset(self):
        self.value = None
    
    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * new_value
        return self.value

# ============================================================================
# Main ROS Node
# ============================================================================
class LidarNavigationNode:
    def __init__(self, checkpoint_path, device="cuda:0"):
        rospy.init_node('lidar_navigation_node')
        
        # Set device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"Using device: {self.device}")
        
        # Load Sample Factory model on specified device
        self.inference = RL_Nav_Interface(checkpoint_path, device=str(self.device))

        # State variables
        self.position = np.zeros(3)
        self.target_position = np.zeros(3)
        self.rpy = np.zeros(3)
        self.body_lin_vel = np.zeros(3)
        self.body_ang_vel = np.zeros(3)
        self.prev_action = np.zeros(cfg.ACTION_DIM)
        self.downsampled_lidar = np.zeros(cfg.LIDAR_DIM)

        self.obs_cpu = torch.zeros(cfg.TOTAL_OBS_DIM, device="cpu", dtype=torch.float32)
        self.obs = torch.zeros(cfg.TOTAL_OBS_DIM, device=self.device, dtype=torch.float32)
        
        # Control state
        self.enable = False
        self.action_filter = EMA(alpha=cfg.ACTION_FILTER_ALPHA)
        
        # Publishers
        self.action_pub = rospy.Publisher(cfg.ACTION_TOPIC, Twist, queue_size=1)
        self.action_viz_pub = rospy.Publisher(cfg.ACTION_TOPIC + "_viz", TwistStamped, queue_size=1)
        self.filtered_action_pub = rospy.Publisher(cfg.ACTION_TOPIC + "_filtered", Twist, queue_size=1)
        
        # Subscribers
        self.image_sub = rospy.Subscriber(cfg.IMAGE_TOPIC, Image, self.image_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber(cfg.ODOM_TOPIC, Odometry, self.odom_callback, queue_size=1)
        self.target_sub = rospy.Subscriber(cfg.TARGET_TOPIC, PoseStamped, self.target_callback, queue_size=1)
        self.reset_sub = rospy.Subscriber("/reset", Empty, self.reset_callback, queue_size=1)
        
        if cfg.USE_MAVROS_STATE:
            self.state_sub = rospy.Subscriber(cfg.MAVROS_STATE_TOPIC, State, self.state_callback, queue_size=1)
        
        rospy.loginfo("Lidar Navigation Node initialized")
    
    def odom_callback(self, msg):
        """Extract odometry data"""
        self.position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        
        # Extract orientation (quaternion -> euler)
        q = msg.pose.pose.orientation
        rot = R.from_quat([q.x, q.y, q.z, q.w])
        self.rpy = rot.as_euler('xyz', degrees=False)
        
        # Body frame velocities
        self.body_lin_vel = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])
        
        self.body_ang_vel = np.array([
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z
        ])
    
    def target_callback(self, msg):
        """Update target position"""
        self.target_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        # Reset on new target
        self.inference.reset()
        self.prev_action = np.zeros(cfg.ACTION_DIM)
        self.action_filter.reset()
        rospy.loginfo(f"New target: {self.target_position}")
    
    def reset_callback(self, msg):
        """Reset network state"""
        self.inference.reset()
        self.prev_action = np.zeros(cfg.ACTION_DIM)
        self.action_filter.reset()
        rospy.loginfo("Network reset")
    
    def state_callback(self, msg):
        """Check MAVROS state"""
        if cfg.USE_MAVROS_STATE:
            was_enabled = self.enable
            self.enable = (msg.mode == "OFFBOARD" or msg.mode == "GUIDED")
            if self.enable and not was_enabled:
                self.reset_callback(None)
    
    def process_depth_image(self, depth_image):
        """
        Process depth image to match training:
        1. Normalize to [0, 1]
        2. Clamp invalid pixels
        3. Downsample using max pooling to 16x20
        
        Returns torch tensor on GPU (NOT numpy!)
        """
        # Normalize
        depth_image = depth_image / cfg.IMAGE_MAX_DEPTH
        depth_image[depth_image > 1.0] = 1.0
        depth_image[depth_image < cfg.IMAGE_MIN_DEPTH / cfg.IMAGE_MAX_DEPTH] = 1.0
        
        # Convert to torch tensor and send to GPU (SINGLE TRANSFER)
        depth_tensor = torch.from_numpy(depth_image).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Downsample using max pooling (3, 6) kernel on GPU
        # Input: (1, 1, H, W) -> Output: (1, 1, 16, 20)
        downsampled = 1 - torch.nn.functional.max_pool2d(
            1 - depth_tensor, kernel_size=(3, 6)
        )
        
        # Flatten to 320-dim vector, on CPU
        return downsampled.squeeze().flatten()
    
    def prepare_observation(self):
        """
        Prepare observation matching lidar_navigation_task:
        - [0:3]: unit vector to target (in vehicle frame)
        - [3]: distance to target
        - [4]: roll
        - [5]: pitch
        - [6]: 0.0 (placeholder)
        - [7:10]: body linear velocity
        - [10:13]: body angular velocity
        - [13:17]: previous actions
        - [17:337]: downsampled lidar (320 dims)
        
        Optimized: Fill state obs on CPU, transfer ONCE to GPU along with lidar
        """
        # Compute vector to target in vehicle frame (yaw-only rotation)
        vec_to_target = self.target_position - self.position
        vehicle_yaw = self.rpy[2]
        vehicle_rot = R.from_euler('z', vehicle_yaw)
        vec_to_target_vehicle = vehicle_rot.inv().apply(vec_to_target)
        
        # Distance to target
        dist_to_target = np.linalg.norm(vec_to_target_vehicle)
        
        # Unit vector to target
        unit_vec_to_target = vec_to_target_vehicle / (dist_to_target + 1e-6)
        
        # Fill state observation on CPU first (17 dims)
        self.state_obs_cpu[0:3] = torch.from_numpy(unit_vec_to_target).float()
        self.state_obs_cpu[3] = dist_to_target
        self.state_obs_cpu[4] = self.rpy[0]  # roll
        self.state_obs_cpu[5] = self.rpy[1]  # pitch
        self.state_obs_cpu[6] = 0.0
        self.state_obs_cpu[7:10] = torch.from_numpy(self.body_lin_vel).float()
        self.state_obs_cpu[10:13] = torch.from_numpy(self.body_ang_vel).float()
        self.state_obs_cpu[13:17] = torch.from_numpy(self.prev_action).float()
        
        # Transfer state obs to GPU and fill into obs_gpu (SINGLE TRANSFER)
        self.obs_gpu[0:cfg.STATE_DIM] = self.state_obs_cpu.to(self.device)
        
        # Fill lidar (already on GPU, no transfer!)
        self.obs_gpu[cfg.STATE_DIM:] = self.lidar_tensor
        
        return self.obs_gpu
    
    def transform_action(self, action):
        """
        Transform network output to velocity commands
        Match your training's action_transformation_function
        """
        # Assuming action is in [-1, 1] range
        # Transform to body frame velocities
        vel_x = (action[0] + 1) * cfg.SPEED_SCALE
        vel_y = action[1] * cfg.SPEED_SCALE
        vel_z = action[2] * cfg.SPEED_SCALE
        yaw_rate = action[3] * cfg.YAW_RATE_SCALE
        
        return np.array([vel_x, vel_y, vel_z, yaw_rate])
    
    def publish_action(self, action):
        """Publish action as Twist message"""
        # Transform action
        vel_cmd = self.transform_action(action)
        
        # Apply EMA filter
        filtered_vel = self.action_filter.update(vel_cmd)
        
        # Create Twist message
        twist_msg = Twist()
        twist_msg.linear.x = filtered_vel[0]
        twist_msg.linear.y = filtered_vel[1]
        twist_msg.linear.z = filtered_vel[2]
        twist_msg.angular.z = filtered_vel[3]
        
        # Publish
        self.filtered_action_pub.publish(twist_msg)
        self.action_pub.publish(twist_msg)
        
        # Publish visualization
        viz_msg = TwistStamped()
        viz_msg.header.stamp = rospy.Time.now()
        viz_msg.header.frame_id = cfg.BODY_FRAME_ID
        viz_msg.twist = twist_msg
        self.action_viz_pub.publish(viz_msg)
    
    def image_callback(self, msg):
        """Main control loop triggered by image"""
        if not self.enable and cfg.USE_MAVROS_STATE:
            # Publish zero command
            self.publish_action(np.array([-1.0, 0.0, 0.0, 0.0]))
            return
        
        # Convert ROS Image to numpy
        if msg.encoding == "32FC1":  # Simulation
            depth_image = np.array(struct.unpack("f" * msg.height * msg.width, msg.data))
            depth_image = depth_image.reshape((msg.height, msg.width))
            depth_image[np.isnan(depth_image)] = cfg.IMAGE_MAX_DEPTH
        else:  # Real camera
            depth_image = np.ndarray((msg.height, msg.width), "<H", msg.data, 0)
            depth_image = depth_image.astype('float32') * 0.001
            depth_image[np.isnan(depth_image)] = cfg.IMAGE_MAX_DEPTH
        
        # Process image on GPU (returns GPU tensor)
        self.lidar_tensor = self.process_depth_image(depth_image)
        
        # Prepare observation on GPU (returns GPU tensor)
        obs_tensor_gpu = self.prepare_observation()
        
        # Get action from network (input is GPU tensor, output is numpy on CPU)
        action = self.inference.get_action(obs_tensor_gpu, normalize=True)
        
        # Store for next iteration
        self.prev_action = action
        
        # Publish action
        self.publish_action(action)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to Sample Factory checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use for inference (cuda:0, cuda:1, cpu, etc.)")
    args = parser.parse_args()
    
    node = LidarNavigationNode(args.checkpoint, device=args.device)
    rospy.spin()