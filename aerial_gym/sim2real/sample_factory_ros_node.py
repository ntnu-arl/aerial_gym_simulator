import rospy

# import Image, Twist, Odometry, Velocity message from ROS
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, PoseStamped, Vector3, TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray, Empty
from visualization_msgs.msg import Marker

from mavros_msgs.msg import State
import cv2
import numpy as np
import math
import time
import torch
from scipy.spatial.transform import Rotation as R
from config import *
import argparse
import struct

from sample_factory_inference import RL_Nav_Interface
from vae_image_encoder import VAEImageEncoder

from aerial_gym import AERIAL_GYM_DIRECTORY


class vae_config:
    use_vae = True
    latent_dims = 64
    model_file = (
        AERIAL_GYM_DIRECTORY
        + "/aerial_gym/utils/vae/weights/ICRA_test_set_more_sim_data_kld_beta_3_LD_64_epoch_49.pth"
    )
    model_folder = AERIAL_GYM_DIRECTORY
    image_res = (270, 480)
    interpolation_mode = "nearest"
    return_sampled_latent = True


# Exponential moving average filter
class EMA:
    def __init__(self, beta):
        self.beta = beta
        self.value = None

    def reset(self):
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.beta * self.value + (1 - self.beta) * new_value
        return self.value


class RlNavClass:
    def __init__(self, args):
        self.odom = Odometry()
        self.action = np.zeros(ACTION_DIMS)  # Dimensions are to be edited
        self.prev_action = self.action
        self.target = np.zeros(GOAL_DIR_DIMS)
        self.position = np.zeros(GOAL_DIR_DIMS)
        self.rpy = np.zeros(ATTITUDE_DIMS)
        self.body_lin_vel = np.zeros(LIN_VELOCITY_DIMS)
        self.body_ang_vel = np.zeros(ANG_VELOCITY_DIMS)
        self.dist_divider = DIST_DIVIDER
        self.inclination_angle_multiplier = INCLINATION_MULTIPLER
        self.yaw_rate_multipler = YAW_RATE_MULTIPLIER
        self.decoded_image_topic = DECODED_IMAGE_TOPIC_NAME
        self.input_image_topic = IMAGE_TOPIC
        self.odom_topic = ODOM_TOPIC
        self.action_topic = ACTION_PUB_TOPIC
        self.mavros_state_topic = MAVROS_STATE_TOPIC
        self.sample_from_latent = SAMPLE_FROM_LATENT
        self.target_topic = TARGET_TOPIC
        self.latent_space_topic = LATENT_SPACE_TOPIC_NAME
        self.body_frame_dir_topic = BODY_FRAME_DIR_TOPIC
        self.goal_dir_topic = GOAL_DIR_TOPIC_NAME
        self.device = device
        self.use_mavros_state = USE_MAVROS_STATE
        self.enable = False
        self.action_filter = EMA(beta=ACTION_FILTER_BETA)
        # Initialize networks
        self.RL_net_interface = RL_Nav_Interface()
        self.VAE_net_interface = VAEImageEncoder(config=vae_config, device=self.device)

        ## ROS Interfaces

        ## Publishers
        self.decode_img_publisher = rospy.Publisher(self.decoded_image_topic, Image, queue_size=1)
        print("Publishing decoded image to: ", self.decoded_image_topic)
        self.filtered_image_publisher = rospy.Publisher(
            self.input_image_topic + "_filtered", Image, queue_size=1
        )
        print("Publishing filtered image to: ", self.input_image_topic + "_filtered")

        # Command
        self.action_pub = rospy.Publisher(self.action_topic, Twist, queue_size=1)
        print("Publishing action to: ", self.action_topic)
        # Latent space as Float64MultiArray
        self.latent_space_publisher = rospy.Publisher(
            self.latent_space_topic, Float32MultiArray, queue_size=2
        )
        print("Publishing latent space to: ", self.latent_space_topic)

        self.current_direction_body_pub = rospy.Publisher(
            self.body_frame_dir_topic, PoseStamped, queue_size=1
        )
        print("Publishing relative direction to: ", self.body_frame_dir_topic)

        self.goal_dir_publisher = rospy.Publisher(self.goal_dir_topic, Marker, queue_size=1)
        print("Publishing goal dir marker to: ", self.goal_dir_topic)

        self.action_viz_pub = rospy.Publisher(
            self.action_topic + "_viz", TwistStamped, queue_size=1
        )
        print("Publishing action viz to: ", self.action_topic + "_viz")

        self.filtered_action_viz_pub = rospy.Publisher(
            self.action_topic + "_filtered_viz", TwistStamped, queue_size=1
        )
        print("Publishing action viz to: ", self.action_topic + "_filtered_viz")

        self.filtered_action_pub = rospy.Publisher(
            self.action_topic + "_filtered", Twist, queue_size=1
        )
        print("Publishing action viz to: ", self.action_topic + "_filtered")

        ## Subscribers
        # Image
        self.image_sub = rospy.Subscriber(
            self.input_image_topic, Image, self.image_callback, queue_size=1
        )
        print("Subscribed to: ", self.input_image_topic)

        # Odom
        self.odom_sub = rospy.Subscriber(
            self.odom_topic, Odometry, self.odom_callback, queue_size=1
        )
        print("Subscribed to: ", self.odom_topic)

        self.mavros_state_sub = rospy.Subscriber(
            self.mavros_state_topic, State, self.mavros_state_callback, queue_size=1
        )
        print("Subscribed to: ", self.mavros_state_topic)

        self.target_sub = rospy.Subscriber(
            self.target_topic, PoseStamped, self.target_callback, queue_size=1
        )
        print("Subscribed to: ", self.target_topic)

        self.reset_sub = rospy.Subscriber("/reset", Empty, self.reset_callback, queue_size=1)

        self.odom_timer = rospy.timer.Timer(rospy.Duration(0.5), self.timer_cb)

    def timer_cb(self, event):
        print("No odom received")

    def target_callback(self, target):
        self.target[0] = target.pose.position.x
        self.target[1] = target.pose.position.y
        self.target[2] = target.pose.position.z
        # Also reset Network state when a new target is received
        self.RL_net_interface.reset()
        self.prev_action = np.array([0.0, 0.0, 0.0, 0.0])

    def reset_callback(self, msg):
        self.RL_net_interface.reset()
        self.prev_action = np.array([0.0, 0.0, 0.0, 0.0])
        self.action = np.zeros(ACTION_DIMS)
        self.prev_action = self.action
        self.action_filter.reset()

    def odom_callback(self, odom):
        self.odom = odom
        self.position[0] = self.odom.pose.pose.position.x
        self.position[1] = self.odom.pose.pose.position.y
        self.position[2] = self.odom.pose.pose.position.z
        # print("Current position: ", self.position)
        # print("current orientation: ", self.odom.pose.pose.orientation)
        quat_msg = self.odom.pose.pose.orientation
        quat = R.from_quat([quat_msg.x, quat_msg.y, quat_msg.z, quat_msg.w])
        ### TODO: Check if this is the correct order for the euler angles
        rpy = quat.as_euler("xyz", degrees=False)

        self.rpy[0] = rpy[0]
        self.rpy[1] = rpy[1]
        self.rpy[2] = rpy[2]

        self.vehicle_rpy = self.rpy.copy()
        self.vehicle_rpy[2] = 0.0
        self.vehicle_frame_matrix = R.from_euler("xyz", self.rpy, degrees=False)
        # print(self.rpy)

        self.body_lin_vel = np.zeros(3)
        self.body_lin_vel[0] = odom.twist.twist.linear.x
        self.body_lin_vel[1] = odom.twist.twist.linear.y
        self.body_lin_vel[2] = odom.twist.twist.linear.z

        self.body_ang_vel[0] = odom.twist.twist.angular.x
        self.body_ang_vel[1] = odom.twist.twist.angular.y
        self.body_ang_vel[2] = odom.twist.twist.angular.z
        self.odom_timer.shutdown()
        self.odom_timer = rospy.timer.Timer(rospy.Duration(0.5), self.timer_cb)

    def process_image(self, image, to_torch=True, device="cpu"):
        image[image > IMAGE_MAX_DEPTH] = IMAGE_MAX_DEPTH
        image[image < IMAGE_MIN_DEPTH] = -1

        image = image / IMAGE_MAX_DEPTH
        image[image < IMAGE_MIN_DEPTH / IMAGE_MAX_DEPTH] = -1.0
        if to_torch:
            image = torch.from_numpy(image).float()
            image = (
                torch.nn.functional.interpolate(
                    image.unsqueeze(0).unsqueeze(0), (IMAGE_HEIGHT, IMAGE_WIDTH)
                )
                .squeeze(0)
                .squeeze(0)
            )
            image = image.to(device)
            filtered_image = image.clone()
        return filtered_image, image

    def publish_action(self, action=None):
        if action is None:
            action = self.action

        # check if action is a torch tensor or np array
        if isinstance(action, np.ndarray):
            action[action > 1.0] = 1.0
            action[action < -1.0] = -1.0
        elif isinstance(action, torch.Tensor):
            action = torch.clamp(action, -1.0, 1.0).cpu().numpy()
        else:
            print(
                "Weird bug. Action is neither a torch tensor nor a numpy array. Action is of type: ",
                type(action),
            )

        action_npy = action
        vel_x = (
            SPEED_MAGNITUDE
            * (action_npy[0] + 1.0)
            * math.cos(self.inclination_angle_multiplier * action_npy[1])
            / 2.0
        )
        vel_y = 0.0
        vel_z = (
            SPEED_MAGNITUDE
            * (action_npy[0] + 1.0)
            * math.sin(self.inclination_angle_multiplier * action_npy[1])
            / 2.0
        )
        yaw_rate = action_npy[2] * self.yaw_rate_multipler

        if yaw_rate > MAX_YAW_RATE:
            yaw_rate = MAX_YAW_RATE
        if yaw_rate < -MAX_YAW_RATE:
            yaw_rate = -MAX_YAW_RATE

        # self action 4_d array
        self.prev_action = np.array([vel_x, vel_y, vel_z, yaw_rate])
        filtered_action = self.action_filter.update(self.prev_action)
        filtered_vx = filtered_action[0]
        filtered_vy = filtered_action[1]
        filtered_vz = filtered_action[2]
        filtered_yaw_rate = filtered_action[3]

        # assuming everything else is 0 at initialization
        action_msg = Twist()
        action_msg.linear.x = vel_x
        action_msg.linear.z = vel_z
        action_msg.angular.z = yaw_rate
        # self.action_pub.publish(action_msg)
        action_viz_msg = TwistStamped()
        action_viz_msg.header.frame_id = BODY_FRAME_ID
        action_viz_msg.twist = action_msg
        self.action_viz_pub.publish(action_viz_msg)

        filtered_action_msg = Twist()
        filtered_action_msg.linear.x = filtered_vx
        filtered_action_msg.linear.z = filtered_vz
        filtered_action_msg.angular.z = filtered_yaw_rate
        self.filtered_action_pub.publish(filtered_action_msg)
        filtered_action_viz_msg = TwistStamped()
        filtered_action_viz_msg.header.frame_id = BODY_FRAME_ID
        filtered_action_viz_msg.twist = filtered_action_msg
        self.filtered_action_viz_pub.publish(filtered_action_viz_msg)

        if USE_FILTERED_ACTIONS:
            self.action_pub.publish(filtered_action_msg)
        else:
            self.action_pub.publish(action_msg)

        # Action viz message for plotjuggler
        self.action_viz_pub.publish(action_viz_msg)

        goal_dir_marker = Marker()
        goal_dir_marker.header.stamp = rospy.Time.now()
        goal_dir_marker.header.frame_id = BODY_FRAME_ID
        # marker type arrow
        goal_dir_marker.type = 0
        # start point is 0,0,0
        goal_dir_marker.points.append(Vector3(0.0, 0.0, 0.0))
        # end point is the goal direction
        goal_dir_marker.points.append(Vector3(self.goal_dir[0], self.goal_dir[1], self.goal_dir[2]))
        goal_dir_marker.scale.x = 0.1
        goal_dir_marker.scale.y = 0.3
        goal_dir_marker.scale.z = 0.2
        goal_dir_marker.color.a = 1.0
        goal_dir_marker.color.r = 0.0
        goal_dir_marker.color.g = 0.0
        goal_dir_marker.color.b = 1.0

        self.goal_dir_publisher.publish(goal_dir_marker)

    def mavros_state_callback(self, data):
        # check if mavros state is "OFFBOARD" or "GUIDED".
        # if it is either of them, set enable to True, otherwise set it to False
        if self.use_mavros_state:
            if data.mode == "OFFBOARD" or data.mode == "GUIDED":
                if self.enable == False:
                    self.RL_net_interface.reset()
                    self.prev_action = np.array([0.0, 0.0, 0.0, 0.0])
                    print("[DONE] Resetting network.")
                self.enable = True
            else:
                self.enable = False
        else:
            self.enable = True

    def prepare_state_input_tensor(self):
        nn_input = np.zeros(TOTAL_IP_DIMS)
        self.goal_dir = self.target - self.position
        self.goal_dir = self.vehicle_frame_matrix.inv().apply(self.goal_dir)
        # print("Goal dir: ", self.goal_dir)
        # print("Target: ", self.target)
        # print("Position: ", self.position)

        self.current_dir_msg = PoseStamped()
        self.current_dir_msg.pose.position.x = self.position[0]
        self.current_dir_msg.pose.position.y = self.position[1]
        self.current_dir_msg.pose.position.z = self.position[2]
        self.current_dir_msg.header.frame_id = "base_link"
        self.current_dir_msg.pose.orientation.w = 1.0
        self.current_direction_body_pub.publish(self.current_dir_msg)

        if self.use_mavros_state == False:
            self.enable = True
        if self.enable == False:
            # publish a zero action
            self.publish_action(np.array([-1.0, 0.0, 0.0]))
            return
        nn_input = np.zeros(TOTAL_IP_DIMS)
        nn_input[0:3] = self.goal_dir / np.linalg.norm(self.goal_dir)
        nn_input[3] = np.linalg.norm(self.goal_dir) / self.dist_divider
        nn_input[4] = self.rpy[0]
        nn_input[5] = self.rpy[1]
        nn_input[6] = 0.0
        nn_input[7:10] = self.body_lin_vel
        nn_input[10:13] = self.body_ang_vel
        nn_input[13:17] = self.prev_action

        self.nn_input_tensor = torch.from_numpy(nn_input).float().to(self.device)
        self.nn_input_tensor[17 : 17 + LATENT_SPACE] = self.img_latent
        self.nn_input_tensor = self.nn_input_tensor.unsqueeze(0).to(self.device)

    def image_callback(self, data):
        image_cb_start = time.time()
        # # convert from ROS image to torch image
        if IS_SIM:
            cv_image = np.array(struct.unpack("f" * data.height * data.width, data.data))
            cv_image = cv_image.reshape((data.height, data.width))
            cv_image[np.isnan(cv_image)] = IMAGE_MAX_DEPTH
        else:
            cv_image = np.ndarray((data.height, data.width), "<H", data.data, 0)
            cv_image = cv_image.astype("float32") * 0.001  # convert pixel value to meter
            cv_image[np.isnan(cv_image)] = IMAGE_MAX_DEPTH  # max_range

        # send image to GPU on torch
        np_image = cv_image.copy()
        np_image.setflags(write=1)
        filtered_image, input_image = self.process_image(
            np_image, to_torch=True, device=self.device
        )
        self.img_latent = self.VAE_net_interface.encode(filtered_image.unsqueeze(0).unsqueeze(0))
        # prepare state_input_tensor
        self.prepare_state_input_tensor()
        obs = {
            "observations": self.nn_input_tensor,
        }

        self.action = self.RL_net_interface.step(obs)[
            0
        ]  # action is expected to be 2 dimensional even for a single agent

        self.publish_action()
        self.action = np.array([-1.0, 0.0, 0.0])
        latent_space_msg = Float32MultiArray()
        latent_space_msg.data = self.img_latent.flatten().tolist()
        self.latent_space_publisher.publish(latent_space_msg)
        image_cb_end = time.time()
        return


if __name__ == "__main__":
    rospy.init_node("vae_interface")

    # Use argparser to distinguish between simulation and real world
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", default="True")
    parser.add_argument("--show_cv", default="False")
    args, unknown_args = parser.parse_known_args()

    # if args.sim == "False":
    #     IS_SIM = False
    #     IMAGE_TOPIC = "/d455/depth/image_rect_raw"
    #     ODOM_TOPIC = "/mavros/local_position/odom_in_map"
    #     ACTION_PUB_TOPIC = "/mavros/setpoint_velocity/cmd_vel_unstamped"
    #     MAVROS_STATE_TOPIC = "/mavros/state"
    #     TARGET_TOPIC = "/target"
    #     CALCULATE_RECONSTRUCTION = True
    #     TARGET_FRAME_ID = "map"
    #     BODY_FRAME_ID = "state"
    #     MAVROS_STATE_TOPIC = "/mavros/state"
    #     USE_MAVROS_STATE = True
    #     BODY_FRAME_DIR_TOPIC = "/current_direction"
    #     ODOM_FRAME_ID = "t265_pose_frame"
    #     PUBLISH_FILTERED_IMAGE = True

    # else:
    # IS_SIM = True
    # IMAGE_TOPIC = "/m100/depth_image"
    # ODOM_TOPIC = "/m100/odom"
    # ACTION_PUB_TOPIC = "/m100/cmd_vel"
    # MAVROS_STATE_TOPIC = "/m100/mavros/state"
    # CALCULATE_RECONSTRUCTION = True
    # USE_MAVROS_STATE = False
    # TARGET_FRAME_ID = "empty_world"
    # BODY_FRAME_DIR_TOPIC = "/current_direction"

    # BODY_FRAME_ID = "m100/base_link"
    # ODOM_FRAME_ID = "m100/base_link"
    # PUBLISH_FILTERED_IMAGE = False

    IS_SIM = True
    IMAGE_TOPIC = "/m100/front/depth_image"
    ODOM_TOPIC = "/m100/odometry"
    ACTION_PUB_TOPIC = "/m100/command/velocity"
    MAVROS_STATE_TOPIC = "/m100/mavros/state"
    CALCULATE_RECONSTRUCTION = False
    USE_MAVROS_STATE = False
    TARGET_FRAME_ID = "cosmos"
    BODY_FRAME_DIR_TOPIC = "/current_direction"

    BODY_FRAME_ID = "m100/base_link"
    ODOM_FRAME_ID = "m100/base_link"
    PUBLISH_FILTERED_IMAGE = False

    print("parsed args: ", args)
    print("unknown args: ", unknown_args)
    print("Node Initialized. Loading weights.")
    nav_interface = RlNavClass(args)
    print("Loaded weights. Lets gooooooooooo.......")
    rospy.spin()
