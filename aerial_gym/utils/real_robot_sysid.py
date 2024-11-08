#!/usr/bin/env python3

import rospy
from mavros_msgs.msg import PositionTarget


class PositionTargetCommandNode:
    def __init__(self):
        rospy.init_node("position_target_command_node")

        # Publisher
        self.pos_target_pub = rospy.Publisher(
            "/mavros/setpoint_raw/local", PositionTarget, queue_size=10
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
        self.pos_target_pub.publish(msg)

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz

        while not rospy.is_shutdown():
            # Example: Send a combined velocity and acceleration command
            # Move forward at 1 m/s with forward acceleration of 0.5 m/s^2
            self.send_position_target_command(0.0, 0.0, 0.0, 0.0, mode="velocity")
            rate.sleep()


if __name__ == "__main__":
    try:
        node = PositionTargetCommandNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
