#!/usr/bin/env python3

import rosbag
import csv
import numpy as np
from sensor_msgs.msg import Imu


def csv_to_imu_msgs(csv_file):
    imu_msgs = []
    timestamp = 0.0

    with open(csv_file, "r") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            # Extract timestep and data
            timestep = float(row[0])

            ax = float(row[1])
            ay = float(row[2])
            az = float(row[3])
            gx = float(row[4])
            gy = float(row[5])
            gz = float(row[6])

            imu_msg = Imu()
            imu_msg.header.stamp.secs = int(timestep)
            imu_msg.header.stamp.nsecs = int((timestep % 1) * 1e9)
            imu_msg.header.frame_id = "imu_link"

            imu_msg.linear_acceleration.x = ax
            imu_msg.linear_acceleration.y = ay
            imu_msg.linear_acceleration.z = az
            imu_msg.angular_velocity.x = gx
            imu_msg.angular_velocity.y = gy
            imu_msg.angular_velocity.z = gz
            if timestep == int(timestep):
                print(timestep)

            imu_msgs.append((timestep, imu_msg))

    return imu_msgs

    return imu_msgs


def write_to_rosbag(imu_msgs, bag_file):
    with rosbag.Bag(bag_file, "w") as bag:
        for timestamp, msg in imu_msgs:
            bag.write("imu/data", msg, msg.header.stamp)


if __name__ == "__main__":
    csv_file = "imu_data_2.csv"  # Replace with your CSV file path
    bag_file = "output_imu_data_2.bag"  # Name of the output rosbag file

    print("Starting conversion process...")
    imu_msgs = csv_to_imu_msgs(csv_file)
    write_to_rosbag(imu_msgs, bag_file)

    print(f"Rosbag file '{bag_file}' has been created successfully.")
