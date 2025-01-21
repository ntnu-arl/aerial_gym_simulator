import torch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_euler_angles


if __name__ == "__main__":
    
    exp_name = "ete_seed_56_2025-1-15-14-06"
    
    # Load data
    data = pd.read_csv("./ulogs/" + exp_name + "/" + exp_name + "_neural_control_0.csv")
    esc_rpm_data = pd.read_csv("./ulogs/" + exp_name + "/" + exp_name + "_esc_status_0.csv")
    pos_lin_vel_data = pd.read_csv("./ulogs/" + exp_name + "/" + exp_name + "_vehicle_local_position_0.csv")
    
    # Extract data and remove 0 values from rpm (publishing issue)
    pos_real = np.array([pos_lin_vel_data["x"].tolist(),pos_lin_vel_data["y"].tolist(),pos_lin_vel_data["z"].tolist()])
    pos = np.array([data["observation[0]"].tolist(),data["observation[1]"].tolist(),data["observation[2]"].tolist()])[:,1:]
    lin_vel = np.array([data["observation[9]"].tolist(),data["observation[10]"].tolist(),data["observation[11]"].tolist()])[:,1:]
    ori_6d = np.array([data["observation[3]"].tolist(),data["observation[4]"].tolist(),data["observation[5]"].tolist(),
                       data["observation[6]"].tolist(),data["observation[7]"].tolist(),data["observation[8]"].tolist()])[:,1:]
    ang_vel = np.array([data["observation[12]"].tolist(),data["observation[13]"].tolist(),data["observation[14]"].tolist()])[:,1:]
    m_thrust = np.array([data["motor_thrust[0]"].tolist(),data["motor_thrust[1]"].tolist(),
                         data["motor_thrust[2]"].tolist(),data["motor_thrust[3]"].tolist()])[:,1:]
    
    rpm0 = np.array(esc_rpm_data["esc[0].esc_rpm"].tolist()) 
    rpm1 = np.array(esc_rpm_data["esc[1].esc_rpm"].tolist()) 
    rpm2 = np.array(esc_rpm_data["esc[2].esc_rpm"].tolist()) 
    rpm3 = np.array(esc_rpm_data["esc[3].esc_rpm"].tolist()) 
    
    # Transform data
    transformation_1 = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    transformation_2 = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    
    pos_enu_real = transformation_1 @ transformation_2 @ pos_real
    
    ori_matrix = rotation_6d_to_matrix(torch.tensor(ori_6d).T)
    ori_euler = matrix_to_euler_angles(ori_matrix, "XYZ").detach().numpy()
    
    # Get time points
    uorb_timestep_size = 1/1e6 # one micro second
    
    start_time = min([data["timestamp"].iloc[0],pos_lin_vel_data["timestamp"].iloc[0],esc_rpm_data["timestamp"].iloc[0]])
    
    time_points = np.array((data["timestamp"].iloc[1:] - start_time))*uorb_timestep_size
    time_points_pos_vel = np.array((pos_lin_vel_data["timestamp"].iloc[:] - start_time))*uorb_timestep_size
    time_points_rpm = np.array((esc_rpm_data["timestamp"].iloc[:] - start_time))*uorb_timestep_size
    
    fig = plt.figure()
    plt.plot(time_points_rpm[1:] - time_points_rpm[:-1])
    
    # Interpolate data
    f_pos_real = interp1d(time_points_pos_vel, pos_enu_real, axis=1)
    f_pos = interp1d(time_points, pos)
    f_lin_vel = interp1d(time_points, lin_vel)
    f_ori = interp1d(time_points, ori_euler.T)
    f_ang_vel = interp1d(time_points, ang_vel, axis=1)
    f_m_thrust = interp1d(time_points, m_thrust)
    f_rpm0 = interp1d(time_points_rpm, rpm0)
    f_rpm1 = interp1d(time_points_rpm, rpm1)
    f_rpm2 = interp1d(time_points_rpm, rpm2)
    f_rpm3 = interp1d(time_points_rpm, rpm3)
    
    sim_dt = 0.01
    T_start = max([time_points_rpm[0], time_points[0], time_points_pos_vel[0]])
    T_end = min([time_points_rpm[-1], time_points[-1], time_points_pos_vel[-1]])
    n_points = int((T_end - T_start)/sim_dt)
    
    time_points = np.linspace(T_start, T_end, n_points)
    
    pos_enu_real_interp = f_pos_real(time_points)
    pos_interp = f_pos(time_points)
    lin_vel_interp = f_lin_vel(time_points)
    ori_interp = f_ori(time_points)
    ang_vel_interp = f_ang_vel(time_points)
    m_thrust_interp = f_m_thrust(time_points)
    
    rpm0_interp = f_rpm0(time_points)
    rpm1_interp = f_rpm1(time_points)
    rpm2_interp = f_rpm2(time_points)
    rpm3_interp = f_rpm3(time_points)
    
    k_t = 0.00001286412
    f_0_interp = k_t * (rpm0_interp/60)**2
    f_1_interp = k_t * (rpm1_interp/60)**2
    f_2_interp = k_t * (rpm2_interp/60)**2
    f_3_interp = k_t * (rpm3_interp/60)**2
    
    setpoint_positions = np.array([[0.,0.,1.],[-.7,-.7,1.],[.7,-.7,1.],[.7,.7,1.],[-.7,.7,1.],[-.7,-.7,1.],[0.,0.,1.]])
    
    fig, axs = plt.subplots(3, 2)
    fig.set_size_inches(8,20)
    axs[0,0].plot(time_points, pos_interp.T)
    axs[0,0].legend(["x", "y", "z"])
    axs[0,0].set_title("Position")
    axs[0,0].set_xlabel("Time")
    axs[0,0].set_ylabel("Position")
    
    axs[0,1].plot(time_points,lin_vel_interp.T)
    axs[0,1].legend(["x", "y", "z"])
    axs[0,1].set_title("Velocity")
    axs[0,1].set_xlabel("Time")
    axs[0,1].set_ylabel("Velocity")
    
    axs[1,0].plot(time_points,ori_interp.T)
    axs[1,0].legend(["x", "y", "z"])
    axs[1,0].set_title("Angles")
    axs[1,0].set_xlabel("Time")
    axs[1,0].set_ylabel("Angles")
    
    axs[1,1].plot(time_points,ang_vel_interp.T)
    axs[1,1].set_title("Angular Velocity")
    axs[1,1].set_xlabel("Time")
    axs[1,1].set_ylabel("Angular Velocity")
    axs[1,1].legend(["x", "y", "z"])
    
    axs[2,0].plot(time_points,m_thrust_interp.T)
    axs[2,0].set_title("Actions motor commands")
    axs[2,0].set_xlabel("Time")
    axs[2,0].set_ylabel("Action")
    axs[2,0].legend(["u1", "u2", "u3", "u4"])
    
    plt.savefig("errors.pdf")