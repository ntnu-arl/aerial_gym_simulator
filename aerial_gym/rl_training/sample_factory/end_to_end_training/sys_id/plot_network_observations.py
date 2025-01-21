import torch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

import scipy.fftpack as fft_pack

from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_euler_angles, quaternion_to_matrix, matrix_to_rotation_6d, euler_angles_to_matrix


if __name__ == "__main__":
    
    #exp_name = "2024-11-13-15-31_domain_randomization_with_noise_square"
    #exp_name = "2024-11-26-10-15_small_noise_for_all_yaw_fix_working_comp_in_sim_corrected_scaling"
    #exp_name = "2024-11-29-10-16-21_updated_motor_model_and_activated_pid"
    #exp_name = "ete_seed_16_2025-1-15-13-21"
    #exp_name = "ete_seed_26_2025-1-15-13-27"
    #exp_name = "ete_seed_36_2025-1-15-13-51"
    #exp_name = "ete_seed_46_2025-1-15-14-00"
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
    wrenches = np.array([data["wrench[0]"].tolist(),data["wrench[1]"].tolist(),data["wrench[2]"].tolist(),
                            data["wrench[3]"].tolist(),data["wrench[4]"].tolist(),data["wrench[5]"].tolist()])[:,1:]
    
    rpm0 = np.array(esc_rpm_data["esc[0].esc_rpm"].tolist()) #np.array([x for x in esc_rpm_data["esc[0].esc_rpm"].tolist() if x != 0])
    rpm1 = np.array(esc_rpm_data["esc[1].esc_rpm"].tolist()) #np.array([x for x in esc_rpm_data["esc[3].esc_rpm"].tolist() if x != 0])
    rpm2 = np.array(esc_rpm_data["esc[2].esc_rpm"].tolist()) #np.array([x for x in esc_rpm_data["esc[1].esc_rpm"].tolist() if x != 0])
    rpm3 = np.array(esc_rpm_data["esc[3].esc_rpm"].tolist()) #np.array([x for x in esc_rpm_data["esc[2].esc_rpm"].tolist() if x != 0])
    
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
    
    #time_points_rpm0 = np.array([time_points[i] for i in range(len(time_points)) if esc_rpm_data["esc[0].esc_rpm"].iloc[i] != 0])
    #time_points_rpm2 = np.array([time_points[i] for i in range(len(time_points)) if esc_rpm_data["esc[1].esc_rpm"].iloc[i] != 0])
    #time_points_rpm3 = np.array([time_points[i] for i in range(len(time_points)) if esc_rpm_data["esc[2].esc_rpm"].iloc[i] != 0])
    #ime_points_rpm1 = np.array([time_points[i] for i in range(len(time_points)) if esc_rpm_data["esc[3].esc_rpm"].iloc[i] != 0])
    
    # Interpolate data
    f_pos_real = interp1d(time_points_pos_vel, pos_enu_real, axis=1)
    f_pos = interp1d(time_points, pos)
    f_lin_vel = interp1d(time_points, lin_vel)
    f_ori = interp1d(time_points, ori_euler.T)
    f_ang_vel = interp1d(time_points, ang_vel, axis=1)
    f_m_thrust = interp1d(time_points, m_thrust)
    f_wrenches = interp1d(time_points, wrenches)
    f_rpm0 = interp1d(time_points_rpm, rpm0)
    f_rpm1 = interp1d(time_points_rpm, rpm1)
    f_rpm2 = interp1d(time_points_rpm, rpm2)
    f_rpm3 = interp1d(time_points_rpm, rpm3)
    
    sim_dt = 0.01
    #T_start = max([time_points_rpm[0], time_points[0], time_points_pos_vel[0]])
    T_end = min([time_points_rpm[-1], time_points[-1], time_points_pos_vel[-1]])
    T_start = 22#time_points[0]
    #T_end = 29#time_points[-1]
    n_points = int((T_end - T_start)/sim_dt)
    
    time_points = np.linspace(T_start, T_end, n_points)
    
    pos_enu_real_interp = f_pos_real(time_points)
    pos_interp = f_pos(time_points)
    lin_vel_interp = f_lin_vel(time_points)
    ori_interp = f_ori(time_points)
    ang_vel_interp = f_ang_vel(time_points)
    m_thrust_interp = f_m_thrust(time_points)
    wrenches_interp = f_wrenches(time_points)
    
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
    
    axs[2,1].plot(time_points,wrenches_interp.T)
    axs[2,1].set_title("Actions wrench commands")
    axs[2,1].set_xlabel("Time")
    axs[2,1].set_ylabel("Action")
    axs[2,1].legend(["fx", "fy", "fz", "Tx", "Ty", "Tz"])
    
    plt.savefig("errors.pdf")
    
    yf_u1 = fft_pack.fft(m_thrust_interp[0])
    yf_u2 = fft_pack.fft(m_thrust_interp[1])
    yf_u3 = fft_pack.fft(m_thrust_interp[2])
    yf_u4 = fft_pack.fft(m_thrust_interp[3])

    xf = np.linspace(0.0, 1.0/(2.0*sim_dt), n_points//2)
    
    fig, ax = plt.subplots(2, 2)
    ax[0,0].plot(xf[1:], 2.0/n_points * np.abs(yf_u1[1:n_points//2]))
    ax[0,0].set_title("u1")
    ax[0,0].set_xlabel("Frequency")
    ax[0,0].set_ylabel("Amplitude")
    ax[0,1].plot(xf[1:], 2.0/n_points * np.abs(yf_u2[1:n_points//2]))
    ax[0,1].set_title("u2")
    ax[0,1].set_xlabel("Frequency")
    ax[0,1].set_ylabel("Amplitude")
    ax[1,0].plot(xf[1:], 2.0/n_points * np.abs(yf_u3[1:n_points//2]))
    ax[1,0].set_title("u3")
    ax[1,0].set_xlabel("Frequency")
    ax[1,0].set_ylabel("Amplitude")
    ax[1,1].plot(xf[1:], 2.0/n_points * np.abs(yf_u4[1:n_points//2]))
    ax[1,1].set_title("u4")
    ax[1,1].set_xlabel("Frequency")
    ax[1,1].set_ylabel("Amplitude")
    
    fig, ax = plt.subplots(2, 2)
    ax[0,0].plot(time_points, f_0_interp)
    ax[0,0].plot(time_points, m_thrust_interp[0])
    ax[0,0].set_title("u1")
    ax[0,0].legend(["thrust calculated from rpm feed back", "network command"])
    ax[0,0].set_xlabel("Time")
    ax[0,0].set_ylabel("Thrust")
    ax[0,1].plot(time_points, f_1_interp)
    ax[0,1].plot(time_points, m_thrust_interp[1])
    ax[0,1].set_title("u2")
    ax[0,1].legend(["thrust calculated from rpm feed back", "network command"])
    ax[0,1].set_xlabel("Time")
    ax[0,1].set_ylabel("Thrust")
    ax[1,0].plot(time_points, f_2_interp)
    ax[1,0].plot(time_points, m_thrust_interp[2])
    ax[1,0].set_title("u3")
    ax[1,0].legend(["thrust calculated from rpm feed back", "network command"])
    ax[1,0].set_xlabel("Time")
    ax[1,0].set_ylabel("Thrust")
    ax[1,1].plot(time_points, f_3_interp)
    ax[1,1].plot(time_points, m_thrust_interp[3])
    ax[1,1].set_title("u4")
    ax[1,1].legend(["thrust calculated from rpm feed back", "network command"])
    ax[1,1].set_xlabel("Time")
    ax[1,1].set_ylabel("Thrust")
    fig.suptitle("interpolated thrust setpoints and measured rpm")
    
    fig, ax = plt.subplots(2, 2)
    ax[0,0].plot(esc_rpm_data["timestamp"].iloc[:],k_t * (rpm0/60)**2)
    ax[0,0].plot(np.array(data["timestamp"].iloc[1:]),m_thrust[0])
    ax[0,0].set_title("u1")
    ax[0,0].set_xlabel("Time")
    ax[0,0].set_ylabel("Thrust")
    ax[0,0].legend(["thrust calculated from rpm feed back", "network command"])
    ax[0,1].plot(esc_rpm_data["timestamp"].iloc[:],k_t * (rpm1/60)**2)
    ax[0,1].plot(np.array(data["timestamp"].iloc[1:]),m_thrust[1])
    ax[0,1].set_title("u2")
    ax[0,1].set_xlabel("Time")
    ax[0,1].set_ylabel("Thrust")
    ax[0,1].legend(["thrust calculated from rpm feed back", "network command"])
    ax[1,0].plot(esc_rpm_data["timestamp"].iloc[:],k_t * (rpm2/60)**2)
    ax[1,0].plot(np.array(data["timestamp"].iloc[1:]),m_thrust[2])
    ax[1,0].set_title("u3")
    ax[1,0].set_xlabel("Time")
    ax[1,0].set_ylabel("Thrust")
    ax[1,0].legend(["thrust calculated from rpm feed back", "network command"])
    ax[1,1].plot(esc_rpm_data["timestamp"].iloc[:],k_t * (rpm3/60)**2)
    ax[1,1].plot(np.array(data["timestamp"].iloc[1:]),m_thrust[3])
    ax[1,1].set_title("u4")
    ax[1,1].set_xlabel("Time")
    ax[1,1].set_ylabel("Thrust")
    ax[1,1].legend(["thrust calculated from rpm feed back", "network command"])
    fig.suptitle("real thrust setpoints and measured rpm")
    
    plt.figure()
    plt.plot(esc_rpm_data["timestamp"].iloc[:],rpm0)#, 'r')
    plt.plot(esc_rpm_data["timestamp"].iloc[:],rpm1)#, 'g')
    plt.plot(esc_rpm_data["timestamp"].iloc[:],rpm2)#, 'b')
    plt.plot(esc_rpm_data["timestamp"].iloc[:],rpm3)#, 'y')
    # plt.plot(esc_rpm_data["timestamp"].iloc[:],k_t * (rpm0/60)**2)#, 'r')
    # plt.plot(esc_rpm_data["timestamp"].iloc[:],k_t * (rpm1/60)**2)#, 'g')
    # plt.plot(esc_rpm_data["timestamp"].iloc[:],k_t * (rpm2/60)**2)#, 'b')
    # plt.plot(esc_rpm_data["timestamp"].iloc[:],k_t * (rpm3/60)**2)#, 'y')
    # plt.plot(np.array(data["timestamp"].iloc[1:]),m_thrust[0], 'r:')
    # plt.plot(np.array(data["timestamp"].iloc[1:]),m_thrust[1], 'g:')
    # plt.plot(np.array(data["timestamp"].iloc[1:]),m_thrust[2], 'b:')
    # plt.plot(np.array(data["timestamp"].iloc[1:]),m_thrust[3], 'y:')
    # plt.legend(["r_m1", "r_m2", "r_m3", "r_m4", "n_m1", "n_m2", "n_m3", "n_m4"])
    plt.legend(["r_m1", "r_m2", "r_m3", "r_m4"])
    plt.title("real thrust setpoints and measured rpm")
    
    print("mean action interp values m1: ", np.mean(f_0_interp))
    print("mean action interp values m2: ", np.mean(f_1_interp))
    print("mean action interp values m3: ", np.mean(f_2_interp))
    print("mean action interp values m4: ", np.mean(f_3_interp))
    
    print("correlated action interp values m1: ", np.correlate(f_0_interp, m_thrust_interp[0])[0])
    print("correlated action interp values m2: ", np.correlate(f_1_interp, m_thrust_interp[1])[0])
    print("correlated action interp values m3: ", np.correlate(f_2_interp, m_thrust_interp[2])[0])
    print("correlated action interp values m4: ", np.correlate(f_3_interp, m_thrust_interp[3])[0])
    
    #plt.show()
# fig, ax = plt.subplots(2, 2)
# ax[0,0].plot(time_points_pos_vel, pos_enu_real[0])
# ax[0,0].plot(np.array(data["timestamp"].iloc[1:])*uorb_timestep_size, pos[0])
# ax[0,0].set_xlabel("Time")
# ax[0,1].set_ylabel("x")
# ax[0,0].legend(["real", "what network sees"])
# ax[1,0].plot(time_points_pos_vel, pos_enu_real[1])
# ax[1,0].plot(np.array(data["timestamp"].iloc[1:])*uorb_timestep_size, pos[1])
# ax[1,0].set_xlabel("Time")
# ax[0,1].set_ylabel("y")
# ax[1,0].legend(["real", "what network sees"])
# ax[1,1].plot(time_points_pos_vel, pos_enu_real[2])
# ax[1,1].plot(np.array(data["timestamp"].iloc[1:])*uorb_timestep_size, pos[2])
# ax[1,1].set_xlabel("Time")
# ax[1,1].set_ylabel("z")
# ax[1,1].legend(["real", "what network sees"])