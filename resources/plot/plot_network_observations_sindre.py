import torch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

import scipy.fftpack as fft_pack

from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_euler_angles


if __name__ == "__main__":

    exp_name = "live_square_friday"
    config = "quad"

    # Load data
    data = pd.read_csv("./ulogs/" + exp_name + "/" + exp_name + "_neural_control_0.csv")
    pos_lin_vel_data = pd.read_csv("./ulogs/" + exp_name + "/" + exp_name + "_vehicle_local_position_0.csv")

    # Extract data and remove 0 values from rpm (publishing issue)
    pos_real = np.array([pos_lin_vel_data["x"].tolist(),pos_lin_vel_data["y"].tolist(),pos_lin_vel_data["z"].tolist()])
    pos = np.array([data["observation[0]"].tolist(),data["observation[1]"].tolist(),data["observation[2]"].tolist()])[:,1:]
    lin_vel = np.array([data["observation[9]"].tolist(),data["observation[10]"].tolist(),data["observation[11]"].tolist()])[:,1:]
    ori_6d = np.array([data["observation[3]"].tolist(),data["observation[4]"].tolist(),data["observation[5]"].tolist(),
                       data["observation[6]"].tolist(),data["observation[7]"].tolist(),data["observation[8]"].tolist()])[:,1:]
    ang_vel = np.array([data["observation[12]"].tolist(),data["observation[13]"].tolist(),data["observation[14]"].tolist()])[:,1:]

    if config == "hex":
        m_thrust = np.array([data["motor_thrust[0]"].tolist(),data["motor_thrust[1]"].tolist(),
                            data["motor_thrust[2]"].tolist(),data["motor_thrust[3]"].tolist(),
                            data["motor_thrust[4]"].tolist(),data["motor_thrust[5]"].tolist()])[:,1:]
    elif config == "quad":
        m_thrust = np.array([data["motor_thrust[0]"].tolist(),data["motor_thrust[1]"].tolist(),
                            data["motor_thrust[2]"].tolist(),data["motor_thrust[3]"].tolist()])[:,1:]

    inference_time = np.array(data["inference_time"].tolist())[1:]
    controller_time = np.array(data["controller_time"].tolist())[1:]
    mean_inference_time = np.mean(inference_time)
    mean_controller_time = np.mean(controller_time)
    print("Mean inference time: ", mean_inference_time)
    print("Mean controller time: ", mean_controller_time)

    # Transform data
    transformation_1 = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    transformation_2 = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    pos_enu_real = transformation_1 @ transformation_2 @ pos_real

    ori_matrix = rotation_6d_to_matrix(torch.tensor(ori_6d).T)
    ori_euler = matrix_to_euler_angles(ori_matrix, "XYZ").detach().numpy()

    # Get time points
    uorb_timestep_size = 1/1e6 # one micro second
    start_time = min([data["timestamp"].iloc[0],pos_lin_vel_data["timestamp"].iloc[0]])

    time_points = np.array((data["timestamp"].iloc[1:] - start_time))*uorb_timestep_size
    time_points_pos_vel = np.array((pos_lin_vel_data["timestamp"].iloc[:] - start_time))*uorb_timestep_size

    # Interpolate data
    f_pos_real = interp1d(time_points_pos_vel, pos_enu_real, axis=1)
    f_pos = interp1d(time_points, pos)
    f_lin_vel = interp1d(time_points, lin_vel)
    f_ori = interp1d(time_points, ori_euler.T)
    f_ang_vel = interp1d(time_points, ang_vel, axis=1)
    f_m_thrust = interp1d(time_points, m_thrust)


    sim_dt = 0.01
    T_start = max([time_points[0], time_points_pos_vel[0]])
    T_end = min([time_points[-110], time_points_pos_vel[-1]])
    #T_start = 22#time_points[0]
    #T_end = 29#time_points[-1]
    n_points = int((T_end - T_start)/sim_dt)


    time_points = np.linspace(T_start, T_end, n_points)

    pos_enu_real_interp = f_pos_real(time_points)
    pos_interp = f_pos(time_points)
    lin_vel_interp = f_lin_vel(time_points)
    ori_interp = f_ori(time_points)
    ang_vel_interp = f_ang_vel(time_points)
    m_thrust_interp = f_m_thrust(time_points)

    k_t = 0.000015

    setpoint_positions = np.array([[0,0,1.],[-.7,.7,1.],[-.7,-.7,1.],[.7,-.7,1.],[.7,.7,1.],[0,0,1.]])

    plt.rcParams['text.usetex'] = False
    plt.rcParams['figure.figsize'] = (10, 10)  # Width, height in inches

    # Create a time array that starts at 0 for plotting purposes
    plotting_time = time_points - T_start + 0.8

    # fig, axs = plt.subplots(3, 2)
    # axs[0,0].plot(plotting_time, pos_interp.T)
    # axs[0,0].legend([r"$p_x$", r"$p_y$", r"$p_z$"])
    # axs[0,0].set_ylabel(r"Position Error [$m$]")

    # axs[0,1].plot(plotting_time, lin_vel_interp.T)
    # axs[0,1].legend([r"$v_x$", r"$v_y$", r"$v_z$"])
    # axs[0,1].set_ylabel(r"Velocity [$\frac{m}{s}$]")

    # axs[1,0].plot(plotting_time, ori_interp.T)
    # axs[1,0].legend([r"$\phi$", r"$\theta$", r"$\psi$"])
    # axs[1,0].set_ylabel(r"Euler Angles [$rad$]")

    # axs[1,1].plot(plotting_time, ang_vel_interp.T)
    # axs[1,1].set_ylabel(r"Angular Velocity [$\frac{rad}{s}$]")
    # axs[1,1].legend([r"$\omega_x$", r"$ω_y$", r"$ω_z$"])

    # axs[2,0].plot(plotting_time, m_thrust_interp.T)
    # axs[2,0].set_xlabel(r"Time [$s$]")
    # axs[2,0].set_ylabel(r"Motor Thrust [% of max]")
    # if config == "hex":
    #     axs[2,0].legend([r"$u_1$", r"$u_2$", r"$u_3$", r"$u_4$", r"$u_5$", r"$u_6$"])
    # elif config == "quad":
    #     axs[2,0].legend([r"$u_1$", r"$u_2$", r"$u_3$", r"$u_4$"])

    fig, axs = plt.subplots(3)
    axs[0].plot(plotting_time, pos_interp.T)
    axs[0].legend([r"$p_x$", r"$p_y$", r"$p_z$"])
    axs[0].set_ylabel(r"Position Error [$m$]")
    #axs[0].xlim(0, 30)

    axs[1].plot(plotting_time, lin_vel_interp.T)
    axs[1].legend([r"$v_x$", r"$v_y$", r"$v_z$"])
    axs[1].set_ylabel(r"Velocity [$\frac{m}{s}$]")
    #axs[2].xlim(0, 30)

    axs[2].plot(plotting_time, m_thrust_interp.T)
    axs[2].set_xlabel(r"Time [$s$]")
    axs[2].set_ylabel(r"Motor Thrust [% of max]")
    axs[2].legend([r"$u_1$", r"$u_2$", r"$u_3$", r"$u_4$"])
    #axs[2].xlim(0, 30)


    fig.suptitle("Neural control in live flight", fontsize=16)


    plt.savefig(exp_name + "_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

    # from mpl_toolkits.mplot3d import Axes3D

    # # Get the raw position data without transformations
    # pos_x = pos_lin_vel_data["x"].to_numpy()
    # pos_y = pos_lin_vel_data["y"].to_numpy()
    # pos_z = pos_lin_vel_data["z"].to_numpy()

    # # Interpolate the raw position data
    # f_pos_x = interp1d(time_points_pos_vel, pos_x)
    # f_pos_y = interp1d(time_points_pos_vel, pos_y)
    # f_pos_z = interp1d(time_points_pos_vel, -pos_z)

    # # Get interpolated values at the common timepoints
    # pos_x_interp = f_pos_x(time_points)
    # pos_y_interp = f_pos_y(time_points)
    # pos_z_interp = f_pos_z(time_points)

    # # Create the 3D plot
    # fig3d = plt.figure(figsize=(10, 10))
    # ax3d = fig3d.add_subplot(111, projection='3d')

    # # Plot waypoints (use same coordinates as your setpoints)
    # ax3d.scatter(setpoint_positions[:, 0], setpoint_positions[:, 1], setpoint_positions[:, 2],
    #              c='r', marker='o', s=100, label='Waypoints')
    # ax3d.plot(setpoint_positions[:, 0], setpoint_positions[:, 1], setpoint_positions[:, 2],
    #           'r--', alpha=0.5, linewidth=1)

    # # Plot drone trajectory using direct position data
    # ax3d.plot(pos_x_interp, pos_y_interp, pos_z_interp,
    #           'b-', linewidth=2, label='Drone Path')

    # # Set labels and adjust view
    # ax3d.set_xlabel('X Position [m]')
    # ax3d.set_ylabel('Y Position [m]')
    # ax3d.set_zlabel('Z Position [m]')
    # ax3d.set_title('3D Flight Trajectory')
    # ax3d.set_zlim(0.5, 1.5)

    # # Set equal aspect ratio
    # ax3d.set_box_aspect([1, 1, 1])
    # ax3d.grid(True)
    # ax3d.legend()

    # # Adjust view angle
    # ax3d.view_init(elev=30, azim=-45)

    # # Save the figure
    # plt.savefig(exp_name + "_3D_trajectory.png", dpi=300, bbox_inches='tight', pad_inches=0.5)
    # plt.show()