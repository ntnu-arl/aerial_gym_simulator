from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
import torch
from aerial_gym.utils.helpers import get_args

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

if __name__ == "__main__":
    
    # PX4 uses NED coordinate system, while the simulator uses ENU coordinate system
    # The transformation matrix is the following:
    transformation_1 = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    transformation_2 = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    
    exp_name = "hover_2"
    
    # Load data
    pos_lin_vel_data = pd.read_csv("./ulogs/" + exp_name + "/" + exp_name + "_vehicle_local_position_0.csv")
    att_data = pd.read_csv("./ulogs/" + exp_name + "/" + exp_name + "_vehicle_attitude_0.csv")
    ang_vel_data = pd.read_csv("./ulogs/" + exp_name + "/" + exp_name + "_vehicle_angular_velocity_0.csv")
    esc_rpm_data = pd.read_csv("./ulogs/" + exp_name + "/" + exp_name + "_esc_status_0.csv")
    
    # Extract data and remove 0 values from rpm (publishing issue)
    pos = np.array([pos_lin_vel_data["x"].tolist(),pos_lin_vel_data["y"].tolist(),pos_lin_vel_data["z"].tolist()])
    lin_vel = np.array([pos_lin_vel_data["vx"].tolist(),pos_lin_vel_data["vy"].tolist(),pos_lin_vel_data["vz"].tolist()])
    q_att = np.array([att_data["q[1]"].tolist(),att_data["q[2]"].tolist(),att_data["q[3]"].tolist(),att_data["q[0]"].tolist()])
    ang_vel = np.array([ang_vel_data["xyz[0]"].tolist(),ang_vel_data["xyz[1]"].tolist(),ang_vel_data["xyz[2]"].tolist()])
    rpm0 = np.array([x for x in esc_rpm_data["esc[2].esc_rpm"].tolist() if x != 0])
    rpm1 = np.array([x for x in esc_rpm_data["esc[7].esc_rpm"].tolist() if x != 0])
    rpm2 = np.array([x for x in esc_rpm_data["esc[3].esc_rpm"].tolist() if x != 0])
    rpm3 = np.array([x for x in esc_rpm_data["esc[6].esc_rpm"].tolist() if x != 0])
    
    # Transform data
    pos_enu = transformation_1 @ transformation_2 @ pos
    
    lin_vel_enu = transformation_1 @ transformation_2 @ lin_vel
    
    r = R.from_quat(q_att.T)
    r_enu = R.from_matrix(transformation_1 @ (transformation_2 @ r.as_matrix()) @ transformation_1.T)
    ang_enu = r_enu.as_euler("xyz", degrees=True)
    q_att_enu = r_enu.as_quat().T
    
    ang_vel_enu = transformation_1 @ ang_vel
    
    # Get time points
    uorb_timestep_size = 1/1e6 # one micro second
    
    time_points_pos_vel = np.array((pos_lin_vel_data["timestamp"].iloc[:] - pos_lin_vel_data["timestamp"].iloc[0]))*uorb_timestep_size
    time_points_att = np.array((att_data["timestamp"].iloc[:] - att_data["timestamp"].iloc[0]))*uorb_timestep_size
    time_points_ang_vel = np.array((ang_vel_data["timestamp"].iloc[:] - ang_vel_data["timestamp"].iloc[0]))*uorb_timestep_size
    time_points_rpm = np.array((esc_rpm_data["timestamp"].iloc[:] - esc_rpm_data["timestamp"].iloc[0]))*uorb_timestep_size
    time_points_rpm0 = np.array([time_points_rpm[i] for i in range(len(time_points_rpm)) if esc_rpm_data["esc[2].esc_rpm"].iloc[i] != 0])
    time_points_rpm1 = np.array([time_points_rpm[i] for i in range(len(time_points_rpm)) if esc_rpm_data["esc[7].esc_rpm"].iloc[i] != 0])
    time_points_rpm2 = np.array([time_points_rpm[i] for i in range(len(time_points_rpm)) if esc_rpm_data["esc[3].esc_rpm"].iloc[i] != 0])
    time_points_rpm3 = np.array([time_points_rpm[i] for i in range(len(time_points_rpm)) if esc_rpm_data["esc[6].esc_rpm"].iloc[i] != 0])
    
    # Interpolate data
    f_pos = interp1d(time_points_pos_vel, pos_enu, axis=1)
    f_lin_vel = interp1d(time_points_pos_vel, lin_vel_enu, axis=1)
    f_q_att = interp1d(time_points_att, q_att_enu, axis=1)
    f_ang_vel = interp1d(time_points_ang_vel, ang_vel_enu, axis=1)
    f_rpm0 = interp1d(time_points_rpm0, rpm0)
    f_rpm1 = interp1d(time_points_rpm1, rpm1)
    f_rpm2 = interp1d(time_points_rpm2, rpm2)
    f_rpm3 = interp1d(time_points_rpm3, rpm3)
    
    sim_dt = 0.004
    T_start = 10 #max([time_points_rpm0[0], time_points_rpm1[0], time_points_rpm2[0], time_points_rpm3[0]])
    T_end = 40 #min([time_points_rpm0[-1], time_points_rpm1[-1], time_points_rpm2[-1], time_points_rpm3[-1]])
    n_points = int((T_end - T_start)/sim_dt)
    
    time_points = np.linspace(T_start, T_end, n_points)
    
    pos_enu_interp = f_pos(time_points)
    lin_vel_enu_interp = f_lin_vel(time_points)
    q_att_enu_interp = f_q_att(time_points)
    ang_vel_enu_interp = f_ang_vel(time_points)
    rpm0_interp = f_rpm0(time_points)
    rpm1_interp = f_rpm1(time_points)
    rpm2_interp = f_rpm2(time_points)
    rpm3_interp = f_rpm3(time_points)
    
    # calculate forces from rpm
    k_t = 0.000016781
    f_0_interp = k_t * (rpm0_interp/60)**2
    f_1_interp = k_t * (rpm1_interp/60)**2
    f_2_interp = k_t * (rpm2_interp/60)**2
    f_3_interp = k_t * (rpm3_interp/60)**2
    
    # simulate system
    args = get_args()
    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="empty_env",
        robot_name="quadBug",
        controller_name="no_control",
        args=None,
        device="cuda:0",
        num_envs=1,
        headless=args.headless,
        use_warp=args.use_warp,
    )
    
    actions = torch.tensor([f_0_interp, f_1_interp, f_2_interp, f_3_interp], dtype=torch.float).to("cuda:0").unsqueeze(0)
    #actions[:,:,:] = (env_manager.robot_manager.robot.cfg.robot_asset.robot_model.frame_mass + sum(env_manager.robot_manager.robot.cfg.robot_asset.robot_model.motor_masses))*9.81/4
    
    env_manager.reset()
    
    # get initial state
    init_state = np.hstack([pos_enu_interp[:,0].squeeze(), 
                            q_att_enu_interp[:,0].squeeze(), 
                            lin_vel_enu_interp[:,0].squeeze(),
                            ang_vel_enu_interp[:,0].squeeze()])
    
    init_state = torch.tensor(init_state, dtype=torch.float).to("cuda:0").unsqueeze(0)
    #init_state = init_state*0
    #init_state[0,6] = 1.0
    print(init_state)
    
    #env_manager.robot_manager.robot.reset_idx([0], init_state)
    
    pos_sim = [init_state[0,:3].cpu().detach().numpy()]
    lin_vel_sim = [init_state[0,7:10].cpu().detach().numpy()]
    q_att_sim = [init_state[0,3:7].cpu().detach().numpy()]
    ang_vel_sim = [init_state[0,10:].cpu().detach().numpy()]
    
    n_sub_points = 1000
    
    for i in range(n_sub_points-1):#n_points-1):
        
        env_manager.step(actions=actions[:,:,i])
        state_obtained = env_manager.robot_manager.robot.robot_state
        pos_sim.append(state_obtained[0,:3].cpu().detach().numpy())
        lin_vel_sim.append(state_obtained[0,7:10].cpu().detach().numpy())
        q_att_sim.append(state_obtained[0,3:7].cpu().detach().numpy())
        ang_vel_sim.append(state_obtained[0,10:].cpu().detach().numpy())
    
    # Plot results
    if True:
        fig, axs = plt.subplots(3, 2)
        axs[0,0].plot(time_points[:n_sub_points], (pos_enu_interp.T)[:n_sub_points])
        axs[0,0].plot(time_points[:n_sub_points], pos_sim)
        #axs[0,0].plot(pos_sim)
        #axs[0,0].plot(time_points_pos_vel, pos_enu.T, ":")
        axs[0,0].legend(["x_r", "y_r", "z_r","x_s", "y_s", "z_s"])
        axs[0,0].set_title("Position")
        axs[0,0].set_xlabel("Time")
        axs[0,0].set_ylabel("Position")
        
        axs[0,1].plot(time_points[:n_sub_points],(lin_vel_enu_interp.T)[:n_sub_points])
        axs[0,1].plot(time_points[:n_sub_points],lin_vel_sim)
        #axs[0,1].plot(lin_vel_sim)
        #axs[0,1].plot(time_points_pos_vel, lin_vel_enu.T, ":")
        axs[0,1].legend(["x_r", "y_r", "z_r", "x_s", "y_s", "z_s"])
        axs[0,1].set_title("Velocity")
        axs[0,1].set_xlabel("Time")
        axs[0,1].set_ylabel("Velocity")
        
        axs[1,0].plot(time_points[:n_sub_points],(q_att_enu_interp.T)[:n_sub_points])
        axs[1,0].plot(time_points[:n_sub_points],q_att_sim)
        #axs[1,0].plot(q_att_sim)
        #axs[1,0].plot(time_points_att, q_att_enu.T, ":")
        axs[1,0].legend(["x_r", "y_r", "z_r", "w_r","x_s", "y_s", "z_s", "w_s"])
        axs[1,0].set_title("Quaternions")
        axs[1,0].set_xlabel("Time")
        axs[1,0].set_ylabel("Angles")
        
        axs[1,1].plot(time_points[:n_sub_points],(ang_vel_enu_interp.T)[:n_sub_points])
        axs[1,1].plot(time_points[:n_sub_points],ang_vel_sim)
        #axs[1,1].plot(ang_vel_sim)
        #axs[1,1].plot(time_points_ang_vel, ang_vel_enu.T, ":")
        axs[1,1].set_title("Angular Velocity")
        axs[1,1].set_xlabel("Time")
        axs[1,1].set_ylabel("Angular Velocity")
        axs[1,1].legend(["x_r", "y_r", "z_r","x_s", "y_s", "z_s"])
        
        #axs[2,0].plot(actions.squeeze().cpu().detach().numpy().T)
        axs[2,0].plot(time_points[:n_sub_points],f_0_interp[:n_sub_points])
        #axs[2,0].plot(time_points_rpm0,rpm0, ":")
        axs[2,0].plot(time_points[:n_sub_points],f_1_interp[:n_sub_points])
        #axs[2,0].plot(time_points_rpm1,rpm1, ":")
        axs[2,0].plot(time_points[:n_sub_points],f_2_interp[:n_sub_points])
        #axs[2,0].plot(time_points_rpm2,rpm2, ":")
        axs[2,0].plot(time_points[:n_sub_points],f_3_interp[:n_sub_points])
        #axs[2,0].plot(time_points_rpm3,rpm3, ":")
        axs[2,0].set_title("Actions motor commands")
        axs[2,0].set_xlabel("Time")
        axs[2,0].set_ylabel("Action (motor rpm)")
        axs[2,0].legend(["u1", "u2", "u3", "u4"])
        
        #axs[2,1].plot(time_points_att,ang_enu_interp)
        #axs[2,1].set_title("Euler Angles")
        #axs[2,1].set_xlabel("Time")
        #axs[2,1].set_ylabel("Angles")
        #axs[2,1].legend(["roll", "pitch", "yaw"])
        
        plt.show()
    