import matplotlib.pyplot as plt
import numpy as np

from aerial_gym.rl_training.sample_factory.end_to_end_training.helper import parse_aerialgym_cfg

from aerial_gym.utils.logging import CustomLogger
from aerial_gym.registry.task_registry import task_registry

from aerial_gym.rl_training.sample_factory.end_to_end_training.enjoy_ros import NN_Inference_ROS
from aerial_gym.rl_training.sample_factory.end_to_end_training.deployment.convert_model import convert_model_to_script_model

from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_euler_angles

from aerial_gym.registry.robot_registry import robot_registry
from aerial_gym.registry.env_registry import env_config_registry

logger = CustomLogger(__name__)

import torch

from tqdm import tqdm as tqdm
    
def test_policy_script_export():
    
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    
    cfg = parse_aerialgym_cfg(evaluation=True)
    cfg.train_dir = "./train_dir"
    
    env_cfg = task_registry.get_task_config("position_setpoint_task_sim2real_end_to_end")
    env = task_registry.make_task("position_setpoint_task_sim2real_end_to_end", num_envs=1, headless=True)
    obs = env.reset()[0]
    
    robot_name = "tinyprop"
    controller_name = "no_control"
    _, robot_config = robot_registry.make_robot(
            robot_name, controller_name, env_config_registry.get_env_config(env_cfg.env_name), "cuda:0")
    
    n_motors = robot_config.control_allocator_config.num_motors
    min_u = robot_config.control_allocator_config.motor_model_config.min_thrust
    max_u = robot_config.control_allocator_config.motor_model_config.max_thrust
    
    pos_list = []
    pos_err_list = []
    vel_list = []
    ori_list = []
    ang_vel_list = []
    
    action_motor_command_list = []
    
    goals = torch.tensor([[0.,0.,1.],[-.7,-.7,1.],[.7,-.7,1.],[.7,.7,1.],[-.7,.7,1.],[-.7,-.7,1.],[0.,0.,1.]] ).to(torch.float).to("cuda:0")

    nn_model = NN_Inference_ROS(cfg, env_cfg)
    
    model_deploy = convert_model_to_script_model(nn_model, max_u, min_u, n_motors).cpu()
    
    dt = 0.01
    reset_time = 5
    n_steps = int(reset_time*7/dt)
    reset_counter = 1
    
    n_crashes = 0
    total_runs = 0
    for i in range(n_steps):
        
        obs["observations"][:,:3] = goals[reset_counter-1] + obs["observations"][:,:3]
        
        actions_motor_commands = model_deploy(obs["observations"].squeeze().cpu()).unsqueeze(0).to("cuda")
            
        obs, _, crashes, successes, _ = env.step(actions=actions_motor_commands)
        
        n_crashes += torch.sum(crashes)
        total_runs += torch.sum(crashes) + torch.sum(successes)
    
        pos_list.append((-obs["observations"].squeeze().cpu()[0:3].detach().numpy()).tolist())
        pos_err_list.append((goals[reset_counter-1].cpu() + obs["observations"].squeeze().cpu()[0:3]).detach().numpy().tolist())
        
        ori_6d = obs["observations"].squeeze().cpu()[3:9].detach()
        ori_matrix = rotation_6d_to_matrix(ori_6d)
        ori_euler = matrix_to_euler_angles(ori_matrix, "XYZ")
        ori_list.append(ori_euler.detach().numpy().tolist())

        vel_list.append(obs["observations"].squeeze().cpu()[9:12].detach().numpy().tolist())
        
        ang_vel_list.append(obs["observations"].squeeze().cpu()[12:15].detach().numpy().tolist())

        actions_real =  actions_motor_commands * (max_u - min_u)/2 + (max_u + min_u)/2
        
        action_motor_command_list.append(actions_real.detach().squeeze().cpu().numpy().tolist())
        
        if i*dt > reset_time*reset_counter:
            print("switching goal")
            reset_counter += 1
            if reset_counter == goals.shape[0]+1:
                break        
    
    print("crash rate: ", n_crashes/total_runs)
    
    plot_results(pos_list, pos_err_list, vel_list, ori_list, 
                 ang_vel_list, action_motor_command_list, goals.cpu().numpy().tolist(), dt)
    
    
    
def plot_results(pos_list, pos_err_list, vel_list, ori_list, 
                 ang_vel_list, action_motor_command_list, goals, dt):
    
    n_steps = len(pos_list)
    # Plot the data
    fig, axs = plt.subplots(3, 2)
    axs[0,0].plot(np.linspace(0,n_steps,n_steps)*dt, np.array(pos_err_list))
    axs[0,0].legend(["x", "y", "z"])
    axs[0,0].set_title("Position")
    axs[0,0].set_xlabel("Time")
    axs[0,0].set_ylabel("Position")
    
    axs[0,1].plot(np.linspace(0,n_steps,n_steps)*dt,np.array(vel_list))
    axs[0,1].legend(["x", "y", "z"])
    axs[0,1].set_title("Velocity")
    axs[0,1].set_xlabel("Time")
    axs[0,1].set_ylabel("Velocity")
    
    axs[1,0].plot(np.linspace(0,n_steps,n_steps)*dt,np.array(ori_list))
    axs[1,0].legend(["x", "y", "z"])
    axs[1,0].set_title("Angles")
    axs[1,0].set_xlabel("Time")
    axs[1,0].set_ylabel("Angles")
    
    axs[1,1].plot(np.linspace(0,n_steps,n_steps)*dt,np.array(ang_vel_list))
    axs[1,1].set_title("Angular Velocity")
    axs[1,1].set_xlabel("Time")
    axs[1,1].set_ylabel("Angular Velocity")
    axs[1,1].legend(["x", "y", "z"])
    
    axs[2,0].plot(np.linspace(0,n_steps,n_steps)*dt,np.array(action_motor_command_list))
    axs[2,0].set_title("Actions motor commands")
    axs[2,0].set_xlabel("Time")
    axs[2,0].set_ylabel("Action")
    axs[2,0].legend(["u1", "u2", "u3", "u4"])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(np.array(pos_list)[:,0], np.array(pos_list)[:,1], np.array(pos_list)[:,2])
    ax.scatter(np.array(goals)[:,0], np.array(goals)[:,1], np.array(goals)[:,2], c='r', marker='o')
    plt.show()
    
if __name__ == "__main__":
    
    test_policy_script_export()
