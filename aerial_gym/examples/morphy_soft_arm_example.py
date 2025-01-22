from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
import torch
from aerial_gym.utils.helpers import get_args

import matplotlib.pyplot as plt
import numpy as np

try:
    import scienceplots

    # set theme to a scientific theme for matplotlib
    plt.style.use(["science", "vibrant"])
except:
    # set plt theme to seaborn colorblind
    plt.style.use("seaborn-v0_8-colorblind")

import csv

angle_list = []


def read_csv(filename):
    with open(filename, "r") as file:
        reader = csv.reader(file)
        for t, theta in reader:
            try:
                if float(t) > 0.06 and float(theta) < 15.0:
                    angle_list.append([float(t), float(theta)])
            except:
                pass


filename = "./stored_data/joint_step.csv"

read_csv(filename)
time_stamp = np.array([x[0] for x in angle_list])
angle_rad = np.array([x[1] * torch.pi / 180.0 for x in angle_list])


def mass_spring_damper(y, t, k_p, k_v):
    theta, omega = y
    dydt = [omega, -k_v * omega - k_p * torch.sign(theta) * theta**2]
    return dydt


if __name__ == "__main__":
    args = get_args()

    logger.warning(
        "This example demonstrates the logging of the arm data of the Morphy robot in an empty environment."
    )

    env_manager = SimBuilder().build_env(
        sim_name="base_sim_2ms",
        env_name="empty_env_2ms",
        robot_name="morphy_fixed_base",
        controller_name="no_control",
        args=None,
        device="cuda:0",
        num_envs=2,
        headless=args.headless,
        use_warp=args.use_warp,
    )
    if env_manager.robot_manager.robot.cfg.robot_asset.fix_base_link == False:
        logger.error(
            "The base link is not fixed for this robot. The base link should be fixed in morphy_config.py for this example to work."
        )
        exit(1)

    actions = 0.0 * 0.3 * 9.81 * torch.ones((env_manager.num_envs, 4)).to("cuda:0")
    actions[:, [0, 1, 2]] = 0.0
    dof_pos = torch.zeros((env_manager.num_envs, 8)).to("cuda:0")
    env_manager.reset()
    arm1_pitch_list = []
    arm1_yaw_list = []

    arm2_pitch_list = []
    arm2_yaw_list = []

    arm3_pitch_list = []
    arm3_yaw_list = []

    arm4_pitch_list = []
    arm4_yaw_list = []
    popt = [5834.85432241, 229.23612708]

    logger.warning(
        "Please make sure the morphy_config.py has fix_base_link set to True. Also make thre the initial states of the arms are appropriately set as indicated in the conig file."
    )

    for i in range(100000):
        if i % (1500) == 0:
            logger.info(f"Step {i}, changing target setpoint.")
            actions[:, 0:3] = 0 * 1.0 * (torch.rand_like(actions[:, 0:3]) * 2 - 1)
            actions[:, 3] = 0 * torch.pi * (torch.rand_like(actions[:, 3]) * 2 - 1)
            env_manager.reset()
            env_manager.robot_manager.robot.set_dof_position_targets(dof_pos)
            env_manager.robot_manager.robot.set_dof_velocity_targets(dof_pos)
            if i == 0:
                continue
            x_labels = torch.arange(0, len(arm1_pitch_list), 1).cpu().numpy()

            x = [0.25, 0]
            N = 7500
            dt = 0.002
            t = []
            d = []
            v = []
            for i in range(N):
                t.append(i * dt)
                d.append(x[0])
                v.append(x[1])
                x_inp = torch.tensor(x).to("cuda:0")
                x_inp[0] -= 7.2 * torch.pi / 180.0
                xdot = mass_spring_damper(x_inp, 0, *popt)
                # print(xdot)
                for j in range(2):
                    x[j] += xdot[j].cpu().numpy() * dt

            fig, ax = plt.subplots(figsize=(6, 2.5))
            ax.plot(
                x_labels * 0.01,
                arm1_pitch_list,
                label="Simulated response",
                marker="o",
                markersize=4,
            )
            ax.plot(t, d, "--", label="Identified model", linewidth=2, color="black")
            ax.plot(time_stamp, angle_rad, label="Ground Truth Response")
            ax.set(xlabel="Time (s)", ylabel=r"$\theta_j$ (rad)")
            ax.legend()
            # save plot as a pdf
            # plt.savefig('morphy_response.pdf')
            plt.show()
            exit(0)
        env_manager.step(actions=actions)
        # get dof states and plot them
        dof_states = env_manager.global_tensor_dict["dof_state_tensor"]
        robot_0_dof_states = dof_states[0].cpu().numpy()
        # plot the states
        arm1_pitch_list.append(robot_0_dof_states[0, 0])
        arm1_yaw_list.append(robot_0_dof_states[1, 0])

        arm2_pitch_list.append(robot_0_dof_states[2, 0])
        arm2_yaw_list.append(robot_0_dof_states[3, 0])

        arm3_pitch_list.append(robot_0_dof_states[4, 0])
        arm3_yaw_list.append(robot_0_dof_states[5, 0])

        arm4_pitch_list.append(robot_0_dof_states[6, 0])
        arm4_yaw_list.append(robot_0_dof_states[7, 0])
