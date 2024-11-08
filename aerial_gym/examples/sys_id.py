from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
import torch
from aerial_gym.utils.helpers import get_args
from matplotlib import pyplot as plt

CONTROLLER_MODES = {
    "attitude": "lmf2_attitude_control",
    "velocity": "lmf2_velocity_control",
    "acceleration": "lmf2_acceleration_control",
}

DICT_MAP = {
    "attitude": "robot_euler_angles",
    "velocity": "robot_vehicle_linvel",
    "acceleration": "imu_measurement",
}

Y_AXIS_LABELS = {
    0: "X",
    1: "Y",
    2: "Z",
    3: "Yaw Rate",
}

if __name__ == "__main__":
    CONTROL_MODE_NAME = "velocity"
    DICT_MAP_ENTRY = DICT_MAP[CONTROL_MODE_NAME]
    CONTROLLER_NAME = CONTROLLER_MODES[CONTROL_MODE_NAME]
    args = get_args()
    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="empty_env",
        robot_name="lmf2",
        controller_name=CONTROLLER_NAME,
        args=None,
        device="cuda:0",
        num_envs=args.num_envs,
        headless=args.headless,
        use_warp=args.use_warp,
    )
    actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")
    env_manager.reset()
    tensor_dict = env_manager.get_obs()
    observations = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")
    ACTION_MAGNITUDE = 1.0
    SIM_DURATION_IN_SECONDS = 5.0
    SIM_DT = 0.01
    TIME_CONSTANT_MAGNITUDE = ACTION_MAGNITUDE * 0.63212
    num_sim_steps = int(SIM_DURATION_IN_SECONDS / SIM_DT)
    observation_sequence = torch.zeros((num_sim_steps, env_manager.num_envs, 4)).to("cuda:0")
    actions_sequence = torch.zeros((num_sim_steps, env_manager.num_envs, 4)).to("cuda:0")
    time_elapsed_np = torch.arange(0, SIM_DURATION_IN_SECONDS, SIM_DT).cpu().numpy()
    print(f"\n\n\n\nPerforming System Identification for {CONTROL_MODE_NAME} control mode\n\n")
    for action_index in range(4):
        actions[:] = 0.0
        print("Action Index: ", action_index)
        time_constant = 0.0
        for i in range(num_sim_steps):
            if i == num_sim_steps // 2:
                actions[:, action_index] = ACTION_MAGNITUDE
            env_manager.step(actions)
            observations[:, 0:3] = tensor_dict[DICT_MAP_ENTRY][:, 0:3]
            observations[:, 3] = tensor_dict["robot_angvel"][:, 2]
            if CONTROL_MODE_NAME == "attitude":
                if action_index > 0:
                    if (
                        observations[0, action_index - 1] > TIME_CONSTANT_MAGNITUDE
                        and time_constant == 0.0
                    ):
                        time_constant = (i - num_sim_steps // 2) * SIM_DT
            else:
                if observations[0, action_index] > TIME_CONSTANT_MAGNITUDE and time_constant == 0.0:
                    time_constant = (i - num_sim_steps // 2) * SIM_DT
            observation_sequence[i] = observations
            if CONTROL_MODE_NAME == "attitude":
                actions_sequence[i, :, 0:2] = actions[:, 1:3]
                actions_sequence[i, :, 3] = actions[:, 3]
            else:
                actions_sequence[i] = actions

        # plot the response of the system:
        observation_sequence_np = observation_sequence.clone().cpu().numpy()
        actions_sequence_np = actions_sequence.clone().cpu().numpy()
        fig, axs = plt.subplots(4, 1)
        print("Time Constant: ", time_constant)
        fig.suptitle(
            f"System ID for {CONTROL_MODE_NAME} with activation in action {action_index}",
            fontsize=16,
        )
        for i in range(4):
            axs[i].plot(time_elapsed_np, observation_sequence_np[:, 0, i])
            axs[i].plot(time_elapsed_np, actions_sequence_np[:, 0, i])
            axs[i].set_ylabel(Y_AXIS_LABELS[i])
            axs[i].set_xlabel("Time")
            axs[i].set_ylim(-2, 2)
        plt.show(block=False)
        env_manager.reset()

    plt.show()
