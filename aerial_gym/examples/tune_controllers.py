from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
import torch
import numpy as np
from aerial_gym.utils.helpers import get_args
from matplotlib import pyplot as plt
import argparse


def calculate_cost_function(measured_metrics, target_metrics, weights=None, labels=None):
    """
    Calculate cost function to compare measured vs target metrics.
    
    Args:
        measured_metrics: dict with 'time_constants', 'overshoots', 'settling_times'
        target_metrics: dict with same structure
        weights: dict with weights for each metric type
        labels: list of labels for each action
    
    Returns:
        total_cost: float
    """
    if weights is None:
        weights = {
            'time_constant': [1.0, 2.0, 2.0, 5.0],  # Heavy weight on Roll/Pitch and Very Heavy on Yaw
            'overshoot': [0.5, 1.0, 1.0, 1.0],
            'settling_time': [0.8, 1.5, 1.5, 2.0]
        }
    
    if labels is None:
        labels = ["Thrust", "Roll", "Pitch", "Yaw"]
        
    cost = 0.0
    per_axis_costs = []
    
    # Per-axis weighted errors
    breakdown_str = "  [Cost Breakdown]"
    
    for i in range(len(measured_metrics['time_constants'])):
        if i == 0: # Skip thrust
            per_axis_costs.append(0.0)
            continue
            
        # Error components
        tc_error = abs(measured_metrics['time_constants'][i] - target_metrics['time_constants'][i])
        overshoot_error = abs(measured_metrics['overshoots'][i] - target_metrics['overshoots'][i])
        settling_error = abs(measured_metrics['settling_times'][i] - target_metrics['settling_times'][i])
        
        # Weighted axis cost
        axis_cost = (weights['time_constant'][i] * tc_error + 
                     weights['overshoot'][i] * overshoot_error / 100.0 + 
                     weights['settling_time'][i] * settling_error)
        
        per_axis_costs.append(axis_cost)
        cost += axis_cost
        label = labels[i] if i < len(labels) else f"Axis {i}"
        breakdown_str += f" {label}:{axis_cost:.3f}"
    
    print(breakdown_str)
    
    return cost


def update_controller_gains(env_manager, controller_name, gains_dict):
    """
    Update controller gains in the simulation using set_controller_gains.
    
    Args:
        env_manager: Environment manager
        controller_name: Name of the controller
        gains_dict: Dictionary with 'K_pos', 'K_vel', 'K_rot', 'K_angvel' as torch tensors
    """
    # Access the controller through the robot manager
    if hasattr(env_manager.robot_manager, 'robot'):
        controller = env_manager.robot_manager.robot.controller
        
        # Use the new set_controller_gains method
        if hasattr(controller, 'set_controller_gains'):
            K_pos = gains_dict.get('K_pos', controller.K_pos_tensor_current)
            K_vel = gains_dict.get('K_vel', controller.K_linvel_tensor_current)
            K_rot = gains_dict.get('K_rot', controller.K_rot_tensor_current)
            K_angvel = gains_dict.get('K_angvel', controller.K_angvel_tensor_current)
            
            controller.set_controller_gains(K_pos, K_vel, K_rot, K_angvel)
            
            if 'K_pos' in gains_dict:
                print(f"  K_pos:    {K_pos.cpu().numpy()}")
            if 'K_vel' in gains_dict:
                print(f"  K_vel:    {K_vel.cpu().numpy()}")
            if CONTROL_MODE_NAME == "attitude" or 'K_rot' in gains_dict:
                print(f"  K_rot:    {K_rot.cpu().numpy()}")
            print(f"  K_angvel: {K_angvel.cpu().numpy()}")
        else:
            print("Warning: Controller does not have set_controller_gains method")
    else:
        print("Warning: Could not access robot controller")


def get_current_controller_gains(env_manager):
    """
    Get current controller gains from the simulation.
    
    Returns:
        gains_dict: Dictionary with current gains
    """
    if hasattr(env_manager.robot_manager, 'robot'):
        controller = env_manager.robot_manager.robot.controller
        
        if hasattr(controller, 'K_pos_tensor_current'):
            return {
                'K_pos': controller.K_pos_tensor_current.clone(),
                'K_vel': controller.K_linvel_tensor_current.clone(),
                'K_rot': controller.K_rot_tensor_current.clone(),
                'K_angvel': controller.K_angvel_tensor_current.clone(),
            }
    return None


def run_system_identification(env_manager, actions, observations, tensor_dict, 
                               CONTROL_MODE_NAME, DICT_MAP_ENTRY, ACTION_DIM,
                               observation_sequence, actions_sequence,
                               num_sim_steps, step_time_index, device, target_metrics,
                               sim_dt=0.01):
    """
    Run system identification for all actions.
    
    Returns:
        metrics: dict with measured system metrics
    """
    time_elapsed_np = torch.arange(0, num_sim_steps * sim_dt, sim_dt).cpu().numpy()
    SETTLING_THRESHOLD = 0.02
    
    time_constants = []
    rise_times = []
    settling_times = []
    overshoots = []
    steady_state_errors = []
    
    for action_index in range(1, ACTION_DIM):
        actions[:] = 0.0
        env_manager.reset()
        
        # Run simulation and collect observations
        for i in range(num_sim_steps):
            if i == step_time_index:
                actions[:, action_index] = 1.0
            
            env_manager.step(actions)
            tensor_dict = env_manager.get_obs()
            observations[:, 0:3] = tensor_dict[DICT_MAP_ENTRY][:, 0:3]
            observations[:, 3] = tensor_dict["robot_body_angvel"][:, 2]
            
            # Store observations
            observation_sequence[i] = observations.cpu()
            if CONTROL_MODE_NAME == "attitude" or CONTROL_MODE_NAME == "bodyrate":
                actions_sequence[i, :, 0:3] = actions[:, 1:4].cpu()
            else:
                actions_sequence[i] = actions.cpu()
        
        # Post-process to extract metrics
        observation_sequence_np = observation_sequence.clone().cpu().numpy()
        
        # Determine which observation index to analyze
        if action_index == 3:
            obs_index = 3  # Yaw rate
        elif CONTROL_MODE_NAME == "attitude" or CONTROL_MODE_NAME == "bodyrate":
            obs_index = action_index - 1 if action_index > 0 else 0
        else:
            obs_index = action_index
        
        response = observation_sequence_np[:, 0, obs_index]
        steady_state_start = int(0.9 * num_sim_steps)
        steady_state_value = np.mean(response[steady_state_start:])
        
        post_step_response = response[step_time_index:]
        post_step_time = time_elapsed_np[step_time_index:] - time_elapsed_np[step_time_index]
        
        # Time constant
        target_63 = 0.632 * steady_state_value
        idx_63 = np.where(post_step_response >= target_63)[0]
        time_constant = post_step_time[idx_63[0]] if len(idx_63) > 0 else 0.0
        
        # Rise time
        target_10 = 0.1 * steady_state_value
        target_90 = 0.9 * steady_state_value
        idx_10 = np.where(post_step_response >= target_10)[0]
        idx_90 = np.where(post_step_response >= target_90)[0]
        rise_time = (post_step_time[idx_90[0]] - post_step_time[idx_10[0]]) if (len(idx_10) > 0 and len(idx_90) > 0) else 0.0
        
        # Settling time
        settling_band_upper = steady_state_value * (1 + SETTLING_THRESHOLD)
        settling_band_lower = steady_state_value * (1 - SETTLING_THRESHOLD)
        settling_time = 0.0
        for idx in range(len(post_step_response) - 1, 0, -1):
            if post_step_response[idx] > settling_band_upper or post_step_response[idx] < settling_band_lower:
                settling_time = post_step_time[idx]
                break
        
        # Overshoot
        max_value = np.max(post_step_response)
        overshoot = ((max_value - steady_state_value) / steady_state_value) * 100 if steady_state_value > 0 else 0.0
        
        # Steady-state error
        expected_value = 1.0
        steady_state_error = ((expected_value - steady_state_value) / expected_value) * 100 if expected_value > 0 else 0.0
        
        time_constants.append(time_constant)
        rise_times.append(rise_time)
        settling_times.append(settling_time)
        overshoots.append(overshoot)
        steady_state_errors.append(steady_state_error)
    
    return {
        'time_constants': [target_metrics['time_constants'][0]] + time_constants,
        'rise_times': [0.0] + rise_times,
        'settling_times': [target_metrics['settling_times'][0]] + settling_times,
        'overshoots': [target_metrics['overshoots'][0]] + overshoots,
        'steady_state_errors': [0.0] + steady_state_errors
    }


def verify_controller_api(env_manager, actions, observations, tensor_dict, 
                         CONTROL_MODE_NAME, DICT_MAP_ENTRY, ACTION_DIM,
                         observation_sequence, actions_sequence,
                         num_sim_steps, step_time_index, device, target_metrics,
                         controller_name, initial_gains):
    """
    Verify that changing gains actually changes the system response.
    Sets extreme values and prints the measured response.
    """
    print("\n" + "#"*80)
    print("VERIFYING CONTROLLER API WITH EXTREME GAIN VALUES")
    print("#"*80)
    
    # Save original
    original_gains = {k: v.clone() for k, v in initial_gains.items()}
    
    results = {}
    
    # Cases to test
    test_cases = {
        "EXtremly Low Gains (0.01)": 0.01,
        "Extremely High Gains (10.0)": 10.0,
        "Original Gains": 1.0 # Multiplier of 1.0 but we'll use actual original
    }
    
    for case_name, multiplier in test_cases.items():
        print(f"\nTesting Case: {case_name}")
        
        test_gains = {k: v.clone() for k, v in original_gains.items()}
        if case_name != "Original Gains":
            if CONTROL_MODE_NAME == "attitude":
                test_gains['K_rot'][..., 0:2] = multiplier
                test_gains['K_angvel'][..., 2] = multiplier
            else: # bodyrate
                test_gains['K_angvel'][..., 0:3] = multiplier
        
        update_controller_gains(env_manager, controller_name, test_gains)
        env_manager.reset()
        
        metrics = run_system_identification(
            env_manager, actions, observations, tensor_dict,
            CONTROL_MODE_NAME, DICT_MAP_ENTRY, ACTION_DIM,
            observation_sequence, actions_sequence,
            num_sim_steps, step_time_index, device, target_metrics,
            sim_dt=SIM_DT
        )
        results[case_name] = metrics
        
        # Print summary for this case
        for i in range(1, ACTION_DIM):
            print(f"  {Y_AXIS_LABELS[i]}: Measured TC={metrics['time_constants'][i]:.4f}s")

    print("\nAPI VERIFICATION SUMMARY:")
    print(f"{'Case':<30} | {'Roll/Pitch TC (Avg)':<25} | {'Yaw Rate TC':<15}")
    print("-" * 75)
    for case_name, metrics in results.items():
        avg_rp_tc = (metrics['time_constants'][1] + metrics['time_constants'][2]) / 2.0
        yaw_tc = metrics['time_constants'][3]
        print(f"{case_name:<30} | {avg_rp_tc:<25.4f} | {yaw_tc:<15.4f}")
    
    # Verify spread
    low_tc = results["EXtremly Low Gains (0.01)"]['time_constants'][1]
    high_tc = results["Extremely High Gains (10.0)"]['time_constants'][1]
    
    if abs(low_tc - high_tc) > 0.05:
        print("\n✓ SUCCESS: Controller API is working! System response changes significantly with gains.")
    else:
        print("\n⚠ WARNING: Controller API might not be responding as expected. Response spread is small.")
        print(f"  Spread: {abs(low_tc - high_tc):.4f}s")
    
    # Restore original
    update_controller_gains(env_manager, controller_name, original_gains)
    print("\nRestored original gains for optimization.")
    print("#"*80 + "\n")


def run_binary_search_tuning(env_manager, actions, observations, tensor_dict, 
                            CONTROL_MODE_NAME, DICT_MAP_ENTRY, ACTION_DIM,
                            observation_sequence, actions_sequence,
                            num_sim_steps, step_time_index, device, target_metrics,
                            controller_name, initial_gains, sim_dt=0.01):
    """
    Search for the optimal gain using binary search logic for each axis.
    Assumes that increasing gain decreases time constant.
    """
    print("\n" + "="*80)
    print("STARTING BINARY SEARCH CONTROLLER TUNING")
    print("="*80 + "\n")
    
    best_gains = {k: v.clone() for k, v in initial_gains.items()}
    best_cost = calculate_cost_function(
        run_system_identification(
            env_manager, actions, observations, tensor_dict,
            CONTROL_MODE_NAME, DICT_MAP_ENTRY, ACTION_DIM,
            observation_sequence, actions_sequence,
            num_sim_steps, step_time_index, device, target_metrics,
            sim_dt=sim_dt
        ), target_metrics
    )
    
    # Tuning targets
    if CONTROL_MODE_NAME == "bodyrate":
        tuning_config = [
            {'gain_name': 'K_angvel', 'axis_idx': 0, 'action_idx': 1, 'label': 'Roll Rate'},
            {'gain_name': 'K_angvel', 'axis_idx': 1, 'action_idx': 2, 'label': 'Pitch Rate'},
            {'gain_name': 'K_angvel', 'axis_idx': 2, 'action_idx': 3, 'label': 'Yaw Rate'},
        ]
    else: # attitude
        tuning_config = [
            {'gain_name': 'K_rot', 'axis_idx': 0, 'action_idx': 1, 'label': 'Roll'},
            {'gain_name': 'K_rot', 'axis_idx': 1, 'action_idx': 2, 'label': 'Pitch'},
            {'gain_name': 'K_angvel', 'axis_idx': 2, 'action_idx': 3, 'label': 'Yaw Rate'},
        ]

    max_binary_iterations = 20
    termination_threshold = 0.005 # 5ms precision
    
    for config in tuning_config:
        print(f"\nTuning {config['label']} (Action Index {config['action_idx']})")
        print("-" * 40)
        
        target_tc = target_metrics['time_constants'][config['action_idx']]
        gain_name = config['gain_name']
        axis_idx = config['axis_idx']
        
        # Define search range (0.01 to 50.0)
        low_gain = 0.01
        high_gain = 50.0
        
        for iteration in range(max_binary_iterations):
            mid_gain = (low_gain + high_gain) / 2.0
            
            # Update only this component
            current_test_gains = {k: v.clone() for k, v in best_gains.items()}
            current_test_gains[gain_name][..., axis_idx] = mid_gain
            
            update_controller_gains(env_manager, controller_name, current_test_gains)
            env_manager.reset()
            
            # System ID for this specific action to save time
            actions[:] = 0.0
            env_manager.reset()
            for i in range(num_sim_steps):
                if i == step_time_index:
                    actions[:, config['action_idx']] = 1.0 # Step input
                env_manager.step(actions)
                tensor_dict = env_manager.get_obs()
                observations[:, 0:3] = tensor_dict[DICT_MAP_ENTRY][:, 0:3]
                observations[:, 3] = tensor_dict["robot_body_angvel"][:, 2]
                observation_sequence[i] = observations.clone().cpu()
            
            # Analyze response
            obs_index = config['action_idx'] if config['action_idx'] == 3 else config['action_idx'] - 1
            response = observation_sequence.clone().cpu().numpy()[:, 0, obs_index]
            steady_state_value = np.mean(response[int(0.9 * num_sim_steps):])
            
            post_step_response = response[step_time_index:]
            post_step_time = torch.arange(0, num_sim_steps * sim_dt, sim_dt).cpu().numpy()[step_time_index:] 
            post_step_time -= post_step_time[0]
            
            target_63 = 0.632 * steady_state_value
            idx_63 = np.where(post_step_response >= target_63)[0]
            measured_tc = post_step_time[idx_63[0]] if len(idx_63) > 0 else 999.0
            
            print(f"  Iteration {iteration+1}: Gain={mid_gain:.4f} -> TC={measured_tc:.4f}s (Target={target_tc:.4f}s)")
            
            # Binary search logic: if measured TC is too slow (higher than target), we need more gain
            if measured_tc > target_tc:
                low_gain = mid_gain
            else:
                high_gain = mid_gain
            
            # Update best gains so far
            best_gains[gain_name][..., axis_idx] = mid_gain
            
            if abs(measured_tc - target_tc) <= termination_threshold:
                print(f"  ✓ Target reached for {config['label']}!")
                break

    return best_gains

CONTROLLER_MODES = {
    "bodyrate": "magpie_rates_control",
    "attitude": "magpie_attitude_control",
}

DICT_MAP = {
    "bodyrate": "robot_body_angvel",
    "attitude": "robot_euler_angles",
}

# Action dimensions for each controller mode
ACTION_DIMS = {
    "bodyrate": 4,      # [thrust, roll_rate, pitch_rate, yaw_rate]
    "attitude": 4,      # [thrust, roll, pitch, yaw_rate]
}

# Axis labels for printing metrics
Y_AXIS_LABELS_MAP = {
    "bodyrate": {
        0: "Thrust",
        1: "Roll Rate",
        2: "Pitch Rate",
        3: "Yaw Rate",
    },
    "attitude": {
        0: "Thrust",
        1: "Roll",
        2: "Pitch",
        3: "Yaw Rate",
    },
}

# Target system metrics from real robot (set these based on your real system identification)
TARGET_METRICS = {
    "bodyrate": {
        "time_constants": [0.10, 0.023, 0.026, 0.389],  # [thrust, roll_rate, pitch_rate, yaw_rate]
        "overshoots": [0.0, 10.0, 10.0, 10.0],  # Percentage
        "settling_times": [0.3, 0.5, 0.5, 0.8],  # seconds
    },
    "attitude": {
        "time_constants": [0.10, 0.099, 0.095, 0.389],  # [thrust, roll, pitch, yaw_rate]
        "overshoots": [0.0, 10.0, 10.0, 5.0],
        "settling_times": [0.3, 0.7, 0.7, 0.8],
    },
}

if __name__ == "__main__":
    CONTROL_MODE_NAME = "attitude"
    DICT_MAP_ENTRY = DICT_MAP[CONTROL_MODE_NAME]
    CONTROLLER_NAME = CONTROLLER_MODES[CONTROL_MODE_NAME]
    ACTION_DIM = ACTION_DIMS[CONTROL_MODE_NAME]  # Get correct action dimension
    Y_AXIS_LABELS = Y_AXIS_LABELS_MAP[CONTROL_MODE_NAME]
    
    args = get_args()
    
    # Force device to be consistent
    device = "cpu"
    print(f"Using device: {device}")
    
    env_manager = SimBuilder().build_env(
        sim_name="base_sim_2ms",
        env_name="empty_env_2ms",
        robot_name="magpie",
        controller_name=CONTROLLER_NAME,
        args=None,
        device=device,  # Use the device variable
        num_envs=args.num_envs,
        headless=args.headless,
        use_warp=args.use_warp,
    )
    actions = torch.zeros((env_manager.num_envs, ACTION_DIM), device=device)  # Use device directly
    env_manager.reset()
    tensor_dict = env_manager.get_obs()
    
    # Debug: Print action space info
    print(f"Actions shape: {actions.shape}")
    print(f"Expected action dim: {ACTION_DIM}")
    print(f"Number of envs: {env_manager.num_envs}")
    
    observations = torch.zeros((env_manager.num_envs, 4), device=device)
    ACTION_MAGNITUDE = 0.5
    SIM_DURATION_IN_SECONDS = 2.0  # Increased for better settling
    SIM_DT = 0.002
    TIME_CONSTANT_MAGNITUDE = ACTION_MAGNITUDE * 0.632  # 63.2% of final value
    SETTLING_THRESHOLD = 0.02  # 2% settling band
    num_sim_steps = int(SIM_DURATION_IN_SECONDS / SIM_DT)
    observation_sequence = torch.zeros((num_sim_steps, env_manager.num_envs, 4)).to("cpu")
    actions_sequence = torch.zeros((num_sim_steps, env_manager.num_envs, 4)).to("cpu")
    time_elapsed_np = torch.arange(0, SIM_DURATION_IN_SECONDS, SIM_DT).cpu().numpy()
    print(f"\n\n\n\nPerforming System Identification for {CONTROL_MODE_NAME} control mode\n\n")
    
    target_metrics = TARGET_METRICS[CONTROL_MODE_NAME]

    # Storage for metrics
    time_constants = [target_metrics['time_constants'][0]]
    rise_times = [0.0]
    settling_times = [target_metrics['settling_times'][0]]
    overshoots = [target_metrics['overshoots'][0]]
    steady_state_errors = [0.0]
    
    for action_index in range(1, ACTION_DIM):  # Skip Thrust (Index 0)
        actions[:] = 0.0
        print(f"\n{'='*60}")
        print(f"Action Index: {action_index} ({Y_AXIS_LABELS[action_index]})")
        print(f"{'='*60}")
        
        # Metrics for this action
        time_constant = 0.0
        rise_time_10_90 = 0.0
        settling_time = 0.0
        overshoot = 0.0
        steady_state_value = 0.0
        
        step_time_index = num_sim_steps // 2
        
        for i in range(num_sim_steps):
            if i == step_time_index:
                actions[:, action_index] = ACTION_MAGNITUDE
            
            env_manager.step(actions)
            tensor_dict = env_manager.get_obs()
            observations[:, 0:3] = tensor_dict[DICT_MAP_ENTRY][:, 0:3]
            observations[:, 3] = tensor_dict["robot_body_angvel"][:, 2]
            
            observation_sequence[i] = observations
            if CONTROL_MODE_NAME == "attitude" or CONTROL_MODE_NAME == "bodyrate":
                actions_sequence[i, :, 0:3] = actions[:, 1:4] # Map Roll, Pitch, Yaw
            else:
                actions_sequence[i] = actions
        
        # Post-process to extract metrics
        observation_sequence_np = observation_sequence.clone().cpu().numpy()
        actions_sequence_np = actions_sequence.clone().cpu().numpy()
        
        if action_index == 3:
            obs_index = 3 # Yaw rate
        elif CONTROL_MODE_NAME == "attitude" or CONTROL_MODE_NAME == "bodyrate":
            obs_index = action_index - 1 if action_index > 0 else 0
        else:
            obs_index = action_index
        
        # Get response for first environment
        response = observation_sequence_np[:, 0, obs_index]
        
        # Calculate steady-state value (average of last 10% of data)
        steady_state_start = int(0.9 * num_sim_steps)
        steady_state_value = np.mean(response[steady_state_start:])
        
        # Calculate metrics only after step input
        post_step_response = response[step_time_index:]
        post_step_time = time_elapsed_np[step_time_index:] - time_elapsed_np[step_time_index]
        
        # 1. Time Constant (63.2% of steady-state)
        target_63 = 0.632 * steady_state_value
        idx_63 = np.where(post_step_response >= target_63)[0]
        if len(idx_63) > 0:
            time_constant = post_step_time[idx_63[0]]
        
        # 2. Rise Time (10% to 90% of steady-state)
        target_10 = 0.1 * steady_state_value
        target_90 = 0.9 * steady_state_value
        idx_10 = np.where(post_step_response >= target_10)[0]
        idx_90 = np.where(post_step_response >= target_90)[0]
        if len(idx_10) > 0 and len(idx_90) > 0:
            rise_time_10_90 = post_step_time[idx_90[0]] - post_step_time[idx_10[0]]
        
        # 3. Settling Time (within 2% of steady-state)
        settling_band_upper = steady_state_value * (1 + SETTLING_THRESHOLD)
        settling_band_lower = steady_state_value * (1 - SETTLING_THRESHOLD)
        for idx in range(len(post_step_response) - 1, 0, -1):
            if post_step_response[idx] > settling_band_upper or post_step_response[idx] < settling_band_lower:
                settling_time = post_step_time[idx]
                break
        
        # 4. Overshoot (%)
        max_value = np.max(post_step_response)
        if steady_state_value > 0:
            overshoot = ((max_value - steady_state_value) / steady_state_value) * 100
        
        # 5. Steady-state error (%)
        expected_value = ACTION_MAGNITUDE
        if expected_value > 0:
            steady_state_error = ((expected_value - steady_state_value) / expected_value) * 100
        else:
            steady_state_error = 0.0
        
        # Store metrics
        time_constants.append(time_constant)
        rise_times.append(rise_time_10_90)
        settling_times.append(settling_time)
        overshoots.append(overshoot)
        steady_state_errors.append(steady_state_error)
        
        # Print metrics
        print(f"\nPerformance Metrics:")
        print(f"  Time Constant (τ):        {time_constant:.4f} s")
        print(f"  Rise Time (10-90%):       {rise_time_10_90:.4f} s")
        print(f"  Settling Time (±2%):      {settling_time:.4f} s")
        print(f"  Overshoot:                {overshoot:.2f} %")
        print(f"  Steady-State Value:       {steady_state_value:.4f}")
        print(f"  Steady-State Error:       {steady_state_error:.2f} %")
        print(f"  Expected Value:           {expected_value:.4f}")
        
        # Skip Plotting as requested
        # fig, axs = plt.subplots(4, 1, figsize=(10, 12))
        # ... (rest of plot logic omitted)
        env_manager.reset()

    # Compare with target metrics
    print(f"\n{'='*80}")
    print("COMPARISON WITH TARGET METRICS (Real System)")
    print(f"{'='*80}\n")
    
    measured_metrics = {
        'time_constants': time_constants,
        'overshoots': overshoots,
        'settling_times': settling_times,
    }
    
    # target_metrics = TARGET_METRICS[CONTROL_MODE_NAME] # Already defined above
    
    for i, label in Y_AXIS_LABELS.items():
        if i < ACTION_DIM and i > 0: # Skip Thrust
            print(f"{label}:")
            print(f"  Time Constant:  Measured={time_constants[i]:.4f}s  Target={target_metrics['time_constants'][i]:.4f}s  Error={abs(time_constants[i]-target_metrics['time_constants'][i]):.4f}s")
            print(f"  Overshoot:      Measured={overshoots[i]:.2f}%   Target={target_metrics['overshoots'][i]:.2f}%   Error={abs(overshoots[i]-target_metrics['overshoots'][i]):.2f}%")
            print(f"  Settling Time:  Measured={settling_times[i]:.4f}s  Target={target_metrics['settling_times'][i]:.4f}s  Error={abs(settling_times[i]-target_metrics['settling_times'][i]):.4f}s")
            print()
    
    # Calculate total cost
    total_cost = calculate_cost_function(measured_metrics, target_metrics, labels=Y_AXIS_LABELS)
    print(f"Total Cost Function: {total_cost:.6f}")
    print(f"{'='*80}\n")
    
    # Ask user which tuning method to use
    print("\nChoose Tuning Method:")
    print("1. Gradient-Free Local Search (Considers overshoot/settling time)")
    print("2. Binary Search (Fastest way to hit Time Constant target)")
    method_choice = input("Enter Choice (1/2) or 'n' to skip: ").strip().lower()
    
    if method_choice in ['1', '2']:
        # Get current gains
        current_gains = get_current_controller_gains(env_manager)
        if current_gains is None:
            print("Error: Could not get current controller gains")
        else:
            # API Verification Step
            verify_controller_api(
                env_manager, actions, observations, tensor_dict,
                CONTROL_MODE_NAME, DICT_MAP_ENTRY, ACTION_DIM,
                observation_sequence, actions_sequence,
                num_sim_steps, step_time_index, device, target_metrics,
                CONTROLLER_NAME, current_gains
            )

            if method_choice == '2':
                best_gains = run_binary_search_tuning(
                    env_manager, actions, observations, tensor_dict,
                    CONTROL_MODE_NAME, DICT_MAP_ENTRY, ACTION_DIM,
                    observation_sequence, actions_sequence,
                    num_sim_steps, step_time_index, device, target_metrics,
                    CONTROLLER_NAME, current_gains, sim_dt=SIM_DT
                )
                # Recalculate cost for binary search result
                update_controller_gains(env_manager, CONTROLLER_NAME, best_gains)
                final_metrics = run_system_identification(
                    env_manager, actions, observations, tensor_dict,
                    CONTROL_MODE_NAME, DICT_MAP_ENTRY, ACTION_DIM,
                    observation_sequence, actions_sequence,
                    num_sim_steps, num_sim_steps // 2, device, target_metrics,
                    sim_dt=SIM_DT
                )
                best_cost = calculate_cost_function(final_metrics, target_metrics, labels=Y_AXIS_LABELS)
            else:
                # Gradient-free logic (existing)
                best_cost = total_cost
                best_gains = {k: v.clone() for k, v in current_gains.items()}
                
                max_iterations = 30
                learning_rate = 0.15
                
                for iteration in range(max_iterations):
                    print(f"\nIteration {iteration + 1}/{max_iterations}")
                    print("-" * 60)
                    
                    improved = False
                    
                    # Try adjusting each relevant gain parameter
                    # For bodyrate: tune K_angvel (affects roll/pitch/yaw rates)
                    # For attitude: tune K_rot (affects roll/pitch) and K_angvel (yaw rate)
                    if CONTROL_MODE_NAME == "bodyrate":
                        tuning_gains = ['K_angvel']
                    else:  # attitude
                        tuning_gains = ['K_rot', 'K_angvel']
                    
                    for gain_name in tuning_gains:
                        current_gain = best_gains[gain_name].clone()
                        
                        # Define which axes to tune for this gain in this mode
                        if CONTROL_MODE_NAME == "attitude":
                            if gain_name == "K_rot":
                                active_axes = [0, 1] # Roll, Pitch angles
                            else: # K_angvel
                                active_axes = [2]    # Yaw Rate
                        else: # bodyrate
                            active_axes = [0, 1, 2] # Roll, Pitch, Yaw rates
                            
                        # For vector gains, try adjusting active elements independently
                        if current_gain.dim() > 0 and current_gain.numel() > 1:
                            for idx in active_axes:
                                # Try increasing this component
                                test_gains = {k: v.clone() for k, v in best_gains.items()}
                                test_gains[gain_name][..., idx] = current_gain[..., idx] * (1 + learning_rate)
                                
                                # Update controller
                                update_controller_gains(env_manager, CONTROLLER_NAME, test_gains)
                                
                                test_metrics = run_system_identification(
                                    env_manager, actions, observations, tensor_dict,
                                    CONTROL_MODE_NAME, DICT_MAP_ENTRY, ACTION_DIM,
                                    observation_sequence, actions_sequence,
                                    num_sim_steps, num_sim_steps // 2, device, target_metrics,
                                    sim_dt=SIM_DT
                                )
                                
                                test_cost = calculate_cost_function(test_metrics, target_metrics, labels=Y_AXIS_LABELS)
                                axis_name = ['X', 'Y', 'Z'][idx]
                                print(f"  {gain_name}[{axis_name}] +{learning_rate*100:.1f}%: cost={test_cost:.6f}", end="")
                                
                                if test_cost < best_cost:
                                    best_cost = test_cost
                                    best_gains = {k: v.clone() for k, v in test_gains.items()}
                                    improved = True
                                    print(" ✓ IMPROVED")
                                    # Print which axis improved
                                    for i in range(1, ACTION_DIM):
                                        if abs(test_metrics['time_constants'][i] - target_metrics['time_constants'][i]) < 0.01:
                                            print(f"    {Y_AXIS_LABELS[i]} is now close to target!")
                                else:
                                    # Try decreasing
                                    test_gains[gain_name][..., idx] = current_gain[..., idx] * (1 - learning_rate)
                                    update_controller_gains(env_manager, CONTROLLER_NAME, test_gains)
                                    env_manager.reset()
                                    
                                    test_metrics = run_system_identification(
                                        env_manager, actions, observations, tensor_dict,
                                        CONTROL_MODE_NAME, DICT_MAP_ENTRY, ACTION_DIM,
                                        observation_sequence, actions_sequence,
                                        num_sim_steps, num_sim_steps // 2, device, target_metrics,
                                        sim_dt=SIM_DT
                                    )
                                    
                                    test_cost_decrease = calculate_cost_function(test_metrics, target_metrics, labels=Y_AXIS_LABELS)
                                    print(f" | -{learning_rate*100:.1f}%: cost={test_cost_decrease:.6f}", end="")
                                    
                                    if test_cost_decrease < best_cost:
                                        best_cost = test_cost_decrease
                                        best_gains = {k: v.clone() for k, v in test_gains.items()}
                                        improved = True
                                        print(" ✓ IMPROVED")
                                        for i in range(1, ACTION_DIM):
                                            if abs(test_metrics['time_constants'][i] - target_metrics['time_constants'][i]) < 0.01:
                                                print(f"    {Y_AXIS_LABELS[i]} is now close to target!")
                                    else:
                                        print()
                        else:
                            # Scalar gain - adjust as before
                            # Try increasing
                            test_gains = {k: v.clone() for k, v in best_gains.items()}
                            test_gains[gain_name] = current_gain * (1 + learning_rate)
                            
                            update_controller_gains(env_manager, CONTROLLER_NAME, test_gains)
                            env_manager.reset()
                            test_metrics = run_system_identification(
                                env_manager, actions, observations, tensor_dict,
                                CONTROL_MODE_NAME, DICT_MAP_ENTRY, ACTION_DIM,
                                observation_sequence, actions_sequence,
                                num_sim_steps, num_sim_steps // 2, device, target_metrics,
                                sim_dt=SIM_DT
                            )
                            
                            test_cost = calculate_cost_function(test_metrics, target_metrics, labels=Y_AXIS_LABELS)
                            print(f"  {gain_name} +{learning_rate*100:.1f}%: cost={test_cost:.6f}", end="")
                            
                            if test_cost < best_cost:
                                best_cost = test_cost
                                best_gains = {k: v.clone() for k, v in test_gains.items()}
                                improved = True
                                print(" ✓ IMPROVED")
                            else:
                                test_gains[gain_name] = current_gain * (1 - learning_rate)
                                update_controller_gains(env_manager, CONTROLLER_NAME, test_gains)
                                env_manager.reset()
                                test_metrics = run_system_identification(
                                    env_manager, actions, observations, tensor_dict,
                                    CONTROL_MODE_NAME, DICT_MAP_ENTRY, ACTION_DIM,
                                    observation_sequence, actions_sequence,
                                    num_sim_steps, num_sim_steps // 2, device, target_metrics,
                                    sim_dt=SIM_DT
                                )
                                
                                test_cost_decrease = calculate_cost_function(test_metrics, target_metrics, labels=Y_AXIS_LABELS)
                                print(f" | -{learning_rate*100:.1f}%: cost={test_cost_decrease:.6f}", end="")
                                
                                if test_cost_decrease < best_cost:
                                    best_cost = test_cost_decrease
                                    best_gains = {k: v.clone() for k, v in test_gains.items()}
                                    improved = True
                                    print(" ✓ IMPROVED")
                                else:
                                    print()
                    
                    print(f"\nBest cost after iteration {iteration + 1}: {best_cost:.6f}")
                    
                    # Show current performance for each axis
                    update_controller_gains(env_manager, CONTROLLER_NAME, best_gains)
                    env_manager.reset()
                    current_metrics = run_system_identification(
                        env_manager, actions, observations, tensor_dict,
                        CONTROL_MODE_NAME, DICT_MAP_ENTRY, ACTION_DIM,
                        observation_sequence, actions_sequence,
                        num_sim_steps, num_sim_steps // 2, device, target_metrics,
                        sim_dt=SIM_DT
                    )
                    print("Current performance:")
                    for i in range(1, ACTION_DIM):
                        error = abs(current_metrics['time_constants'][i] - target_metrics['time_constants'][i])
                        print(f"  {Y_AXIS_LABELS[i]}: TC={current_metrics['time_constants'][i]:.4f}s (Target={target_metrics['time_constants'][i]:.4f}s, Error={error:.4f}s)")
                    
                    # Early stopping if no improvement
                    if not improved:
                        learning_rate *= 0.7  # Reduce learning rate
                        print(f"No improvement. Reducing learning rate to {learning_rate:.3f}")
                        
                        if learning_rate < 0.02:
                            print("Learning rate too small. Stopping.")
                            break
            
            # Apply best gains
            print("\n" + "="*80)
            print("TUNING COMPLETE")
            print("="*80)
            print(f"\nFinal cost: {best_cost:.6f} (Initial: {total_cost:.6f})")
            print(f"Improvement: {((total_cost - best_cost) / total_cost * 100):.2f}%")
            print(f"\nOptimal gains:")
            if CONTROL_MODE_NAME == "attitude":
                print(f"  K_rot:    {best_gains['K_rot'].cpu().numpy()}")
            print(f"  K_angvel: {best_gains['K_angvel'].cpu().numpy()}")
            
            # Apply optimal gains and run final system ID
            update_controller_gains(env_manager, CONTROLLER_NAME, best_gains)
            env_manager.reset()
            
            print("\nRunning final system identification with optimal gains...")
            final_metrics = run_system_identification(
                env_manager, actions, observations, tensor_dict,
                CONTROL_MODE_NAME, DICT_MAP_ENTRY, ACTION_DIM,
                observation_sequence, actions_sequence,
                num_sim_steps, num_sim_steps // 2, device, target_metrics,
                sim_dt=SIM_DT
            )
            
            print("\nFinal Performance Comparison:")
            for i, label in Y_AXIS_LABELS.items():
                if i < ACTION_DIM and i > 0: # Skip Thrust
                    print(f"{label}:")
                    print(f"  Time Constant:  {final_metrics['time_constants'][i]:.4f}s  (Target: {target_metrics['time_constants'][i]:.4f}s)")
                    print(f"  Overshoot:      {final_metrics['overshoots'][i]:.2f}%   (Target: {target_metrics['overshoots'][i]:.2f}%)")
                    print(f"  Settling Time:  {final_metrics['settling_times'][i]:.4f}s  (Target: {target_metrics['settling_times'][i]:.4f}s)")
                    print()
            
            print("\n" + "#"*40)
            print("FINAL OPTIMIZED GAINS")
            print("#"*40)
            print(f"K_pos:    torch.tensor({best_gains['K_pos'].cpu().numpy().tolist()}, device=device)")
            print(f"K_vel:    torch.tensor({best_gains['K_vel'].cpu().numpy().tolist()}, device=device)")
            print(f"K_rot:    torch.tensor({best_gains['K_rot'].cpu().numpy().tolist()}, device=device)")
            print(f"K_angvel: torch.tensor({best_gains['K_angvel'].cpu().numpy().tolist()}, device=device)")
            print("#"*40 + "\n")

    # plt.show() # Disabled
# K_rot:    torch.tensor([[7.039843559265137, 9.4319429397583, 0.32499998807907104]], device=device)
# K_angvel: torch.tensor([[0.5470019578933716, 0.6446386575698853, 14.704326629638672]], device=device)
