"""
This is a custom sim config file. It is used to define the simulation parameters.
"""

from aerial_gym.config.sim_config.base_sim_config import BaseSimConfig


class CustomSimConfig(BaseSimConfig):
    class sim(BaseSimConfig.sim):
        dt = 0.001  # Custom Parameter
        gravity = [+1.0, 0.0, 0.0]  # [m/s^2] # Custom parameter

        class physx(BaseSimConfig.sim.physx):
            num_threads = 5  # Custom Parameter
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 10  # Custom parameter
            num_velocity_iterations = 15  # Custom Parameter
            contact_offset = 0.01  # [m] # Custom parameter
            rest_offset = 0.01  # [m] # Custom parameter
            bounce_threshold_velocity = 0.5  # [m/s] # Custom parameter
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**20  # Custom parameter
            default_buffer_size_multiplier = 5
            contact_collection = 0  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
