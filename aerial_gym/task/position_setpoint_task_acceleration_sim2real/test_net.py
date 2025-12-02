import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict, deque
import threading
import time
import math

class AccelerationControlNetwork(nn.Module):
    """
    Neural network for acceleration-based quadrotor control.
    Based on the network architecture from the rl_games examples.
    """
    def __init__(self, input_dim=17, output_dim=4, hidden_dims=[256, 128, 64]):
        super(AccelerationControlNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ELU()
            ])
            prev_dim = hidden_dim
        
        # Output layer for acceleration commands (x, y, z, yaw_rate)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def load_network(self, path):
        """Load network weights from checkpoint file"""
        try:
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
            
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
            
            # Clean the state dict to match our network structure
            cleaned_state_dict = OrderedDict()
            layer_idx = 0
            
            for key, value in state_dict.items():
                # Skip non-actor network parameters
                if "value" in key or "sigma" in key or "critic" in key:
                    continue
                    
                # Map rl_games network keys to our network structure
                if "a2c_network.actor_mlp" in key:
                    new_key = key.replace("a2c_network.actor_mlp.", "network.")
                elif "a2c_network.mu" in key:
                    new_key = key.replace("a2c_network.mu", f"network.{len(self.network)-1}")
                elif "actor_mlp" in key:
                    new_key = key.replace("actor_mlp.", "network.")
                elif "mu." in key:
                    new_key = key.replace("mu.", f"network.{len(self.network)-1}.")
                else:
                    continue
                    
                cleaned_state_dict[new_key] = value
            
            self.load_state_dict(cleaned_state_dict, strict=False)
        except:
            print(f"Failed to load network from {path}")
            raise

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    # Example usage
    net = AccelerationControlNetwork()
    net.load_network("/home/mihir/workspaces/aerial_gym_simulator_ws/aerial_gym_simulator/aerial_gym/rl_training/rl_games/runs/gen_ppo_09-15-56-42/nn/gen_ppo.pth")
    
    # Dummy input: batch of 1, input_dim=17
    with torch.no_grad():
        start_time = time.time()
        for i in range(1000):
            dummy_input = torch.randn(1, 17)
            output = net(dummy_input)
        end_time = time.time()
        print(f"Average inference time over 1000 runs: {(end_time - start_time) / 1000 * 1000:.3f} ms")

    print("Network output:", output)