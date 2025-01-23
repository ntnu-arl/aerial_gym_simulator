from tqdm import tqdm as tqdm
from aerial_gym import AERIAL_GYM_DIRECTORY
import numpy as np
from sample_factory.utils.typing import ActionSpace
from sample_factory.algo.utils.action_distributions import ContinuousActionDistribution
import torch
import torch.nn as nn
import os

class ModelDeploy(nn.Module):
    def __init__(self, layer_sizes, lims):
        super(ModelDeploy, self).__init__()
        self.control_stack = nn.ModuleList([])
        self.allocation_stack = nn.ModuleList([])
        
        self.max_u = lims["max_u"]
        self.min_u = lims["min_u"]        
        
        # control layers
        self.control_stack.append(nn.Linear(layer_sizes[0], layer_sizes[1]))
        if len(layer_sizes) > 2:
            self.control_stack.append(nn.Tanh())
            for i, input_size in enumerate(layer_sizes[1:-1]):
                output_size = layer_sizes[i+2]
                self.control_stack.append(
                    nn.Linear(input_size, output_size).to(torch.float)).cpu()
                self.control_stack.append(nn.Tanh())
    
    def rescale_actions(self, scaled_command_actions):
        command_actions = scaled_command_actions.clone()
        command_actions = scaled_command_actions * (self.max_u - self.min_u)/2 + (self.max_u + self.min_u)/2
    
        return command_actions


    def forward(self, x):
        for l_or_a in self.control_stack:
            x = l_or_a(x)
        
        return x

def convert_model_to_script_model(nn_model_full, max_u, min_u, n_motors):
    
    max_u = torch.tensor(max_u).to(torch.float).cpu()
    min_u = torch.tensor(min_u).to(torch.float).cpu()

    lims = { "max_u": max_u, "min_u": min_u}

    nn_model_deploy = ModelDeploy([15, 32, 24, n_motors], lims)

    nn_model_deploy.control_stack[0].weight.data[:] = nn_model_full.actor_critic.actor_encoder.encoders.observations.mlp_head[0].weight.data
    nn_model_deploy.control_stack[0].bias.data[:] = nn_model_full.actor_critic.actor_encoder.encoders.observations.mlp_head[0].bias.data
    nn_model_deploy.control_stack[2].weight.data[:] = nn_model_full.actor_critic.actor_encoder.encoders.observations.mlp_head[2].weight.data
    nn_model_deploy.control_stack[2].bias.data[:] = nn_model_full.actor_critic.actor_encoder.encoders.observations.mlp_head[2].bias.data
    nn_model_deploy.control_stack[4].weight.data[:] = nn_model_full.actor_critic.action_parameterization.distribution_linear.weight.data
    nn_model_deploy.control_stack[4].bias.data[:] = nn_model_full.actor_critic.action_parameterization.distribution_linear.bias.data
    
    sm = torch.jit.script(nn_model_deploy)
    torch.jit.save(sm, "./deployment/deployed_models/tinyprop.pt")

    print('Size normal (B):', os.path.getsize("./deployment/deployed_models/tinyprop.pt"))
    
    return sm