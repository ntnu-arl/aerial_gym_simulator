import torch
import torch.nn as nn
import ai_edge_torch

class e2eNetwork(nn.Module):
    def __init__(self):
        # TODO make sure the layers and activation function match the model you have trained
        super(e2eNetwork, self).__init__()
        self.fc1 = nn.Linear(15, 64)  # Input layer (15 inputs, position error, velocity, attitude (6D), angular velocity)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4) # Output layer (number of motors)

    def forward(self, x):
        x = torch.relu(self.fc1(x)) # Make sure to add correct activation functions
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def convert_network():
    # Load the state dictionary. TODO: Change the model name to match the one you have trained
    state_dict = torch.load("gen_ppo.pth", map_location=torch.device('cpu'))
    # Extract the model state dictionary
    model_state_dict = state_dict["model"]

    # Map the keys to match the e2eNetwork structure
    mapped_state_dict = {
        "fc1.weight": model_state_dict["a2c_network.actor_mlp.0.weight"],
        "fc1.bias": model_state_dict["a2c_network.actor_mlp.0.bias"],
        "fc2.weight": model_state_dict["a2c_network.actor_mlp.2.weight"],
        "fc2.bias": model_state_dict["a2c_network.actor_mlp.2.bias"],
        "fc3.weight": model_state_dict["a2c_network.mu.weight"],
        "fc3.bias": model_state_dict["a2c_network.mu.bias"]
    }

    # Initialize the e2eNetwork model
    e2e_model = e2eNetwork()
    e2e_model.load_state_dict(mapped_state_dict)
    e2e_model.eval()

    # Test the model
    sample_input = torch.rand(1, 15)
    pytorch_output = e2e_model(sample_input)

    # Convert to TFLite
    tfLite_model = ai_edge_torch.convert(e2e_model, (sample_input,))
    tfLite_model.export('gen_ppo.tflite')

    # Compare the outputs
    tflite_output = tfLite_model(sample_input)
    print("PyTorch output:", pytorch_output)
    print("TFLite output:", tflite_output)

if __name__ == "__main__":
    convert_network()