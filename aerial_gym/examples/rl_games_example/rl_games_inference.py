from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, path):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, 64)
        self.hidden_fc1 = nn.Linear(64, 32)
        self.output_fc = nn.Linear(32, output_dim)

        self.network = nn.Sequential(
            OrderedDict(
                [
                    ("0", self.input_fc),
                    ("elu1", nn.ReLU()),
                    ("2", self.hidden_fc1),
                    ("elu2", nn.ReLU()),
                    ("mu", self.output_fc),
                ]
            )
        )
        self.load_network(path)

    def load_network(self, path):
        sd = torch.load(path)["model"]

        # clean the state dict and load it
        od2 = OrderedDict()
        for key in sd:
            key2 = str(key).replace("a2c_network.actor_mlp.", "")
            key2 = key2.replace("a2c_network.", "")
            if "a2c_network" in key2 or "value" in key2 or "sigma" in key2:
                continue
            else:
                print(key2)
                od2[key2] = sd[str(key)]
        # strictly load the state dict
        self.network.load_state_dict(od2, strict=True)
        print("Loaded MLP network from {}".format(path))

    def forward(self, x):
        return self.network(x)
