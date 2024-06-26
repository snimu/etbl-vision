
import itertools

import torch 
from torch import nn


def reduce_dimension(total_steps: int, current_step: int, initial_dim: int, final_dim: int):
    """
    Reduce the dimension of a tensor from initial_dim to final_dim over total_steps.
    """
    return round(initial_dim - (initial_dim - final_dim) * current_step / total_steps)


class MLP(nn.Module):
    def __init__(self, num_layers: int = 5, imsize: int = 28*28, num_classes: int = 10):
        super().__init__()
        sizes = [reduce_dimension(num_layers-1, i, imsize, num_classes) for i in range(num_layers)]
        self.layers = nn.ModuleList(
            [
                nn.Linear(s1, s2)
                for s1, s2 in itertools.pairwise(sizes)
            ]
        )
        self.relu = nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        for layer in self.layers:
            x = self.relu(layer(x))

        return self.softmax(x)
