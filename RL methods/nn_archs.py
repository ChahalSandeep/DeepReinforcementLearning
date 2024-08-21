"""
contains the architecture of the neural networks used in RL methods
"""
# built-in packages

# third party packages
from torch import nn

# project packages

# CONSTANTS
HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE_TO_SELECT = 70

class CrossEntropyNeuralNet(nn.Module):
    """
    One hidden layer neural network
    hidden neurons are arbitrary and for example here feel free to play around
    """
    def __init__(self, observation_size, hidden_size, n_actions):
        super(CrossEntropyNeuralNet, self).__init__()
        self.cenn = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.cenn(x)