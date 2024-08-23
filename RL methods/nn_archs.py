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

    how this network works:
    -> takes in a single observation from environment as input
    -> outputs a number for every action we can perform (probability distribution over actions)
    straightforward way would be to include a softmax non-linearity after last layer
    -> we don't include softmax to increase numerical stability
    rather than calculating softmax (uses exponentiation)
    and then calculating cross entropy (uses logarithm of probabilities).
    nn.CrossEntropyLoss combines both softmax and cross entropy into more numerical stable expression.

    CrossEntropyLoss unnormalizes values from the network (logits)
    downside = need to apply softmax everytime we need to get probability as output networks
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
