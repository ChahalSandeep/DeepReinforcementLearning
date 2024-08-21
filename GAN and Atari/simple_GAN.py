"""
contains simple gan architecture
"""

# built-in packages

# third party packages
import torch.nn as nn
from tensorflow.compiler.tf2xla.python.xla import self_adjoint_eig

# source packages

# CONSTANTS
DISCR_FILTERS = 64
GENER_FILTERS = 64
LATENT_VECTOR_SIZE = 100

class Discriminator(nn.Module):
    """
    converges the image to single number
    """
    def __init__(self, input_dim):
        """
        Architecture
        - 5 convolutional layers
        - sigmoid activation function (probability that Discriminator is thinks image is real)
        :param input_dim: input shape of image
        """
        super(Discriminator, self).__init__()
        self.conv_nn = nn.Sequential(
            nn.Conv2d(in_channels=input_dim[0], out_channels=DISCR_FILTERS, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS, out_channels=DISCR_FILTERS*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS*2, out_channels=DISCR_FILTERS*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 4, out_channels=DISCR_FILTERS * 8,kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 8),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 8, out_channels=1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = self.conv_nn(x)
        return conv_out.view(-1, 1).squeeze(dim=1)

class Generator(nn.Module):
    """
    -> takes in random vector (latent vector)
    -> de-convolve converts vector into color image of original resolution
    de-convolve the image to (3, 64, 64)
    """
    def __init__(self, output_dim):
        super(Generator, self).__init__()
        self.conv_nn = nn.Sequential(
            nn.ConvTranspose2d(in_channels=LATENT_VECTOR_SIZE, out_channels=GENER_FILTERS * 8,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(GENER_FILTERS * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 8, out_channels=GENER_FILTERS * 4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 4, out_channels=GENER_FILTERS * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 2, out_channels=GENER_FILTERS,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS, out_channels=output_dim[0],
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.conv_nn(x)