## This script is used to develop an architecture for the VAE model

import torch
import torch.nn as nn # Importing neural network package of torch

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        # Defining the encoder part
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 1 input channel, 32 output channels, 4x4 kernel, stride of 2, padding of 1
            nn.ReLU(),  # ReLU activation function is used
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32 input channels, 64 output channels, 4x4 kernel, stride of 2, padding of 1
            nn.ReLU(),  # ReLU activation function is used
            nn.Flatten(),  # Flatten the output to a 1D tensor
            nn.Linear(64 * 21 * 21, 512),  # Fully connected layer with 512 output units
            nn.ReLU(),  # ReLU activation function is used
            nn.Linear(512, latent_dim * 2)  # Final layer with output size 2 * latent_dim (mu and log_var)
        )
        
        # Defining the decoder part (Not utilised in the project, but still developed to test the information loss)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),  # Fully connected layer from latent_dim to 512 units
            nn.ReLU(),  # ReLU activation function is used
            nn.Linear(512, 64 * 21 * 21),  # Fully connected layer to expand to 64 * 21 * 21 units
            nn.ReLU(),  # ReLU activation function is used
            nn.Unflatten(1, (64, 21, 21)),  # Reshape the 1D tensor back to (64, 21, 21)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Transposed convolution to increase spatial dimensions, 64 input channels to 32 output channels
            nn.ReLU(),  # ReLU activation function is used
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # Final transposed convolution for the original input dimensions, 32 input channels to 1 output channel
            nn.Sigmoid()  # Sigmoid to output values in range [0, 1] (for image reconstruction)
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)  # Computed the standard deviation from the log variance
        eps = torch.randn_like(std)  # Sampled epsilon from a standard normal distribution
        return mu + eps * std  # Applied the reparameterization rule: z = mu + std * epsilon

    def forward(self, x):
        x = self.encoder(x)  # Passing the input through the encoder to get mu and log_var
        mu, log_var = torch.chunk(x, 2, dim=-1)  # Then, splitting the output into mu and log_var along the last dimension
        z = self.reparameterize(mu, log_var)  # Sampling the latent vector z using the reparameterization rule
        return self.decoder(z), mu, log_var  # Pass z through the decoder to reconstruct the input, and return the reconstructed image along with the parameters
