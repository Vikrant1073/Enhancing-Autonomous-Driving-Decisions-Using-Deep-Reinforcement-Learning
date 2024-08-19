## The DQN-VAE model script contains tuning various hyperparameters and
## creating an efficient and stable models to perform training.

import torch
from stable_baselines3 import DQN
from utils.vae_model import VAE

def create_dqn_vae_model(env, log_dir, latent_dim=128):
    
    # Creating a VAE model by importing custom VAE from utils package
    # Also sending to GPU device for faster and parallel performance
    vae_model = VAE(latent_dim=latent_dim).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Creating a DQN model with stable_baseline3 libraries
    model = DQN(
        "CnnPolicy",  # Policy architecture: Convolutional Neural Network (CNN) policy suitable for images
        env,          # The environment in which the DQN agent will be trained
        learning_rate=0.001,  # Learning rate for the optimizer. (A lower rate can help with stability)
        buffer_size=10000,  # Size of the replay buffer that stores past experiences for experience replay
        learning_starts=1000,  # Number of steps taken to collect experiences before starting training
        batch_size=32,  # Number of samples to taken from the replay buffer for each training step
        gamma=0.99,    # Discount factor (value of future rewards compared to immediate rewards)
        train_freq=4,  # Frequency (time steps) at which training is performed
        gradient_steps=1,  # Number of gradient steps to take per training update
        target_update_interval=5000,  # Frequency (time steps) to update the target network
        exploration_fraction=0.3,  # The agent will explore with a decaying epsilon
        exploration_initial_eps=1.0,  # Initial exploration probability (epsilon)
        exploration_final_eps=0.05,  # Final exploration probability after decay
        max_grad_norm=10,  # used for gradient clipping to stabilize training
        verbose=1,  # 1 for info-level logs
        device="cuda" if torch.cuda.is_available() else "cpu",  # Device to run the model on (CUDA if a GPU is available, otherwise CPU)
        tensorboard_log=log_dir  # Directory to save metrics and TensorBoard logs
    )
    return model, vae_model
