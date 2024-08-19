## The PPO model script contains tuning various hyperparameters and
## creating an efficient and stable model to perform training.

import torch
from stable_baselines3 import PPO

def create_ppo_model(env, log_dir):

    # Creating a PPO model with stable_baseline3 libraries  
    model = PPO(
        "CnnPolicy",
        learning_rate=0.0001,  # Lower learning rate for stability
        batch_size=32,  # Moderate batch size, also tried 64 and 128.
        n_steps=1024,  # Balanced to fit within episode length and total steps
        n_epochs=3,  # Small number of epochs to avoid overfitting, also tried 5 and 10.
        gamma=0.99,  # High discount factor for long-term rewards
        gae_lambda=0.95,  # Standard lambda for GAE
        clip_range=0.2,  # Standard clipping range
        ent_coef=0.0,  # Since using custom exploration, removed entropy regularization
        vf_coef=0.5,  # Balance between value and policy loss
        max_grad_norm=0.5,  # Gradient clipping to ensure stability
        verbose=1, # 1 for info-level logs
        device="cuda" if torch.cuda.is_available() else "cpu", # Device to run the model on (CUDA if a GPU is available, otherwise CPU)
        tensorboard_log=log_dir, # Directory to save metrics and TensorBoard logs
    )
    return model
