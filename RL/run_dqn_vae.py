## Main script to run DQN model 
# integrated with VAE for dimension reduction
# without loosing important features.

import gym # To interact with the environment
import numpy as np # For working with arrays
import torch # Using for creating tensors

# Used Stablebaselin3 for handelling multitasking as well as monitoring
from stable_baselines3.common.monitor import Monitor # tracks statistics for rewards
from stable_baselines3.common.vec_env import DummyVecEnv # for mutitasking with different environments together
from stable_baselines3.common.callbacks import EvalCallback # Periodic logging of various metrics

# Importing model and environment
from models.dqn_vae_model import create_dqn_vae_model
from environments.airsim_car_env import AirSimCarEnv

# Using tensorboard's summary writer to visualise real time logs updates
from torch.utils.tensorboard import SummaryWriter

# For opening a web window for tensor board logs
import subprocess 
import webbrowser

# Launch TensorBoard before starting the training, specifying log directory and running on local host
log_dir = "./logs/dqn_vae" 
subprocess.Popen(['tensorboard', '--logdir', log_dir])
webbrowser.open("http://localhost:6006")

# Setting up the environment and its waypoints (for 100m, having small goals after every 10m)
waypoints = [np.array([10, 0, 0]), np.array([20, 0, 0]), np.array([30, 0, 0]), np.array([40, 0, 0]), np.array([50, 0, 0]), np.array([60, 0, 0]), np.array([70, 0, 0]), np.array([80, 0, 0]), np.array([90, 0, 0]), np.array([100, 0, 0])]
env = DummyVecEnv([lambda: Monitor(AirSimCarEnv(ip_address="127.0.0.1", image_shape=(1, 84, 84), waypoints=waypoints))])

# Create the DQN + VAE model using function imported from the model script
model, vae_model = create_dqn_vae_model(env, log_dir)

# Setting up the evaluation environment and callback (frequency is after every 100 timsteps)
eval_env = DummyVecEnv([lambda: Monitor(AirSimCarEnv(ip_address="127.0.0.1", image_shape=(1, 84, 84), waypoints=waypoints))])
eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/dqn_vae_best_model',
                             log_path='./logs/dqn_vae_eval', eval_freq=100,
                             deterministic=True, render=False)

# Initialized TensorBoard SummaryWriter
writer = SummaryWriter(log_dir=log_dir)

# Training loop initialisation
total_steps = 20000 
obs = env.reset()

# Metric tracking
total_reward = 0
episode_length = 0
episode_count =0

# For every step till the max limit
for step in range(total_steps):

    # Preprocess the observation with the VAE
    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(model.device) / 255.0  # Normalize the input
    
    with torch.no_grad():
        latent_rep, _, _ = vae_model(obs_tensor) # Dimension reduction of the image data using VAE
    latent_rep = latent_rep.cpu().numpy()  # Converts back to numpy for the DQN model

    # Feeds the latent representation into the DQN model
    action, _states = model.predict(latent_rep)
    
    # Takes a step in the environment
    obs, reward, done, info = env.step(action)

    # Modifying rewards and updating the episode length
    total_reward += reward
    episode_length += 1

    # Logging the reward for each step
    writer.add_scalar('Reward/Step', reward, step)

    if done:
        # Logging episode metrics
        episode_count +=1
        writer.add_scalar('Episode/TotalReward', total_reward, episode_count)
        writer.add_scalar('Episode/AverageReward', total_reward / episode_length, episode_count)
        writer.add_scalar('Episode/Length', episode_length, episode_count)

        # Resets metrics
        total_reward = 0
        episode_length = 0
        obs = env.reset()

# Closes the TensorBoard writer after training
writer.close()

# Trains the model with the evaluation callback
model.learn(total_timesteps=total_steps, callback=eval_callback)

# Saves the model
model.save("dqn_vae_airsim_car_policy")
