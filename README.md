
# Enhancing Autonomous Driving Decisions Using Deep Reinforcement Learning

This document provides a comprehensive explanation of the entire directory structure of the Project, including the purpose of each folder and file within the project. This should give you a clear understanding of how everything works together.

-------------------------------------------------------------------------

## Project Directory: `RL/`

This is the main directory of the project, where all the necessary components are organised into specific subdirectories, enhancing the readability. The structure is designed to separate the various aspects of the reinforcement learning (RL) project, making it easier to manage, understand, and extend in the future.

## 1. `models/` Directory

Purpose: 
The `models/` directory contains the scripts for creating and managing the different types of reinforcement learning models used in the project. Each file here is responsible for setting up a specific type of RL model, including standard models (like DQN and PPO) and models enhanced with a Variational Autoencoder (VAE).

Files:
- `dqn_model.py`: 
  - Contains the code to create a Deep Q-Network (DQN) model using Stable Baselines3. 
  - This model is designed for environments where decisions are made based on discrete actions.

- `ppo_model.py`: 
  - Contains the code to create a Proximal Policy Optimization (PPO) model.
  - PPO is well-suited for continuous action spaces (but used Discrete as to allow comparison) and is known for its stability and reliability in training.

- `dqn_vae_model.py`: 
  - Combines DQN with a VAE for environments where preprocessing the input data from the cameras is beneficial.
  - The VAE helps reduce the dimensionality of the input data, making the training process more efficient.

- `ppo_vae_model.py`: 
  - Similar to `dqn_vae_model.py`, but combines PPO with a VAE.
  - This setup is benefit for input data preprocessing.

## 2. `environments/` Directory

Purpose:
The `environments/` directory contains the definition of the custom environment in which the created RL models will operate, like controlling a car in AirSim.
Files:
- `airsim_car_env.py`:
  - Defines the `AirSimCarEnv` class, which wraps the AirSim API into a Gym compatible environment.
  - This environment includes the car's control logic, state management, and reward calculations, making it compatible with RL algorithms in Stable Baselines3.
  - The environment also handles waypoints, allowing the car to navigate through predefined paths during training.

## 3. `utils/` Directory

Purpose:
The `utils/` directory contains utility functions and a custom made VAE model architecture that support the main RL training processes. 

Files:
- `vae_model.py`:
  - Contains the implementation of a Variational Autoencoder (VAE).
  - The VAE is used for preprocessing high-dimensional input data (images) by encoding it into a lower-dimensional latent space.

- `utils.py`:
  - Includes functions to save and load models, making it easier to manage the training process and checkpoints.

## 4.  Main run files

Purpose:
Each script is designed to train a specific type of model (DQN, PPO, DQN with VAE, PPO with VAE), and includes configurations for logging, evaluation, and saving the trained models.

Files:
- `run_dqn.py`:
  - The main script to train a DQN model on the custom AirSim car environment.
  - Includes TensorBoard logging for monitoring training metrics like rewards and losses, and uses `EvalCallback` to periodically evaluate the model's performance.

- `run_ppo.py`:
  - Similar to `run_dqn.py`, but for training a PPO model.

- `run_dqn_vae.py`:
  - Trains a DQN model enhanced with a VAE, which preprocesses the input data to improve learning efficiency.
  - Suitable for environments with high-dimensional input data like images.

- `run_ppo_vae.py`:
  - Trains a PPO model with a VAE for environments where data preprocessing are beneficial like one used in the project.


## Running the Project:
To run the project:
- Firstly run the airsim environment (Neighbourhood city environment is used in this project)
- Choose the appropriate script (like `run_dqn.py` for training a DQN model).
- Run the script to start the training process, with TensorBoard logging and evaluation callbacks enabled.

During training, TensorBoard will monitor the progress of your models by visualising metrics like rewards, episode lengths, and more. While, eval callbacks will provide further evaluations periodically.

# Appendix Directories

## 1. `logs/` Directory

Purpose: 
The `logs/` directory contains the tensorflow logs, evaluation trackings logs and best models for some runs saved.

Files:
All four models' log files for a run of 20,000 episodes

## 2. `debug_image/` Directory

Purpose:
The `debug_images/` directory contains the first image of each episode in a run as to understand the correctness of the image captured.

## 3. `setup_path/` file

Purpose:
The `setup_path/` file helps find AirSim installation on the device. this file has been directly imported from the AirSim's Reinforcement Learning package. Available [here](https://github.com/microsoft/AirSim/blob/main/PythonClient/reinforcement_learning/setup_path.py).

## 4. `CUDA_test/` file

Purpose:
The `CUDA_path/` file tests the CUDA availability and prints out its version, used here to install supported CuDNN files built to enhance Deep Neural Network's performance.

-------------------------------------------------------------------------

