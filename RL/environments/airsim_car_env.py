# This is the code file for AirSim Environment used for the project.

import setup_path # Used to find the AirSim module in the parents folders.
import gym # Used to interact with the environment
import airsim # Simulation

import numpy as np # Array representation and calculation for images
from gym import spaces # For defining the space of the environments to interact with actions
from PIL import Image # Image processing using Pillow

# Logging and saving purpose libraries
import os
import time
import logging

logging.basicConfig(level=logging.DEBUG)

import random


## First created an AirSim Environment for the vehicle to interact. (Creating classes so to make it easy to import in main files)

class AirSimCarEnv(gym.Env):

    ## Initialising with various hyperparameters to implement a Custom Decay for the probability of model to take random decisions.
    def __init__(self, ip_address="127.0.0.1", image_shape=(1, 84, 84), waypoints=None,
                 exploration_prob=1.0, exploration_final_prob=0.1, exploration_decay_steps=1000):
        
        super(AirSimCarEnv, self).__init__() 
        
        # Creating an instance for the vehicle type car with IP address to connect with running AirSim module.
        self.car = airsim.CarClient(ip=ip_address)
        self.car_controls = airsim.CarControls() # To establish instance for car controls as well.
        self.image_saved = False
        
        # Using the image observation space for compatibility with CnnPolicy
        self.observation_space = spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8)
        
        # Define the discrete action space for DQN compatibility (As DQN does not support continuous action space)
        self.action_space = spaces.Discrete(6) # With 6 actions
        
        # Camera settings
        self.camera_name = "front_center" # Custom camera generated attached to the vehicle
        self.image_type = airsim.ImageType.Scene
        
        # Requesting the image captured by our custom camera
        self.image_request = airsim.ImageRequest(
            self.camera_name,
            self.image_type,
            False,
            False
        )
        
        # Stores waypoints (As we are using waypoint method to implement lane driving)
        self.waypoints = waypoints if waypoints is not None else []

        # Initialized other necessary state variables
        self.current_waypoint_index = 0
        self.image_shape = image_shape
        self.state = {
            "position": np.zeros(3), # initialising the current position
            "collision": False, # current colision state
            "previous_action": None # at the start no previous action, hence, none.
        }
        self.total_reward = 0
        self.episode_length = 0
        
        # Custom exploration parameters for each model. Ensures exploration initially with a linear decay
        self.exploration_prob = exploration_prob
        self.exploration_final_prob = exploration_final_prob
        self.exploration_decay_steps = exploration_decay_steps
        self.current_step = 0


    # Rest function involves resetting the environment to the initial state
    def reset(self):
        self.car.reset()
        self.car.enableApiControl(True)
        self.car.armDisarm(True)
        
        # Reset to the first waypoint
        # Future scope: Can optimise it to the closest waypoint
        self.current_waypoint_index = 0
        
        # Reset tracking variables
        self.total_reward = 0
        self.episode_length = 0
        self.current_step = 0
        
        # Captures an initial observation
        return self._get_obs()


    
    # Step function will 
    def step(self, action):
        logging.debug(f"Step called with action: {action}")
        
        # Updating the exploration probability based on the decay
        self._update_exploration_prob()
        
        # Custom exploration: with some probability, chooses a random action
        if random.random() < self.exploration_prob:
            action = self.action_space.sample()
            logging.debug(f"Random action selected: {action}") # Viualising the action after each step
        
        # Performs the action
        self._do_action(action)
        
        # Captures the next observation
        obs = self._get_obs()
        
        # Calculates reward, done, and info
        reward, done = self._compute_reward(action)
        self.total_reward += reward # Adding rewards for each step untill the episode is over
        self.episode_length += 1 
        self.current_step += 1
        
        # Information required to update after each step
        info = {
            "position": self.state["position"], # Change in position
            "collision": self.state["collision"], # Detecting Collission
            "total_reward": self.total_reward, # Calculating total rewards
            "episode_length": self.episode_length # Steps in an episode
        }
        
        return obs, reward, done, info

    # Custom decay to the exploration probability
    def _update_exploration_prob(self):
        # Linear decay function
        decay_rate = (self.exploration_prob - self.exploration_final_prob) / self.exploration_decay_steps
        self.exploration_prob = max(self.exploration_final_prob, self.exploration_prob - decay_rate) # Stops decaying when reaches min probability
        logging.debug(f"Exploration probability updated to: {self.exploration_prob}")


    # Selecting an action is crucial for the thesis as it should be done with careful consideration.
    def _do_action(self, action):
        logging.debug(f"Performing action: {action}")

        # Initialising the state of the movements of the car
        # It will start throttling and breaks were removed as the step starts
        self.car_controls.brake = 0
        self.car_controls.throttle = 1

        # Defining actions
        if action == 0: # Break action
            self.car_controls.throttle = 0
            self.car_controls.brake = 1

        elif action == 1: # Idle Action, basically used to move forward with previous movements
            self.car_controls.steering = 0
        
        elif action == 2: # Moving to the right
            self.car_controls.steering = 0.5
        
        elif action == 3: # Moving to the left
            self.car_controls.steering = -0.5
        
        elif action == 4: # Slight movement to the right with lesser throttle
            self.car_controls.throttle = 0.8
            self.car_controls.steering = 0.25

        else: # Slight movement to the left with lesser throttle
            self.car_controls.throttle = 0.8
            self.car_controls.steering = -0.25

        # Sets the control as the action is chosen
        self.car.setCarControls(self.car_controls)
        # Little delay is added to keep the car moving in the action selected
        time.sleep(0.01)


    # Function get the observations
    def _get_obs(self):
        logging.debug("Getting observation.")

        # Saving image caputures from the custom camera
        responses = self.car.simGetImages([self.image_request])
        
        # Checking successful gathering of image data
        if responses is None or len(responses) == 0:
            logging.warning("No image data received from AirSim.")
            return np.zeros(self.observation_space.shape, dtype=np.uint8)

        # Converting the image buffer to 1D array to preprocess like reshape and
        img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)
        
        # Grayscale conversion, resizing and reshaping to fit in the model
        image = Image.fromarray(img_rgb).convert('L')
        image = image.resize((84, 84))
        obs_image = np.array(image).reshape(1, 84, 84)
        
        # Saving the current state on position and collision detection
        self.state["position"] = np.array(self.car.simGetVehiclePose().position.to_numpy_array())
        self.state["collision"] = self.car.simGetCollisionInfo().has_collided

        logging.debug(f"Returning observation: {obs_image.shape}")
        return obs_image


    # Function to calculate rewards attained by the modle at each step
    # Another crucial function to develop reward rules with care
    def _compute_reward(self, action):
        logging.debug("Computing reward.")
        reward = 0
        done = False

        # As waypoint strategy is used to achieve lane driving 
        if self.waypoints and self.current_waypoint_index < len(self.waypoints):
            target_position = self.waypoints[self.current_waypoint_index]
            car_position = self.state["position"]

            # Calculating distance from the next waypoint
            distance_to_target = np.linalg.norm(car_position - target_position)
            
            # Distance-based reward
            reward = -distance_to_target / 20 # small negative value at each step to motivate the vehicle to reach early at the next goal.

            # Taking the approximate of 5m to allowing the goal achieved
            if distance_to_target < 5.0:
                self.current_waypoint_index += 1

                # Condition to check the last waypoint
                if self.current_waypoint_index >= len(self.waypoints):
                    reward += 300  # Big reward for reaching the final waypoint
                    done = True # Reseting the episode
                else:
                    reward += 80 * self.current_waypoint_index  # Reward for reaching waypoints
        
        # Significant Penalty for collision and resets the episode
        if self.state["collision"]:
            reward -= 50 + (self.current_waypoint_index * 5)
            done = True

        # Penalty for repeating actions to avoid getting stuck at the same action
        if action == self.state["previous_action"]:
            reward -= 10  

        # Small penalty for each step to encourage faster completion
        reward -= 0.5  
        
        # Saving action for the next step to check its repetition
        self.state["previous_action"] = action

        logging.debug(f"Computed reward: {reward}, Done: {done}")
        return reward, done

    def render(self, mode='human'):
        pass

    # Function to close the API calls and opens keyboard mode for manual driving
    def close(self):
        self.car.reset()
        self.car.enableApiControl(False)
        self.car.armDisarm(False)
