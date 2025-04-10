

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque

# Import your ROS-enabled environment.
# Make sure swarm_gym.py defines DroneExplorationEnv accordingly.
from rl_interface.grid_world import DroneExplorationEnv

# Set device to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------- Helper Functions -------------------------
# Downsampling factor to reduce input and action space dimensions
COARSE_FACTOR = 10

def downsample_map(full_map):
    """
    Downsamples the voxel map by taking every COARSE_FACTOR-th voxel.
    """
    return full_map[::COARSE_FACTOR, ::COARSE_FACTOR, ::COARSE_FACTOR]

def flatten_observation(obs, num_drones):
    """
    Flattens a coarse version of the map and the drone positions.
    Drone positions are scaled down by COARSE_FACTOR.
    """
    coarse_map = downsample_map(obs["map"])
    # Scale drone positions accordingly (integer division)
    coarse_positions = (obs["drone_positions"] // COARSE_FACTOR).astype(np.int32)
    return np.concatenate([coarse_map.flatten(), coarse_positions.flatten()])

def discrete_to_action(action_idx, coarse_shape, coarse_factor):
    """
    Convert a discrete action index into a 3D grid coordinate (for one drone)
    and upscale it to the full-resolution grid.
    """
    coord = np.unravel_index(action_idx, coarse_shape)
    return np.array(coord) * coarse_factor

# ------------------------- DQN Network and Agent -------------------------
class DQN(nn.Module):
    def __init__(self, state_size, total_action_size, num_drones, coarse_action_size):
        """
        total_action_size = num_drones * coarse_action_size.
        The network outputs a vector that is reshaped into (num_drones, coarse_action_size).
        """
        super(DQN, self).__init__()
        self.num_drones = num_drones
        self.coarse_action_size = coarse_action_size
        
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, total_action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        # Reshape to (batch, num_drones, coarse_action_size)
        x = x.view(-1, self.num_drones, self.coarse_action_size)
        return x

class DQNAgent:
    def __init__(self, state_size, total_action_size, num_drones, coarse_action_size):
        self.state_size = state_size
        self.total_action_size = total_action_size
        self.num_drones = num_drones
        self.coarse_action_size = coarse_action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        self.model = DQN(state_size, total_action_size, num_drones, coarse_action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        # action is a numpy array of shape (num_drones,) containing discrete indices for each drone.
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, coarse_shape):
        if np.random.rand() <= self.epsilon:
            # Random discrete action for each drone
            return np.array([np.random.randint(self.coarse_action_size) for _ in range(self.num_drones)])
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(state_tensor)  # shape: (1, num_drones, coarse_action_size)
        # For each drone, pick the action with the highest Q value.
        actions = torch.argmax(q_values, dim=2)  # shape: (1, num_drones)
        return actions.cpu().numpy()[0]

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = torch.FloatTensor(states).to(device)          # (batch, state_size)
        next_states = torch.FloatTensor(next_states).to(device)
        rewards = torch.FloatTensor(rewards).to(device)          # (batch,)
        dones = torch.FloatTensor(dones).to(device)              # (batch,)
        actions = torch.LongTensor(actions).to(device)           # (batch, num_drones)
        
        batch_size_val = states.shape[0]
        current_q = self.model(states)  # (batch, num_drones, coarse_action_size)
        # Gather Q values for the actions taken (for each drone)
        current_q = current_q.gather(2, actions.view(batch_size_val, self.num_drones, 1)).squeeze(2)
        # Sum Q values over drones for joint action value.
        current_q_sum = torch.sum(current_q, dim=1)  # (batch,)
        
        next_q = self.model(next_states)
        next_q_max = torch.max(next_q, dim=2)[0]       # (batch, num_drones)
        next_q_sum = torch.sum(next_q_max, dim=1)        # (batch,)
        
        targets = rewards + (1 - dones) * self.gamma * next_q_sum
        
        loss = self.criterion(current_q_sum, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filepath="dqn_model.pth"):
        torch.save(self.model.state_dict(), filepath)


# ------------------------- Main Training Loop -------------------------

def main():
    # Create your ROS-enabled environment.
    env = DroneExplorationEnv()
    # Set the number of drones in the environment (derived from the ROS node parameter)
    env.num_drones = env.node.n_rob_
    num_drones = env.num_drones

    # Reset the environment to get the initial observation.
    sample_obs = env.reset()
    # Ensure a default map shape if not yet set (e.g., (20,20,20))
    if np.prod(sample_obs["map"].shape) == 0:
        default_shape = (20, 20, 20)
        sample_obs["map"] = np.zeros(default_shape, dtype=np.int8)
    coarse_map = downsample_map(sample_obs["map"])
    coarse_drone_positions = (sample_obs["drone_positions"] // COARSE_FACTOR).astype(np.int32)
    state_vec = np.concatenate([coarse_map.flatten(), coarse_drone_positions.flatten()])
    state_size = state_vec.shape[0]

    # Define coarse action space shape based on the downsampled map.
    coarse_shape = coarse_map.shape  # e.g., (dim1, dim2, dim3)
    coarse_action_size = int(np.prod(coarse_shape))
    total_action_size = num_drones * coarse_action_size

    print(f"State Size: {state_size}, Coarse Action Size per Drone: {coarse_action_size}, Total Action Size: {total_action_size}")

    agent = DQNAgent(state_size, total_action_size, num_drones, coarse_action_size)
    episodes = 100  # Adjust as needed
    batch_size = 32

    for episode in range(episodes):
        obs = env.reset()
        state = flatten_observation(obs, num_drones)
        done = False
        total_reward = 0

        while not done:
            # Agent selects a discrete action index for each drone.
            action_indices = agent.act(state, coarse_shape)  # shape: (num_drones,)
            # Convert each discrete index into a 3D target coordinate.
            actions = tuple(discrete_to_action(idx, coarse_shape, COARSE_FACTOR) for idx in action_indices)
            
            next_obs, reward, done, _ = env.step(actions)
            next_state = flatten_observation(next_obs, num_drones)
            agent.remember(state, action_indices, reward, next_state, done)
            state = next_state
            total_reward += reward
            print(reward)

        agent.replay(batch_size)
        print(f"Episode {episode+1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")
    
    agent.save_model("dqn_model.pth")
    print("Model saved.")

if __name__ == "__main__":
    main()

