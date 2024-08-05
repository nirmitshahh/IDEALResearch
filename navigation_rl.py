import numpy as np
import random
from collections import deque

class NavigationAgent:
    """
    Reinforcement learning agent for autonomous navigation
    of the terrain-robust rover with gait device.
    """
    
    def __init__(self, state_size=8, action_size=4, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Q-learning parameters
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.95  # Discount factor
        
        # Q-table (simple implementation)
        self.q_table = {}
        self.memory = deque(maxlen=2000)  # experience replay buffer
        
    def get_state_key(self, state):
        """Convert state array to hashable key"""
        # Discretize state for Q-table
        discretized = tuple(int(s * 10) for s in state)
        return tuple(discretized)
    
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        return np.argmax(self.q_table[state_key])
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=32):
        """Train agent on batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in batch:
            state_key = self.get_state_key(state)
            next_state_key = self.get_state_key(next_state)
            
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.action_size)
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = np.zeros(self.action_size)
            
            target = reward
            if not done:
                target = reward + self.gamma * np.max(self.q_table[next_state_key])
            
            current_q = self.q_table[state_key][action]
            self.q_table[state_key][action] = current_q + self.learning_rate * (target - current_q)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath):
        """Save Q-table to file"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
    
    def load_model(self, filepath):
        """Load Q-table from file"""
        import pickle
        with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)
    # save/load so we don't have to retrain every time

class TerrainEnvironment:
    """
    Simulated environment for training navigation agent.
    """
    
    def __init__(self, size=(10, 10)):
        self.size = size
        self.agent_pos = [0, 0]
        self.target_pos = [size[0]-1, size[1]-1]
        self.obstacles = []
        
    def reset(self):
        """Reset environment to initial state"""
        self.agent_pos = [0, 0]
        return self.get_state()
    
    def get_state(self):
        """Get current state representation"""
        # State includes: agent position, target position, distance, terrain info
        dx = self.target_pos[0] - self.agent_pos[0]
        dy = self.target_pos[1] - self.agent_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # Normalize
        state = [
            self.agent_pos[0] / self.size[0],
            self.agent_pos[1] / self.size[1],
            self.target_pos[0] / self.size[0],
            self.target_pos[1] / self.size[1],
            dx / self.size[0],
            dy / self.size[1],
            distance / np.sqrt(self.size[0]**2 + self.size[1]**2),
            0.5  # Placeholder for terrain roughness
        ]
        return np.array(state)
    
    def step(self, action):
        """Execute action and return new state, reward, done"""
        # Actions: 0=up, 1=down, 2=left, 3=right
        moves = {0: [0, 1], 1: [0, -1], 2: [-1, 0], 3: [1, 0]}
        
        move = moves[action]
        new_pos = [self.agent_pos[0] + move[0], self.agent_pos[1] + move[1]]
        
        # Check bounds
        if (0 <= new_pos[0] < self.size[0] and 
            0 <= new_pos[1] < self.size[1]):
            self.agent_pos = new_pos
        
        # Calculate reward
        reward = -0.1  # Small penalty for each step
        if self.agent_pos == self.target_pos:
            reward = 10.0  # Large reward for reaching target
        
        done = (self.agent_pos == self.target_pos)
        
        return self.get_state(), reward, done

