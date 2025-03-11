import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class QTable:
    def __init__(self, state_dim=5, action_dim=3, learning_rate=0.01, gamma=0.95):
        self.q_table = {}
        self.lr = learning_rate
        self.gamma = gamma
        self.state_dim = state_dim
        self.action_dim = action_dim

    def discretize_state(self, state):
        # Discretize continuous state space for Q-table and handle NaN values
        state = np.nan_to_num(state, 0.0)  # Replace NaN with 0
        return tuple(np.round(state, 2))

    def get_action(self, state, epsilon=0.1):
        state = self.discretize_state(state)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_dim)

        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        state = self.discretize_state(state)
        next_state = self.discretize_state(next_state)

        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_dim)

        current_q = self.q_table[state][action]
        next_max_q = np.max(self.q_table[next_state])
        td_error = reward + self.gamma * next_max_q - current_q
        new_q = current_q + self.lr * td_error
        self.q_table[state][action] = new_q
        
        return td_error ** 2  # Return MSE loss

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_dim=5, action_dim=3, learning_rate=0.001, gamma=0.95):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = gamma

    def get_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, 2)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def update(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)

        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + self.gamma * next_q_values

        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()  # Return the loss value

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.LogSoftmax(dim=-1)
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

class A2CAgent:
    def __init__(self, state_dim=5, action_dim=3, learning_rate=0.001, gamma=0.95):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma

    def get_action(self, state):
        """Select action based on actor network output"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # Get log probabilities from actor network with stricter clamping
            log_probs = self.model.actor(state).clamp(-10, 2)
            
            # Handle potential numerical instabilities
            try:
                # Convert to probabilities using more stable softmax
                probs = torch.nn.functional.softmax(log_probs, dim=-1)
                
                # Replace any invalid values
                probs = torch.nan_to_num(probs, 1e-6)
                
                # Ensure strict probability bounds
                probs = probs.clamp(1e-6, 1.0)
                
                # Renormalize to ensure sum to 1
                probs = probs / probs.sum(dim=-1, keepdim=True)
                
                # Verify probability distribution
                if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                    # Fallback to uniform distribution if still invalid
                    probs = torch.ones_like(probs) / probs.size(-1)
                
                action = torch.multinomial(probs, 1).item()
            except RuntimeError:
                # Fallback to random action if all else fails
                action = random.randint(0, probs.size(-1) - 1)
                
        return action

    def update(self, state, action, reward, next_state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

        # Calculate advantage with value clipping for stability
        value = self.model.critic(state)
        next_value = self.model.critic(next_state)
        advantage = (reward + self.gamma * next_value.detach() - value).clamp(-10, 10)

        # Calculate actor (policy) loss using log probabilities directly from network
        log_probs = self.model.actor(state)
        action_log_prob = log_probs[0][action]
        actor_loss = -action_log_prob * advantage.detach()

        # Calculate critic (value) loss with clipping
        critic_loss = advantage.pow(2).clamp(0, 100)

        # Combined loss
        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return actor_loss.item(), critic_loss.item()