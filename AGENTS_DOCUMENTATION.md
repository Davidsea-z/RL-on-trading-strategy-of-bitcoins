# Reinforcement Learning Agents Documentation

This documentation provides detailed information about the three reinforcement learning agents implemented in this project: Q-Table, DQN (Deep Q-Network), and A2C (Advantage Actor-Critic).

## Table of Contents
- [Q-Table Agent](#q-table-agent)
- [DQN Agent](#dqn-agent)
- [A2C Agent](#a2c-agent)

## Q-Table Agent

### Overview
The Q-Table agent implements traditional Q-learning using a discretized state space and tabular representation of Q-values.

### Architecture
- Uses a dictionary-based Q-table to store state-action values
- Discretizes continuous state space for practical implementation
- Handles NaN values in state representation

### Initialization Parameters
```python
QTable(state_dim=5, action_dim=3, learning_rate=0.01, gamma=0.95)
```
- `state_dim`: Dimension of the state space (default: 5)
- `action_dim`: Number of possible actions (default: 3)
- `learning_rate`: Learning rate for Q-value updates (default: 0.01)
- `gamma`: Discount factor for future rewards (default: 0.95)

### Training Process
1. State Discretization:
   - Continuous states are discretized using `discretize_state` method
   - NaN values are replaced with 0
   - States are rounded to 2 decimal places

2. Action Selection:
   ```python
   agent.get_action(state, epsilon=0.1)
   ```
   - Uses ε-greedy policy for exploration
   - With probability ε, selects random action
   - Otherwise, selects action with highest Q-value

3. Q-Value Update:
   ```python
   agent.update(state, action, reward, next_state)
   ```
   - Updates Q-values using the Q-learning formula:
   - Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]

### Usage Example
```python
# Initialize agent
q_agent = QTable(state_dim=5, action_dim=3)

# Training loop
for episode in range(num_episodes):
    state = environment.reset()
    done = False
    while not done:
        # Get action using epsilon-greedy policy
        action = q_agent.get_action(state, epsilon=0.1)
        
        # Take action in environment
        next_state, reward, done, _ = environment.step(action)
        
        # Update Q-values
        q_agent.update(state, action, reward, next_state)
        
        state = next_state
```

## DQN Agent

### Overview
The DQN agent implements Deep Q-Network with experience replay and target network for stable training.

### Architecture

#### Neural Network Structure
```python
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
```
- Three fully connected layers
- ReLU activation functions
- Output layer size matches action space

### Initialization Parameters
```python
DQNAgent(state_dim=5, action_dim=3, learning_rate=0.001, gamma=0.95)
```
- `state_dim`: Dimension of the state space
- `action_dim`: Number of possible actions
- `learning_rate`: Learning rate for neural network
- `gamma`: Discount factor for future rewards

### Training Process
1. Experience Collection:
   - Stores transitions (state, action, reward, next_state) in replay memory
   - Memory size: 10000 transitions
   - Batch size: 64 samples

2. Action Selection:
   ```python
   agent.get_action(state, epsilon=0.1)
   ```
   - Uses ε-greedy policy
   - Converts state to PyTorch tensor
   - Forward pass through policy network

3. Network Update:
   ```python
   agent.update(state, action, reward, next_state)
   ```
   - Samples random batch from replay memory
   - Computes target Q-values using target network
   - Updates policy network using MSE loss

### Usage Example
```python
# Initialize agent
dqn_agent = DQNAgent(state_dim=5, action_dim=3)

# Training loop
for episode in range(num_episodes):
    state = environment.reset()
    done = False
    while not done:
        # Get action
        action = dqn_agent.get_action(state, epsilon=0.1)
        
        # Take action in environment
        next_state, reward, done, _ = environment.step(action)
        
        # Update networks
        dqn_agent.update(state, action, reward, next_state)
        
        state = next_state
```

## A2C Agent

### Overview
The A2C (Advantage Actor-Critic) agent implements a policy gradient method with separate actor and critic networks.

### Architecture

#### Actor Network
```python
self.actor = nn.Sequential(
    nn.Linear(state_dim, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, action_dim),
    nn.LogSoftmax(dim=-1)
)
```
- Outputs action probabilities
- Uses LogSoftmax for numerical stability

#### Critic Network
```python
self.critic = nn.Sequential(
    nn.Linear(state_dim, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)
```
- Estimates state values
- Single output for value function

### Initialization Parameters
```python
A2CAgent(state_dim=5, action_dim=3, learning_rate=0.001, gamma=0.95)
```
- `state_dim`: Dimension of the state space
- `action_dim`: Number of possible actions
- `learning_rate`: Learning rate for both networks
- `gamma`: Discount factor for future rewards

### Training Process
1. Action Selection:
   ```python
   agent.get_action(state)
   ```
   - Converts state to PyTorch tensor
   - Computes action probabilities with numerical stability
   - Samples action using multinomial distribution

2. Network Update:
   ```python
   agent.update(state, action, reward, next_state)
   ```
   - Calculates advantage using critic network
   - Updates actor network using policy gradient
   - Updates critic network using value loss
   - Uses clipping for stability

### Usage Example
```python
# Initialize agent
a2c_agent = A2CAgent(state_dim=5, action_dim=3)

# Training loop
for episode in range(num_episodes):
    state = environment.reset()
    done = False
    while not done:
        # Get action
        action = a2c_agent.get_action(state)
        
        # Take action in environment
        next_state, reward, done, _ = environment.step(action)
        
        # Update networks
        a2c_agent.update(state, action, reward, next_state)
        
        state = next_state
```

## Implementation Notes

### Numerical Stability
- The A2C implementation includes several stability measures:
  - Log probabilities are clamped to [-10, 2]
  - Advantage values are clamped to [-10, 10]
  - Critic loss is clamped to [0, 100]
  - Probabilities are bounded and renormalized

### Device Management
- All agents with neural networks support both CPU and GPU:
  ```python
  self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ```

### Error Handling
- Includes fallback mechanisms for numerical instabilities
- Handles NaN values in state representations
- Provides graceful degradation to random actions when necessary

### Memory Management
- DQN uses a fixed-size replay buffer (10000 transitions)
- Batch processing for efficient training
- Automatic memory cleanup through Python's garbage collection