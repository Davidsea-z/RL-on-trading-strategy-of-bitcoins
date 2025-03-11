import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trading_env import BitcoinTradingEnv
from agents import QTable, DQNAgent, A2CAgent
from indicators import prepare_indicators
from data_api import BitcoinDataAPI
from training_visualization import TrainingVisualizer

def load_data(file_path):
    """Load and preprocess Bitcoin price data"""
    api = BitcoinDataAPI()
    prices, _ = api.prepare_data_for_trading(period='1y', interval='1d')
    return prices

def train_agent(env, agent, visualizer, agent_type, episodes=100, epsilon=0.1):
    """Train an RL agent"""
    returns = []
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_return = 0
        
        while not done:
            # A2C agent doesn't use epsilon for exploration
            if isinstance(agent, A2CAgent):
                action = agent.get_action(state)
            else:
                action = agent.get_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            
            # Update agent and track metrics
            if isinstance(agent, QTable):
                agent.update(state, action, reward, next_state)
                visualizer.update_q_table_metrics(episode, reward, agent.q_table)
            elif isinstance(agent, DQNAgent):
                loss = agent.update(state, action, reward, next_state)
                visualizer.update_dqn_metrics(episode, loss, reward)
            elif isinstance(agent, A2CAgent):
                actor_loss, critic_loss = agent.update(state, action, reward, next_state)
                visualizer.update_a2c_metrics(episode, actor_loss, critic_loss, reward)
            
            state = next_state
            episode_return += reward
            
        returns.append(episode_return)
        if (episode + 1) % 10 == 0:
            print(f'Episode {episode + 1}, Return: {episode_return:.2f}')
            visualizer.plot_training_metrics()
    
    return returns

def evaluate_agent(env, agent, episodes=10):
    """Evaluate a trained agent"""
    returns = []
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_return = 0
        
        while not done:
            # A2C agent doesn't use epsilon
            if isinstance(agent, A2CAgent):
                action = agent.get_action(state)
            else:
                action = agent.get_action(state, epsilon=0)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_return += reward
            
        returns.append(episode_return)
    
    return np.mean(returns)

def main():
    # Create results directory if it doesn't exist
    import os
    results_dir = '/Users/mac/Documents/MMAT5392 AI/Project/results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Load and prepare data
    prices = load_data('bitcoin_prices.csv')
    data = prepare_indicators(prices)
    
    # Create environment and visualizer
    env = BitcoinTradingEnv(data)
    visualizer = TrainingVisualizer()
    
    # Initialize agents
    q_agent = QTable()
    dqn_agent = DQNAgent()
    a2c_agent = A2CAgent()
    
    # Train agents and collect metrics
    print('\nTraining Q-Learning Agent...')
    q_returns = train_agent(env, q_agent, visualizer, 'q_table')
    
    print('\nTraining DQN Agent...')
    dqn_returns = train_agent(env, dqn_agent, visualizer, 'dqn')
    
    print('\nTraining A2C Agent...')
    a2c_returns = train_agent(env, a2c_agent, visualizer, 'a2c')
    
    # Plot and save returns comparison
    plt.figure(figsize=(12, 6))
    plt.plot(q_returns, label='Q-Learning')
    plt.plot(dqn_returns, label='DQN')
    plt.plot(a2c_returns, label='A2C')
    plt.title('Training Returns Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True)
    plt.savefig('/Users/mac/Documents/MMAT5392 AI/Project/results/returns_comparison.png')
    plt.close()
    
    # Plot and save Q-Learning metrics
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(q_returns, label='Returns')
    plt.title('Q-Learning Training Metrics')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(visualizer.q_table_metrics['rewards'], label='Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('/Users/mac/Documents/MMAT5392 AI/Project/results/q_learning_metrics.png')
    plt.close()
    
    # Plot and save DQN metrics
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(dqn_returns, label='Returns')
    plt.title('DQN Training Metrics')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(visualizer.dqn_metrics['losses'], label='Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('/Users/mac/Documents/MMAT5392 AI/Project/results/dqn_metrics.png')
    plt.close()
    
    # Plot and save A2C metrics
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(a2c_returns, label='Returns')
    plt.title('A2C Training Metrics')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(visualizer.a2c_metrics['actor_losses'], label='Actor Loss')
    plt.ylabel('Actor Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(visualizer.a2c_metrics['critic_losses'], label='Critic Loss')
    plt.xlabel('Episode')
    plt.ylabel('Critic Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('/Users/mac/Documents/MMAT5392 AI/Project/results/a2c_metrics.png')
    plt.close()
    
    # Evaluate agents
    print('\nEvaluating Agents...')
    q_eval = evaluate_agent(env, q_agent)
    dqn_eval = evaluate_agent(env, dqn_agent)
    a2c_eval = evaluate_agent(env, a2c_agent)
    
    print(f'Q-Learning Average Return: {q_eval:.2f}')
    print(f'DQN Average Return: {dqn_eval:.2f}')
    print(f'A2C Average Return: {a2c_eval:.2f}')

if __name__ == '__main__':
    main()