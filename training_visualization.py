import matplotlib.pyplot as plt
import numpy as np

class TrainingVisualizer:
    def __init__(self):
        # Initialize metrics storage
        self.q_table_metrics = {'rewards': [], 'q_values': []}
        self.dqn_metrics = {'losses': [], 'rewards': []}
        self.a2c_metrics = {'actor_losses': [], 'critic_losses': [], 'rewards': []}
        
        # Set up the plotting configuration
        plt.style.use('default')  # Using default style instead of seaborn
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Training Metrics', fontsize=16)
    
    def update_q_table_metrics(self, episode, reward, q_table):
        """Update Q-table metrics"""
        self.q_table_metrics['rewards'].append(reward)
        # Calculate mean of Q-table values, not the dictionary itself
        self.q_table_metrics['q_values'].append(np.mean(list(q_table.values())))
    
    def update_dqn_metrics(self, episode, loss, reward):
        """Update DQN metrics"""
        self.dqn_metrics['losses'].append(loss)
        self.dqn_metrics['rewards'].append(reward)
    
    def update_a2c_metrics(self, episode, actor_loss, critic_loss, reward):
        """Update A2C metrics"""
        self.a2c_metrics['actor_losses'].append(actor_loss)
        self.a2c_metrics['critic_losses'].append(critic_loss)
        self.a2c_metrics['rewards'].append(reward)
    
    def plot_training_metrics(self):
        """Plot all training metrics"""
        # Clear previous plots
        for ax in self.axes.flat:
            ax.clear()
        
        # Plot Q-table metrics
        if self.q_table_metrics['rewards']:
            self.axes[0, 0].plot(self.q_table_metrics['rewards'], label='Q-table')
            self.axes[0, 0].set_title('Q-Learning Rewards')
            self.axes[0, 0].set_xlabel('Episode')
            self.axes[0, 0].set_ylabel('Reward')
            self.axes[0, 0].legend()
            self.axes[0, 0].grid(True)
        
        # Plot DQN metrics
        if self.dqn_metrics['rewards']:
            self.axes[0, 1].plot(self.dqn_metrics['rewards'], label='DQN')
            self.axes[0, 1].set_title('DQN Rewards')
            self.axes[0, 1].set_xlabel('Episode')
            self.axes[0, 1].set_ylabel('Reward')
            self.axes[0, 1].legend()
            self.axes[0, 1].grid(True)
            
            self.axes[1, 0].plot(self.dqn_metrics['losses'], label='Loss')
            self.axes[1, 0].set_title('DQN Loss')
            self.axes[1, 0].set_xlabel('Update Step')
            self.axes[1, 0].set_ylabel('Loss')
            self.axes[1, 0].legend()
            self.axes[1, 0].grid(True)
        
        # Plot A2C metrics
        if self.a2c_metrics['rewards']:
            self.axes[1, 1].plot(self.a2c_metrics['actor_losses'], label='Actor Loss', alpha=0.7)
            self.axes[1, 1].plot(self.a2c_metrics['critic_losses'], label='Critic Loss', alpha=0.7)
            self.axes[1, 1].set_title('A2C Losses')
            self.axes[1, 1].set_xlabel('Episode')
            self.axes[1, 1].set_ylabel('Loss')
            self.axes[1, 1].legend()
            self.axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.pause(0.1)  # Small pause to update the plots