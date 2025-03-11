import gym
import numpy as np
from gym import spaces

class BitcoinTradingEnv(gym.Env):
    """Custom Environment for Bitcoin trading"""
    metadata = {'render.modes': ['human']}

    def __init__(self, data, initial_balance=10000):
        super(BitcoinTradingEnv, self).__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0

        # Actions: 0 (HOLD), 1 (BUY), 2 (SELL)
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [balance, btc_held, btc_price, MACD, RVI]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -np.inf, -np.inf]),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf]),
            dtype=np.float32
        )
        
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.btc_held = 0
        self.current_step = 0
        return self._get_observation()

    def _get_observation(self):
        obs = np.array([
            float(self.balance),
            float(self.btc_held),
            float(self.data[self.current_step]['price']),
            float(self.data[self.current_step]['macd']),
            float(self.data[self.current_step]['rvi'])
        ])
        return obs

    def step(self, action):
        current_price = self.data[self.current_step]['price']
        
        # Execute trade
        if action == 1:  # BUY
            btc_to_buy = self.balance / current_price
            self.btc_held += btc_to_buy
            self.balance = 0
        elif action == 2:  # SELL
            self.balance += self.btc_held * current_price
            self.btc_held = 0

        # Move to next timestep
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        # Calculate reward (profit/loss)
        new_portfolio_value = self.balance + self.btc_held * current_price
        reward = new_portfolio_value - self.initial_balance

        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        print(f'Step: {self.current_step}')
        print(f'Balance: ${self.balance:.2f}')
        print(f'BTC Held: {self.btc_held:.8f}')
        print(f'Current BTC Price: ${self.data[self.current_step]["price"]:.2f}')
        print(f'MACD: {self.data[self.current_step]["macd"]:.2f}')
        print(f'RVI: {self.data[self.current_step]["rvi"]:.2f}')