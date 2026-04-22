import gymnasium as gym
from gymnasium import spaces
import numpy as np


class StockTradingEnv(gym.Env):
    """
    Pure-sandbox Gymnasium environment for single-stock RL trading.

    Improvements over v1:
      - Blended reward: differential Sharpe + position-holding incentive
      - Portfolio state in obs (position, equity ratio, drawdown)
      - Borrow cost for shorts
      - Observation is single-step features (not windowed) for LSTM compatibility
    """

    metadata = {'render_modes': ['console']}

    def __init__(self, df, initial_balance=100000.0, transaction_cost=0.001,
                 window_size=20, borrow_cost_daily=0.0001, random_start=False):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.window_size = window_size
        self.borrow_cost_daily = borrow_cost_daily
        self.random_start = random_start

        # Feature columns (PCA components or raw technicals)
        self.feature_cols = [
            c for c in self.df.columns
            if c not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']
        ]
        self.features = self.df[self.feature_cols].values.astype(np.float32)
        self.closes = self.df['Close'].values.astype(np.float64)
        self.n_features = len(self.feature_cols)

        # Action: target portfolio weight [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Observation: single-step features + 3 portfolio state vars
        # Using single-step obs allows RecurrentPPO's LSTM to handle temporal memory
        obs_dim = self.n_features + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Differential Sharpe tracking
        self._ema_return = 0.0
        self._ema_return_sq = 0.0
        self._ema_decay = 0.98

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.random_start:
            max_start = max(len(self.df) // 2, self.window_size + 1)
            self.current_step = self.np_random.integers(self.window_size, max_start)
        else:
            self.current_step = self.window_size

        self.portfolio_value = self.initial_balance
        self.peak_value = self.initial_balance
        self.position = 0.0

        self._ema_return = 0.0
        self._ema_return_sq = 0.0

        return self._get_obs(), {}

    def _get_obs(self):
        # Single-step feature vector (LSTM handles history)
        step_features = self.features[self.current_step]

        # Portfolio state
        equity_ratio = np.float32(self.portfolio_value / self.initial_balance)
        drawdown = np.float32(
            (self.peak_value - self.portfolio_value) / self.peak_value
            if self.peak_value > 0 else 0.0
        )
        pos = np.float32(self.position)

        return np.concatenate([
            step_features,
            np.array([pos, equity_ratio, drawdown], dtype=np.float32)
        ])

    def step(self, action):
        target_position = np.clip(action[0], -1.0, 1.0)

        current_price = self.closes[self.current_step]
        next_price = self.closes[self.current_step + 1]

        asset_return = (next_price - current_price) / current_price

        # Transaction cost
        trade_size = abs(target_position - self.position)
        cost = trade_size * self.transaction_cost

        # Borrow cost for shorts (on the new position being held)
        borrow = abs(min(target_position, 0.0)) * self.borrow_cost_daily

        # Step P&L — use target_position so the agent earns on its new decision
        step_return = (target_position * asset_return) - cost - borrow
        self.portfolio_value = max(self.portfolio_value * (1.0 + step_return), 0.0)

        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value

        # ── Blended Reward ──
        # Component 1: Differential Sharpe Ratio (Moody & Saffell 1998)
        # Must compute DSR using PREVIOUS EMA values, then update EMAs
        A_prev = self._ema_return       # EMA of returns (before this step)
        B_prev = self._ema_return_sq    # EMA of squared returns (before this step)

        delta_A = step_return - A_prev
        delta_B = step_return ** 2 - B_prev
        variance = B_prev - A_prev ** 2

        if variance > 1e-10:
            dsr = (B_prev * delta_A - 0.5 * A_prev * delta_B) / (variance ** 1.5)
        else:
            dsr = step_return * 100.0

        # Now update EMAs for next step
        self._ema_return = self._ema_decay * A_prev + (1 - self._ema_decay) * step_return
        self._ema_return_sq = self._ema_decay * B_prev + (1 - self._ema_decay) * (step_return ** 2)

        # Component 2: Position-holding incentive
        hold_bonus = target_position * asset_return * 30.0

        # Component 3: Exposure cost — gentle nudge toward flat when signal is weak.
        exposure_cost = abs(target_position) * 0.0001

        # Component 4: Amplified transaction cost penalty
        churn_penalty = cost * 2.0

        # Component 5: Drawdown penalty — light penalty to discourage holding through losses.
        drawdown = (self.peak_value - self.portfolio_value) / self.peak_value if self.peak_value > 0 else 0.0
        dd_penalty = drawdown * abs(target_position) * 0.5

        reward = float(dsr + hold_bonus - exposure_cost - churn_penalty - dd_penalty)

        # Advance
        self.position = target_position
        self.current_step += 1

        terminated = self.current_step >= len(self.df) - 2
        truncated = False

        info = {
            'portfolio_value': self.portfolio_value,
            'step_return': step_return,
            'position': self.position,
        }

        return self._get_obs(), reward, terminated, truncated, info
