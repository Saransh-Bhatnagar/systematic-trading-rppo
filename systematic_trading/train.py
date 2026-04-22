import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from environment import StockTradingEnv
from config import (
    PARQUET_FILE, MODELS_DIR, WINDOW_SIZE, TRANSACTION_COST_PCT,
    TRAIN_RATIO, TOTAL_TIMESTEPS, LEARNING_RATE, BATCH_SIZE,
    N_STEPS, N_EPOCHS, NET_ARCH, LSTM_HIDDEN_SIZE,
    CHECKPOINT_FREQ, CURRICULUM_PHASE1_FRACTION, CURRICULUM_PHASE1_TOP_N,
    GAMMA, GAE_LAMBDA, CLIP_RANGE, ENT_COEF,
    VAL_SPLIT, VAL_EVAL_FREQ, VAL_EARLY_STOP_PATIENCE
)


class SharpeValidationCallback(BaseCallback):
    """
    Custom callback that evaluates the agent on a held-out validation set
    using portfolio Sharpe ratio. Saves the best model and optionally
    stops training early if no improvement for `patience` evaluations.
    """

    def __init__(self, val_ticker_dfs, eval_freq, save_path, patience=5, verbose=1):
        super().__init__(verbose)
        self.val_ticker_dfs = val_ticker_dfs
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.patience = patience

        self.best_sharpe = -np.inf
        self.no_improve_count = 0
        self.eval_history = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        # Run deterministic episodes on each validation ticker
        all_returns = []
        for tdf in self.val_ticker_dfs:
            env = StockTradingEnv(
                tdf, window_size=WINDOW_SIZE,
                transaction_cost=TRANSACTION_COST_PCT
            )
            obs, _ = env.reset()
            lstm_states = None
            episode_start = np.ones((1,), dtype=bool)
            step_returns = []

            terminated, truncated = False, False
            while not (terminated or truncated):
                action, lstm_states = self.model.predict(
                    obs, state=lstm_states,
                    episode_start=episode_start, deterministic=True
                )
                obs, _, terminated, truncated, info = env.step(action)
                step_returns.append(info['step_return'])
                episode_start = np.zeros((1,), dtype=bool)

            all_returns.append(np.array(step_returns))

        # Compute equal-weight portfolio Sharpe
        max_len = max(len(s) for s in all_returns)
        aligned = np.full((len(all_returns), max_len), np.nan)
        for i, s in enumerate(all_returns):
            aligned[i, :len(s)] = s
        portfolio_returns = np.nanmean(aligned, axis=0)

        mean_r = np.mean(portfolio_returns)
        std_r = np.std(portfolio_returns)
        sharpe = (mean_r / std_r) * np.sqrt(252) if std_r > 1e-8 else 0.0
        total_return = (np.prod(1 + portfolio_returns) - 1) * 100

        self.eval_history.append({
            'timestep': self.num_timesteps,
            'sharpe': sharpe,
            'return_pct': total_return,
        })

        if self.verbose:
            print(f"\n  [VAL @ {self.num_timesteps:,} steps]  "
                  f"Sharpe: {sharpe:.4f}  Return: {total_return:.2f}%  "
                  f"Best: {self.best_sharpe:.4f}  Patience: "
                  f"{self.patience - self.no_improve_count}/{self.patience}")

        if sharpe > self.best_sharpe:
            self.best_sharpe = sharpe
            self.no_improve_count = 0
            best_path = os.path.join(self.save_path, "rppo_best_val")
            self.model.save(best_path)
            if self.verbose:
                print(f"  [VAL] New best model saved → {best_path}")
        else:
            self.no_improve_count += 1
            if self.no_improve_count >= self.patience:
                if self.verbose:
                    print(f"  [VAL] Early stopping triggered — no improvement "
                          f"for {self.patience} evals. Best Sharpe: {self.best_sharpe:.4f}")
                return False  # stops training

        return True


class ProgressBarCallback(BaseCallback):
    """tqdm progress bar showing steps, elapsed time, and ETA."""

    def __init__(self, total_timesteps):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.pbar: tqdm | None = None

    def _on_training_start(self):
        self.pbar = tqdm(
            total=self.total_timesteps,
            unit="step",
            desc="Training",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

    def _on_step(self) -> bool:
        self.pbar.update(self.model.n_envs)
        return True

    def _on_training_end(self):
        self.pbar.close()


def make_env(ticker_df):
    """Factory closure for vectorized environment spawning."""
    def _init():
        return StockTradingEnv(
            df=ticker_df,
            window_size=WINDOW_SIZE,
            transaction_cost=TRANSACTION_COST_PCT,
            random_start=True,
        )
    return _init


def build_env_fns(train_df, ticker_filter=None):
    """Build environment factory list, optionally filtering to specific tickers."""
    if ticker_filter is not None:
        available = set(train_df['Ticker'].unique())
        tickers = [t for t in ticker_filter if t in available]
    else:
        tickers = train_df['Ticker'].unique().tolist()

    env_fns = []
    for ticker in tickers:
        tdf = train_df[train_df['Ticker'] == ticker]
        if len(tdf) > WINDOW_SIZE + 50:
            env_fns.append(make_env(tdf))

    return env_fns, tickers


def train():
    if not os.path.exists(PARQUET_FILE):
        raise FileNotFoundError(
            f"Data missing. Run data_ingestion.py first to generate {PARQUET_FILE}"
        )

    df = pd.read_parquet(PARQUET_FILE)

    # Temporal split: training window vs OOS
    dates = sorted(df['Date'].unique())
    train_cutoff = dates[int(len(dates) * TRAIN_RATIO)]
    full_train_df = df[df['Date'] < train_cutoff]

    # Further split training window into train / validation
    train_dates = sorted(full_train_df['Date'].unique())
    val_cutoff = train_dates[int(len(train_dates) * VAL_SPLIT)]
    train_df = full_train_df[full_train_df['Date'] < val_cutoff]
    val_df = full_train_df[full_train_df['Date'] >= val_cutoff]

    print(f"Train: < {val_cutoff}  |  Val: {val_cutoff} — {train_cutoff}  |  OOS: >= {train_cutoff}")

    # Prepare validation ticker DataFrames
    val_ticker_dfs = []
    for ticker in val_df['Ticker'].unique():
        tdf = val_df[val_df['Ticker'] == ticker]
        if len(tdf) > WINDOW_SIZE + 5:
            val_ticker_dfs.append(tdf)

    print(f"Validation tickers: {len(val_ticker_dfs)}")

    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    # ── Select Phase 1 tickers by training-period momentum ──
    first_closes = train_df.groupby('Ticker')['Close'].first()
    last_closes = train_df.groupby('Ticker')['Close'].last()
    counts = train_df.groupby('Ticker').size()
    valid_tickers = counts[counts > WINDOW_SIZE + 50].index
    momentum_series = (last_closes[valid_tickers] - first_closes[valid_tickers]) / first_closes[valid_tickers]
    momentum = momentum_series.to_dict()

    top_tickers = sorted(momentum, key=momentum.get, reverse=True)[:CURRICULUM_PHASE1_TOP_N]
    print(f"\nPhase 1 tickers (top {CURRICULUM_PHASE1_TOP_N} by training momentum):")
    for t in top_tickers:
        print(f"  {t:20s}  {momentum[t]*100:+.1f}%")

    curriculum_phases = [
        {"fraction": CURRICULUM_PHASE1_FRACTION, "tickers": top_tickers},
        {"fraction": 1.0 - CURRICULUM_PHASE1_FRACTION, "tickers": None},
    ]

    # ── Curriculum Learning ──
    model = None
    best_val_sharpe = -np.inf  # persist across phases

    for phase_idx, phase in enumerate(curriculum_phases):
        phase_steps = int(TOTAL_TIMESTEPS * phase['fraction'])
        ticker_filter = phase['tickers']

        env_fns, used_tickers = build_env_fns(train_df, ticker_filter)

        if not env_fns:
            print(f"  Phase {phase_idx+1}: No valid environments, skipping.")
            continue

        vec_env = DummyVecEnv(env_fns)

        phase_label = "top momentum tickers" if ticker_filter else "full universe"
        print(f"\n{'='*60}")
        print(f"  CURRICULUM PHASE {phase_idx+1}: {phase_label}")
        print(f"  Tickers: {len(env_fns)} | Steps: {phase_steps:,}")
        print(f"{'='*60}")

        if model is None:
            # First phase: create new model
            policy_kwargs = dict(
                net_arch=NET_ARCH,
                lstm_hidden_size=LSTM_HIDDEN_SIZE,
            )

            model = RecurrentPPO(
                "MlpLstmPolicy",
                vec_env,
                learning_rate=LEARNING_RATE,
                n_steps=N_STEPS,
                batch_size=BATCH_SIZE,
                n_epochs=N_EPOCHS,
                gamma=GAMMA,
                gae_lambda=GAE_LAMBDA,
                clip_range=CLIP_RANGE,
                ent_coef=ENT_COEF,
                policy_kwargs=policy_kwargs,
                verbose=0,
                tensorboard_log="./tensorboard_logs/",
                seed=42,
            )
        else:
            # RecurrentPPO requires save+reload to change env count
            # (set_env fails when n_envs differs)
            tmp_path = os.path.join(MODELS_DIR, "_phase_transition_tmp")
            model.save(tmp_path)
            model = RecurrentPPO.load(tmp_path, env=vec_env)

        checkpoint_cb = CheckpointCallback(
            save_freq=max(CHECKPOINT_FREQ // len(env_fns), 1),
            save_path=MODELS_DIR,
            name_prefix=f"rppo_phase{phase_idx+1}",
        )

        val_cb = SharpeValidationCallback(
            val_ticker_dfs=val_ticker_dfs,
            eval_freq=max(VAL_EVAL_FREQ // len(env_fns), 1),
            save_path=MODELS_DIR,
            patience=VAL_EARLY_STOP_PATIENCE,
        )
        # Carry over best Sharpe from previous phase so we don't overwrite
        # a better Phase 1 model with a worse Phase 2 model
        val_cb.best_sharpe = best_val_sharpe

        progress_cb = ProgressBarCallback(total_timesteps=phase_steps)

        model.learn(
            total_timesteps=phase_steps,
            callback=[checkpoint_cb, val_cb, progress_cb],
            reset_num_timesteps=(phase_idx == 0),
        )

        vec_env.close()

        # Persist best Sharpe across phases
        best_val_sharpe = val_cb.best_sharpe

        # If early stopping fired, stop curriculum too
        if val_cb.no_improve_count >= val_cb.patience:
            print(f"  Early stopping in Phase {phase_idx+1} — skipping remaining phases.")
            break

    # Save final model (best validation model is already saved separately)
    model_path = os.path.join(MODELS_DIR, "rppo_trading_agent")
    model.save(model_path)

    best_path = os.path.join(MODELS_DIR, "rppo_best_val.zip")
    if os.path.exists(best_path):
        print(f"\nBest validation model: {best_path} (Sharpe: {best_val_sharpe:.4f})")
    print(f"Final model: {model_path}")
    print("Training complete.")


if __name__ == "__main__":
    train()