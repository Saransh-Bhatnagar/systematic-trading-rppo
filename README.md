# systematic-trading-rppo

A reinforcement learning pipeline that trains a **Recurrent PPO (LSTM)** agent to trade individual Indian equities (NIFTY universe, ~140 tickers) with long/short capability, transaction and borrow costs, and a risk-adjusted reward.

## Highlights

- **Agent** — `sb3-contrib` `RecurrentPPO` with an MLP-LSTM policy (128 hidden units, `[128, 128]` net arch). Continuous action in `[-1, 1]` representing target portfolio weight (full short ↔ full long).
- **Reward** — blended: Moody–Saffell Differential Sharpe Ratio + position-return bonus − exposure / churn / drawdown penalties. Encourages risk-adjusted alpha, not raw PnL.
- **Features** — `ta` library's ~85 technical indicators per ticker, plus direct return/vol features and market-regime features from the NIFTY 50 index (`mkt_trend`, `mkt_vol`, `mkt_ret_20d`). TA block is reduced via **PCA (95% variance retained)**; scaler and PCA are fit on the training window only to prevent look-ahead bias.
- **Curriculum learning** — Phase 1 (40% of timesteps) trains on the top-15 tickers by in-sample momentum; Phase 2 (60%) on the full universe. Model is saved and reloaded across the phase boundary so `n_envs` can change.
- **Validation & early stopping** — 80/20 train/OOS split, then 90/10 train/val inside the training window. A custom `SharpeValidationCallback` runs deterministic rollouts on the val window every 500k steps, saves the best model by portfolio Sharpe, and early-stops after 5 evals without improvement (persists best Sharpe across phases).
- **Environment** — single-stock `gymnasium.Env` with transaction cost (10 bps), daily borrow cost on shorts, portfolio-state features in obs (position, equity ratio, drawdown), and `random_start=True` for training diversity.

## Project layout

```
systematic_trading/
  main.py            # pipeline entrypoint: ingest → train → evaluate
  config.py          # tickers, dates, hyperparameters
  data_ingestion.py  # yfinance download + TA features + PCA + scaler
  environment.py     # StockTradingEnv (Gymnasium)
  train.py           # RecurrentPPO training, curriculum + val callback
  evaluate.py        # OOS rollout & metrics
  report.py          # performance report generation
Stock_Strategies_DQN.ipynb   # exploratory DQN notebook (precursor work)
environment.yml              # conda env spec
```

## Setup

```bash
conda env create -f environment.yml
conda activate quant_trading
```

## Run

```bash
cd systematic_trading
python main.py
```

The pipeline:
1. Downloads prices (2019–present) for ~140 NIFTY tickers + `^NSEI` index via `yfinance`, engineers features, fits scaler + PCA on the training window, writes `data/nifty150_features.parquet`.
2. Trains the RecurrentPPO agent (5M timesteps by default) with curriculum learning, checkpoints every 250k steps to `models/`, and TensorBoard logs to `../tensorboard_logs/`.
3. Loads `models/rppo_best_val.zip` and runs deterministic OOS rollouts; reports total return, Sharpe, max drawdown, win rate, and turnover per ticker and at the equal-weight portfolio level.

Monitor training:

```bash
tensorboard --logdir tensorboard_logs
```

## Key hyperparameters (`config.py`)

| | |
|---|---|
| Tickers | ~140 NSE symbols across 10 sectors |
| Date range | 2019-01-01 → latest |
| Train / OOS split | 80 / 20 |
| Train / Val split (within train) | 90 / 10 |
| Total timesteps | 5,000,000 |
| Learning rate | 5e-5 |
| γ / λ | 0.995 / 0.98 |
| Clip range | 0.1 |
| Entropy coef | 0.01 |
| Transaction cost | 10 bps |
| Borrow cost (daily) | 1 bp on shorts |

## Notes

- Model checkpoints (`systematic_trading/models/`), TensorBoard logs (`tensorboard_logs/`), and cached feature artifacts (`data/*.parquet`, `data/*.pkl`) are gitignored — regenerate by running the pipeline.
- The DQN notebook is earlier exploratory work and is not wired into the main pipeline.
- Not investment advice. Research code.