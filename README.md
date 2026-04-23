# systematic-trading-rppo

A reinforcement learning pipeline that trains a **Recurrent PPO (LSTM)** agent to trade individual Indian equities (NIFTY universe, 136 tickers across 12 sectors) with long/short capability, transaction and borrow costs, and a risk-adjusted reward.

## Background

This repo contains the RecurrentPPO pipeline. Earlier exploratory work using a DQN formed the basis of a co-authored paper at the **15th ICCCNT (2024)**. Revisiting that work later with more experience, I identified methodological limitations — discrete actions, no temporal memory, last-checkpoint model selection, train/test leakage in normalization — that motivated this rewrite with continuous actions, LSTM temporal modelling, walk-forward validation, and a risk-adjusted reward. The DQN notebook is not part of the current pipeline.

## Highlights

- **Agent** — `sb3-contrib` `RecurrentPPO` with an MLP-LSTM policy (128 hidden units, `[128, 128]` net arch). Continuous action in `[-1, 1]` representing target portfolio weight (full short ↔ full long).
- **Reward** — blended: Moody–Saffell Differential Sharpe Ratio + position-return bonus − exposure / churn / drawdown penalties. Encourages risk-adjusted alpha, not raw PnL.
- **Features** — ~88 technical indicators per ticker (`ta` library), plus direct return/vol features and market-regime features from the NIFTY 50 index (`mkt_trend`, `mkt_vol`, `mkt_ret_20d`). TA block is reduced via **PCA (~88 → ~28 components, 95% variance retained)**; scaler and PCA are fit on the training window only to prevent look-ahead bias.
- **Curriculum learning** — Phase 1 (40% of timesteps) trains on the top-15 tickers by in-sample momentum; Phase 2 (60%) on the full universe. Model is saved and reloaded across the phase boundary so `n_envs` can change.
- **Validation & early stopping** — 80/20 train/OOS split, then 90/10 train/val inside the training window. A custom `SharpeValidationCallback` runs deterministic rollouts on the val window every 500k steps, saves the best model by portfolio Sharpe, and early-stops after 5 evals without improvement (persists best Sharpe across phases).
- **Environment** — single-stock `gymnasium.Env` with transaction cost (10 bps), daily borrow cost on shorts (1 bp/day), portfolio-state features in obs (position, equity ratio, drawdown), and `random_start=True` for training diversity.

## Results — out-of-sample across two market regimes

Evaluated with deterministic rollouts of the best-validation checkpoint, equal-weight across the 136-ticker universe, separate training runs with matched train/OOS splits per regime.

### Bull regime (OOS: Dec 2022 – Oct 2024, ~22 months)

| Metric | RL Agent | Buy & Hold | Δ |
|---|---:|---:|---:|
| Total return | 44.78% | 48.36% | −3.58% |
| **Sharpe ratio** | **4.3056** | 4.2469 | +0.0587 |
| **Max drawdown** | **6.05%** | 6.39% | −0.34% |
| Win rate | 67.56% | 66.81% | +0.75% |
| Avg daily turnover | 0.0100 | — | — |

Captures 92.6% of buy-and-hold upside with a better Sharpe and tighter drawdown. 128 of 136 tickers finished positive. Turnover of 0.01 means the agent holds — it isn't churning for cost.

### Bear regime (OOS: Oct 2024 – Mar 2026, ~17 months)

| Metric | RL Agent | Buy & Hold | Δ |
|---|---:|---:|---:|
| Total return | −1.03% | −1.06% | +0.03% |
| Sharpe ratio | 0.0186 | 0.0223 | −0.0037 |
| Max drawdown | 14.95% | 15.38% | −0.43% |
| Win rate | 54.33% | 54.76% | −0.43% |
| Avg daily turnover | 0.0065 | — | — |

The agent tracks the benchmark without destroying capital and keeps a consistent drawdown advantage. It does **not** short into the downturn — see limitations.

### Interpretation

Across both regimes the agent keeps a consistent drawdown edge (−0.34% in the bull, −0.43% in the bear). Performance is market-neutral in risk terms: it doesn't amplify losses, but in the bear regime it also doesn't produce alpha from the short side.

## Sector-level snapshot (bear-regime OOS)

| Sector | Avg return | Avg Sharpe | Win/Total |
|---|---:|---:|---:|
| Metals & Mining | +37.19% | 0.87 | 7/7 |
| Auto | +11.76% | 0.39 | 7/10 |
| Banks & Financials | +8.75% | 0.29 | 14/24 |
| Pharma & Healthcare | +5.99% | 0.19 | 7/15 |
| Energy & Power | −5.99% | 0.01 | 4/13 |
| FMCG & Consumer | −8.86% | −0.21 | 5/13 |
| Infra & Realty | −18.36% | −0.41 | 4/14 |
| IT & Technology | −29.21% | −0.82 | 0/10 |

IT underperformance was broad-based (0/10) and is a known soft spot — see limitations on uniform conviction.

## Known limitations

- **Long bias from training data.** The 2019+ training window is dominated by a rising NIFTY, so the policy leans long. The bear-regime OOS confirms it: the agent tracks the benchmark down instead of shorting into the drawdown.
- **Single-stock episodes.** Each environment instance trades one ticker in isolation. The "portfolio" is stitched together at evaluation — the agent never learns cross-sectional allocation, correlation, or capital-constraint trade-offs.
- **No regime switching.** The policy cannot dynamically shift between bull and bear strategies; it applies one learned behaviour to both. A regime classifier or ensemble of regime-specific agents would address this.
- **Uniform conviction.** Position signals lack differentiation across tickers — sector-level misses (e.g. IT: 0/10 in the bear regime) suggest the agent under-discriminates when a whole sector turns.
- **Transaction-cost optimism.** A flat 10 bps cost and 1 bp/day borrow cost are a simplification. Real execution incurs slippage, market impact, liquidity constraints on smaller-cap names in the universe, and intraday spread that this backtest does not model.
- **No execution layer.** Actions are target weights applied at the next bar's close. There's no order sizing against available liquidity, no partial fills, no latency, and the continuous action assumes perfect fractional share execution.
- **Survivorship bias in the ticker list.** The universe is the current NIFTY composition; tickers delisted or replaced during the training window are absent, which biases results upward.

### Future directions

Multi-asset environment with cross-sectional observation/action space · walk-forward CV across more regimes · intraday data for microstructure edge · integration with a portfolio optimization layer (risk parity / mean-variance) · ensemble of regime-specific agents.

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
environment.yml      # conda env spec
```

## Key hyperparameters (`config.py`)

| | |
|---|---|
| Tickers | 136 NSE symbols across 12 sectors |
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
1. Downloads prices (2019–present) for 136 NIFTY tickers + `^NSEI` index via `yfinance`, engineers features, fits scaler + PCA on the training window, writes `data/nifty150_features.parquet`. To force a re-fetch (e.g. after changing tickers or dates), delete the parquet file first.
2. Trains the RecurrentPPO agent (5M timesteps by default) with curriculum learning, checkpoints every 250k steps to `models/`, and TensorBoard logs to `../tensorboard_logs/`.
3. Loads `models/rppo_best_val.zip` and runs deterministic OOS rollouts; `report.py` produces the executive report with per-ticker, per-sector, and portfolio-level metrics.

Monitor training:

```bash
tensorboard --logdir tensorboard_logs
```

## Notes

- Model checkpoints (`systematic_trading/models/`), TensorBoard logs (`tensorboard_logs/`), and cached feature artifacts (`data/*.parquet`, `data/*.pkl`) are gitignored — regenerate by running the pipeline.
- Not investment advice. Research code.