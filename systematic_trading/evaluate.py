import pandas as pd
import numpy as np
import os
import warnings

from sb3_contrib import RecurrentPPO
from environment import StockTradingEnv
from config import (
    PARQUET_FILE, MODELS_DIR, WINDOW_SIZE,
    TRANSACTION_COST_PCT, TRAIN_RATIO
)

warnings.filterwarnings('ignore')


def run_ticker_episode(model, ticker_df):
    """Run a deterministic episode on a single ticker's OOS data.
    Returns the per-step returns list, final portfolio value, and turnover.
    Handles RecurrentPPO's LSTM hidden state properly."""
    env = StockTradingEnv(
        ticker_df,
        window_size=WINDOW_SIZE,
        transaction_cost=TRANSACTION_COST_PCT
    )
    obs, _ = env.reset()
    step_returns = []
    positions = [0.0]

    # RecurrentPPO requires LSTM states to be carried across steps
    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_start,
            deterministic=True
        )
        obs, reward, terminated, truncated, info = env.step(action)
        step_returns.append(info['step_return'])
        positions.append(info['position'])
        episode_start = np.zeros((1,), dtype=bool)

    # Turnover = sum of absolute position changes / number of steps
    position_changes = np.abs(np.diff(positions))
    turnover = float(np.sum(position_changes) / len(step_returns)) if step_returns else 0.0

    return step_returns, info['portfolio_value'], turnover, positions[1:]


def compute_metrics(returns_array):
    """Compute core quant metrics from a returns array."""
    total_return = (np.prod(1 + returns_array) - 1) * 100

    mean_r = np.mean(returns_array)
    std_r = np.std(returns_array)
    sharpe = (mean_r / std_r) * np.sqrt(252) if std_r > 1e-8 else 0.0

    cumulative = np.cumprod(1 + returns_array)
    peak = np.maximum.accumulate(cumulative)
    drawdowns = (peak - cumulative) / peak
    max_dd = np.max(drawdowns) * 100 if len(drawdowns) > 0 else 0.0

    win_rate = np.sum(returns_array > 0) / len(returns_array) * 100

    return total_return, sharpe, max_dd, win_rate


def evaluate():
    best_path = os.path.join(MODELS_DIR, "rppo_best_val.zip")
    final_path = os.path.join(MODELS_DIR, "rppo_trading_agent.zip")

    if os.path.exists(best_path):
        model_path = best_path
        print(f"Using best validation model: {model_path}")
    elif os.path.exists(final_path):
        model_path = final_path
        print(f"Best validation model not found, using final model: {model_path}")
    else:
        raise FileNotFoundError(
            "No trained model found. Run train.py first."
        )

    df = pd.read_parquet(PARQUET_FILE)

    # OOS split
    dates = sorted(df['Date'].unique())
    test_cutoff = dates[int(len(dates) * TRAIN_RATIO)]
    test_df = df[df['Date'] >= test_cutoff]

    model = RecurrentPPO.load(model_path)
    tickers = test_df['Ticker'].unique()

    print(f"Evaluating on {len(tickers)} tickers (OOS from {test_cutoff})...\n")

    # Per-ticker evaluation
    ticker_results = {}

    for ticker in sorted(tickers):
        tdf = test_df[test_df['Ticker'] == ticker]
        if len(tdf) <= WINDOW_SIZE + 5:
            continue

        returns, final_pv, turnover, positions = run_ticker_episode(model, tdf)
        returns_arr = np.array(returns)
        positions_arr = np.array(positions)

        total_ret, sharpe, max_dd, win_r = compute_metrics(returns_arr)
        ticker_results[ticker] = {
            'returns': returns_arr,
            'positions': positions_arr,
            'total_return': total_ret,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'win_rate': win_r,
            'final_pv': final_pv,
            'turnover': turnover,
        }

    if not ticker_results:
        print("No tickers had enough OOS data for evaluation.")
        return

    # Per-ticker tearsheet
    print("=" * 85)
    print("  PER-TICKER TEARSHEET (Out-of-Sample)")
    print("=" * 85)
    print(f"{'Ticker':<18} {'Return %':>10} {'Sharpe':>10} {'MaxDD %':>10} {'WinRate %':>10} {'Turnover':>10}")
    print("-" * 85)

    for ticker in sorted(ticker_results.keys()):
        r = ticker_results[ticker]
        print(f"{ticker:<18} {r['total_return']:>10.2f} {r['sharpe']:>10.4f} "
              f"{r['max_dd']:>10.2f} {r['win_rate']:>10.2f} {r['turnover']:>10.4f}")

    # ── Equal-weight portfolio aggregate (NaN-aware, no truncation) ──
    tickers_ordered = list(ticker_results.keys())
    all_return_series = [ticker_results[t]['returns'] for t in tickers_ordered]
    max_len = max(len(s) for s in all_return_series)
    aligned = np.full((len(all_return_series), max_len), np.nan)
    for i, s in enumerate(all_return_series):
        aligned[i, :len(s)] = s
    portfolio_returns = np.nanmean(aligned, axis=0)

    p_total, p_sharpe, p_max_dd, p_win = compute_metrics(portfolio_returns)
    avg_turnover = np.mean([r['turnover'] for r in ticker_results.values()])

    print("\n" + "=" * 72)
    print("  EQUAL-WEIGHT PORTFOLIO TEARSHEET (Aggregated)")
    print("=" * 72)
    print(f"  Tickers evaluated      : {len(ticker_results)}")
    print(f"  Total Portfolio Return  : {p_total:.2f}%")
    print(f"  Sharpe Ratio            : {p_sharpe:.4f}")
    print(f"  Maximum Drawdown        : {p_max_dd:.2f}%")
    print(f"  Win Rate                : {p_win:.2f}%")
    print(f"  Avg Daily Turnover      : {avg_turnover:.4f}")
    print("=" * 85)

    # ── Conviction-weighted portfolio ──
    # Use inverse turnover as a confidence proxy: low turnover = high conviction.
    # This automatically underweights tickers the agent churns on.
    min_turnover_cap = 0.005  # floor to prevent division by near-zero
    inv_turnover = np.array([1.0 / max(ticker_results[t]['turnover'], min_turnover_cap)
                             for t in tickers_ordered])
    conv_weights = inv_turnover / inv_turnover.sum()

    # Weighted portfolio returns (per-step weighted average across tickers)
    conv_portfolio = np.full(max_len, np.nan)
    for step in range(max_len):
        step_returns_at_t = aligned[:, step]
        valid = ~np.isnan(step_returns_at_t)
        if valid.any():
            w = conv_weights[valid]
            w = w / w.sum()  # renormalize for active tickers at this step
            conv_portfolio[step] = np.dot(w, step_returns_at_t[valid])

    conv_portfolio = conv_portfolio[~np.isnan(conv_portfolio)]
    c_total, c_sharpe, c_max_dd, c_win = compute_metrics(conv_portfolio)

    print("\n" + "=" * 72)
    print("  CONVICTION-WEIGHTED PORTFOLIO (Inverse Turnover)")
    print("=" * 72)
    print(f"  Tickers evaluated      : {len(ticker_results)}")
    print(f"  Total Portfolio Return  : {c_total:.2f}%")
    print(f"  Sharpe Ratio            : {c_sharpe:.4f}")
    print(f"  Maximum Drawdown        : {c_max_dd:.2f}%")
    print(f"  Win Rate                : {c_win:.2f}%")
    print("=" * 85)

    # ── Risk-Parity Portfolio (momentum-ranked, sector-diversified) ──
    # The RL agent produces per-stock position signals. This layer:
    # 1. Ranks stocks daily by agent conviction (position magnitude)
    # 2. Selects top-K stocks with positive conviction
    # 3. Weights by inverse volatility (risk parity)
    # 4. Rebalances weekly to reduce turnover
    TOP_K = 30
    REBAL_FREQ = 5  # trading days between rebalances

    # Build aligned position and asset-return matrices
    # Asset returns (not agent returns — we need raw returns for portfolio construction)
    asset_returns_dict = {}
    for t in tickers_ordered:
        tdf = test_df[test_df['Ticker'] == t].reset_index(drop=True)
        closes = tdf['Close'].values
        if len(closes) > WINDOW_SIZE + 2:
            ar = np.diff(closes[WINDOW_SIZE:]) / closes[WINDOW_SIZE:-1]
            asset_returns_dict[t] = ar

    # Align all series to common length
    ar_max_len = max(len(v) for v in asset_returns_dict.values())
    n_tickers = len(tickers_ordered)

    pos_matrix = np.full((n_tickers, ar_max_len), np.nan)
    ar_matrix = np.full((n_tickers, ar_max_len), np.nan)

    for i, t in enumerate(tickers_ordered):
        p = ticker_results[t]['positions']
        pos_matrix[i, :len(p)] = p
        if t in asset_returns_dict:
            ar = asset_returns_dict[t]
            ar_matrix[i, :len(ar)] = ar

    # Rolling 20-day volatility for risk parity weighting
    vol_matrix = np.full_like(ar_matrix, np.nan)
    for i in range(n_tickers):
        for j in range(20, ar_max_len):
            window = ar_matrix[i, j-20:j]
            valid_w = window[~np.isnan(window)]
            if len(valid_w) >= 10:
                vol_matrix[i, j] = np.std(valid_w)

    # Build portfolio: rebalance every REBAL_FREQ days
    rp_returns = np.full(ar_max_len, np.nan)
    current_weights = np.zeros(n_tickers)
    last_rebal = -REBAL_FREQ  # force rebalance on first valid day

    for step in range(20, ar_max_len):
        # Rebalance check
        if step - last_rebal >= REBAL_FREQ:
            # Get agent positions at this step
            positions_today = pos_matrix[:, step]
            vols_today = vol_matrix[:, step]

            # Find tickers with valid data, positive conviction, and valid vol
            valid_mask = (~np.isnan(positions_today) &
                          ~np.isnan(vols_today) &
                          (vols_today > 1e-8))
            conviction = np.where(valid_mask, positions_today, 0.0)

            # Select top-K by conviction magnitude (positive positions only)
            candidates = np.where(conviction > 0.05, conviction, 0.0)
            if np.sum(candidates > 0) > TOP_K:
                threshold = np.sort(candidates)[-TOP_K]
                candidates[candidates < threshold] = 0.0

            # Dynamic exposure: scale total allocation by average conviction.
            # If agent is strongly long (avg position ~1.0), go ~100% invested.
            # If agent is uncertain (avg position ~0.3), hold ~70% cash.
            avg_conviction = np.mean(np.abs(candidates[candidates > 0])) if np.any(candidates > 0) else 0.0
            exposure_scale = np.clip(avg_conviction, 0.0, 1.0)

            # Risk parity: weight inversely by volatility among selected stocks
            selected = candidates > 0
            if selected.any():
                inv_vol = np.where(selected, 1.0 / vols_today, 0.0)
                inv_vol[~np.isfinite(inv_vol)] = 0.0
                total_inv_vol = inv_vol.sum()
                if total_inv_vol > 0:
                    new_weights = (inv_vol / total_inv_vol) * exposure_scale
                else:
                    new_weights = np.zeros(n_tickers)
            else:
                new_weights = np.zeros(n_tickers)

            # Transaction cost for rebalancing
            rebal_cost = np.sum(np.abs(new_weights - current_weights)) * TRANSACTION_COST_PCT
            current_weights = new_weights
            last_rebal = step
        else:
            rebal_cost = 0.0

        # Portfolio return at this step (remainder is cash @ 0% return)
        step_ar = ar_matrix[:, step]
        valid_step = ~np.isnan(step_ar)
        if valid_step.any():
            rp_returns[step] = np.dot(current_weights[valid_step], step_ar[valid_step]) - rebal_cost
        else:
            rp_returns[step] = 0.0

    rp_valid = rp_returns[~np.isnan(rp_returns)]
    if len(rp_valid) > 0:
        rp_total, rp_sharpe, rp_mdd, rp_win = compute_metrics(rp_valid)

        # Count rebalance events and avg active positions
        print("\n" + "=" * 72)
        print("  RISK-PARITY PORTFOLIO (Top-K by Agent Conviction)")
        print("=" * 72)
        print(f"  Max positions held     : {TOP_K}")
        print(f"  Rebalance frequency    : every {REBAL_FREQ} trading days")
        print(f"  Total Portfolio Return  : {rp_total:.2f}%")
        print(f"  Sharpe Ratio            : {rp_sharpe:.4f}")
        print(f"  Maximum Drawdown        : {rp_mdd:.2f}%")
        print(f"  Win Rate                : {rp_win:.2f}%")
        print("=" * 85)

    # Buy-and-Hold benchmark (with entry/exit transaction cost)
    bh_returns_list = []
    for ticker in ticker_results:
        tdf = test_df[test_df['Ticker'] == ticker].reset_index(drop=True)
        closes = tdf['Close'].values
        if len(closes) > WINDOW_SIZE + 2:
            daily_ret = np.diff(closes[WINDOW_SIZE:]) / closes[WINDOW_SIZE:-1]
            # Deduct entry cost on first day, exit cost on last day
            daily_ret[0] -= TRANSACTION_COST_PCT
            daily_ret[-1] -= TRANSACTION_COST_PCT
            bh_returns_list.append(daily_ret)

    if bh_returns_list:
        max_bh = max(len(s) for s in bh_returns_list)
        bh_aligned = np.full((len(bh_returns_list), max_bh), np.nan)
        for i, s in enumerate(bh_returns_list):
            bh_aligned[i, :len(s)] = s
        bh_portfolio = np.nanmean(bh_aligned, axis=0)
        bh_total, bh_sharpe, bh_mdd, bh_win = compute_metrics(bh_portfolio)

        print(f"\n  BENCHMARK (Equal-Weight Buy & Hold)")
        print(f"  {'─' * 40}")
        print(f"  Total Return            : {bh_total:.2f}%")
        print(f"  Sharpe Ratio            : {bh_sharpe:.4f}")
        print(f"  Maximum Drawdown        : {bh_mdd:.2f}%")
        print(f"  Win Rate                : {bh_win:.2f}%")
        print("=" * 85)


if __name__ == "__main__":
    evaluate()
