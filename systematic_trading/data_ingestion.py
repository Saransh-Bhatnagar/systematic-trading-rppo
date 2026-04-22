import os
import pickle
import yfinance as yf
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

from ta import add_all_ta_features
from ta.trend import ema_indicator
from sklearn.decomposition import PCA
from config import (
    DATA_DIR, PARQUET_FILE, SCALER_FILE, PCA_FILE,
    TICKERS, START_DATE, END_DATE, TRAIN_RATIO, PCA_VARIANCE_RATIO
)


def fit_scaler_on_train(df, feature_cols, train_cutoff):
    """Standardize using train-only statistics to prevent data snooping."""
    train_mask = df['Date'] < train_cutoff
    scaler_params = {}

    for col in feature_cols:
        train_series = df.loc[train_mask, col]
        mean = train_series.mean()
        std = train_series.std()
        scaler_params[col] = (mean, std)

        if std > 1e-8 and not pd.isna(std):
            df[col] = (df[col] - mean) / std
        else:
            df[col] = 0.0

    return df, scaler_params


def apply_pca(df, feature_cols, train_cutoff):
    """
    Fit PCA on training data only, then transform the full dataset.
    Reduces ~85 correlated TA features to a compact orthogonal basis
    retaining PCA_VARIANCE_RATIO of explained variance.
    """
    train_mask = df['Date'] < train_cutoff

    X_train = df.loc[train_mask, feature_cols].values
    X_all = df[feature_cols].values

    pca = PCA(n_components=PCA_VARIANCE_RATIO, svd_solver='full')
    pca.fit(X_train)

    n_components = pca.n_components_
    print(f"  PCA: {len(feature_cols)} features → {n_components} components "
          f"({pca.explained_variance_ratio_.sum()*100:.1f}% variance retained)")

    X_pca = pca.transform(X_all)
    pca_col_names = [f'PCA_{i}' for i in range(n_components)]
    pca_df = pd.DataFrame(X_pca, columns=pca_col_names, index=df.index)

    # Drop old features, attach PCA components
    df = df.drop(columns=feature_cols)
    df = pd.concat([df, pca_df], axis=1)

    # Save PCA model for live inference
    with open(PCA_FILE, 'wb') as f:
        pickle.dump(pca, f)

    return df, pca_col_names


def fetch_and_engineer_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    print(f"Downloading data for {len(TICKERS)} tickers...")
    data = yf.download(TICKERS, start=START_DATE, end=END_DATE,
                       group_by='ticker', threads=True)

    # Download NIFTY 50 index for market regime features
    print("Downloading NIFTY 50 index for market regime features...")
    nifty = yf.download("^NSEI", start=START_DATE, end=END_DATE)
    nifty.columns = nifty.columns.get_level_values(0)  # flatten multi-level columns
    nifty = nifty.reset_index()
    nifty_close = nifty[['Date', 'Close']].copy()
    nifty_close.columns = ['Date', 'nifty_close']
    nifty_close['nifty_close'] = nifty_close['nifty_close'].astype(float)
    nifty_close['mkt_trend'] = (
        ema_indicator(close=nifty_close['nifty_close'], window=20, fillna=True)
        / ema_indicator(close=nifty_close['nifty_close'], window=50, fillna=True)
    )
    nifty_close['mkt_vol'] = nifty_close['nifty_close'].pct_change().rolling(20).std().fillna(0.0)
    nifty_close['mkt_ret_20d'] = nifty_close['nifty_close'].pct_change(20).fillna(0.0)
    market_features = nifty_close[['Date', 'mkt_trend', 'mkt_vol', 'mkt_ret_20d']]

    processed_dfs = []

    for ticker in TICKERS:
        try:
            if len(TICKERS) == 1:
                df = data.copy()
            else:
                df = data[ticker].copy()

            if df.empty or len(df) < 100:
                print(f"  SKIP {ticker}: not enough raw rows")
                continue

            df = df.reset_index()
            df = df.dropna(subset=['Close'])

            if df.empty or len(df) < 100:
                print(f"  SKIP {ticker}: not enough valid Close data")
                continue

            df.ffill(inplace=True)
            df.bfill(inplace=True)

            # Core Alpha signals
            df['EMA_8'] = ema_indicator(close=df['Close'], window=8, fillna=True)
            df['EMA_55'] = ema_indicator(close=df['Close'], window=55, fillna=True)

            # SOTA technical indicators
            df = add_all_ta_features(
                df, open="Open", high="High", low="Low",
                close="Close", volume="Volume", fillna=True
            )

            # Price-return and volatility features (kept outside PCA)
            df['ret_1d'] = df['Close'].pct_change(1).fillna(0.0)
            df['ret_5d'] = df['Close'].pct_change(5).fillna(0.0)
            df['ret_20d'] = df['Close'].pct_change(20).fillna(0.0)
            df['vol_20d'] = df['Close'].pct_change().rolling(20).std().fillna(0.0)

            df['Ticker'] = ticker
            processed_dfs.append(df)
            print(f"  OK   {ticker}")

        except Exception as e:
            print(f"  FAIL {ticker}: {e}")

    if not processed_dfs:
        print("No valid data processed.")
        return

    combined_df = pd.concat(processed_dfs, ignore_index=True)

    # Merge market regime features by date
    combined_df = combined_df.merge(market_features, on='Date', how='left')
    combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    combined_df.dropna(inplace=True)

    # ── Temporal cutoff ──
    dates = sorted(combined_df['Date'].unique())
    train_cutoff = dates[int(len(dates) * TRAIN_RATIO)]

    # ── Separate direct features (kept outside PCA) from TA features ──
    direct_feature_cols = ['ret_1d', 'ret_5d', 'ret_20d', 'vol_20d',
                           'mkt_trend', 'mkt_vol', 'mkt_ret_20d']
    meta_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']
    exclude_from_pca = set(meta_cols + direct_feature_cols)

    all_feature_cols = combined_df.columns.difference(meta_cols).tolist()
    pca_input_cols = [c for c in all_feature_cols if c not in exclude_from_pca]

    # Standardize ALL feature columns (both PCA-bound and direct)
    combined_df, scaler_params = fit_scaler_on_train(
        combined_df, all_feature_cols, train_cutoff
    )

    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler_params, f)

    # ── PCA dimensionality reduction (train-only fit) — only on TA features ──
    combined_df, pca_cols = apply_pca(
        combined_df, pca_input_cols, train_cutoff
    )

    print(f"\nSaving {len(combined_df)} records to Parquet...")
    combined_df.to_parquet(PARQUET_FILE, index=False)
    print(f"Data:   {PARQUET_FILE}")
    print(f"Scaler: {SCALER_FILE}")
    print(f"PCA:    {PCA_FILE}")
    print(f"Train cutoff: {train_cutoff}")
    print(f"Final feature dims: {len(pca_cols)} PCA + {len(direct_feature_cols)} direct = {len(pca_cols) + len(direct_feature_cols)}")


if __name__ == "__main__":
    fetch_and_engineer_data()
