# systematic-trading-rppo

Reinforcement learning pipeline for systematic equity trading on NIFTY constituents, using Recurrent PPO (and a PPO baseline) over an engineered feature set.

## Project layout

- `systematic_trading/` — end-to-end training/evaluation pipeline
  - `main.py` — pipeline entrypoint (ingest → train → evaluate)
  - `config.py` — paths, hyperparameters, feature config
  - `data_ingestion.py` — fetches prices and builds the feature parquet
  - `environment.py` — Gymnasium trading environment
  - `train.py` — PPO / RecurrentPPO training with phased curriculum
  - `evaluate.py` — out-of-sample evaluation
  - `report.py` — performance reporting
- `Stock_Strategies_DQN.ipynb` — exploratory DQN notebook
- `environment.yml` — conda environment spec

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

This will (1) fetch data and engineer features into `data/nifty150_features.parquet`, (2) train the agent, writing checkpoints to `models/` and TensorBoard logs to `../tensorboard_logs/`, and (3) evaluate on the held-out window.

## Notes

Trained model checkpoints, TensorBoard logs, and the cached parquet/PCA/scaler artifacts are gitignored — regenerate them by running the pipeline.