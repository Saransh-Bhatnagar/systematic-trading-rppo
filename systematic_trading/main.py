import os
from config import PARQUET_FILE

def run_pipeline():
    print("=========================================================")
    print("       SYSTEMATIC RL TRADING PIPELINE INITIALIZING     ")
    print("=========================================================")

    print("\n>>> Phase 1: Data Ingestion & Feature Engineering")
    if os.path.exists(PARQUET_FILE):
        print(f"  Data already exists at {PARQUET_FILE}. Skipping download.")
        print("  (Delete the file to force re-download.)")
    else:
        from data_ingestion import fetch_and_engineer_data
        fetch_and_engineer_data()

    print("\n>>> Phase 2: PPO Training (Continuous Action)")
    from train import train
    train()

    print("\n>>> Phase 3: Out-of-Sample Evaluation")
    from evaluate import evaluate
    evaluate()


if __name__ == "__main__":
    run_pipeline()
