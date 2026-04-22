import os

# Project structure base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

PARQUET_FILE = os.path.join(DATA_DIR, "nifty150_features.parquet")
SCALER_FILE = os.path.join(DATA_DIR, "train_scaler.pkl")
PCA_FILE = os.path.join(DATA_DIR, "pca_model.pkl")

# Expanded NIFTY universe — verified active NSE tickers
TICKERS = [
    # Banks & Financial Services
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
    "INDUSINDBK.NS", "BANKBARODA.NS", "PNB.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS",
    "BAJFINANCE.NS", "BAJAJFINSV.NS", "CHOLAFIN.NS", "MUTHOOTFIN.NS", "MANAPPURAM.NS",
    "SBICARD.NS", "HDFCLIFE.NS", "ICICIPRULI.NS", "SBILIFE.NS",
    # IT & Technology
    "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
    "LTIM.NS", "MPHASIS.NS", "COFORGE.NS", "PERSISTENT.NS", "LTTS.NS",
    # Oil, Gas & Energy
    "RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS", "GAIL.NS",
    "NTPC.NS", "POWERGRID.NS", "ADANIGREEN.NS", "TATAPOWER.NS", "NHPC.NS",
    # Automobile & Auto Ancillary
    "MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS",
    "EICHERMOT.NS", "ASHOKLEY.NS", "MOTHERSON.NS", "BHARATFORG.NS", "MRF.NS",
    # FMCG & Consumer
    "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS",
    "MARICO.NS", "GODREJCP.NS", "COLPAL.NS", "TATACONSUM.NS", "VBL.NS",
    # Pharma & Healthcare
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS",
    "FORTIS.NS", "LAURUSLABS.NS", "BIOCON.NS", "AUROPHARMA.NS", "LUPIN.NS",
    # Metals & Mining
    "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "COALINDIA.NS",
    "NMDC.NS", "SAIL.NS", "NATIONALUM.NS",
    # Infrastructure & Construction
    "LT.NS", "ADANIENT.NS", "ADANIPORTS.NS", "DLF.NS", "GODREJPROP.NS",
    "ULTRACEMCO.NS", "SHREECEM.NS", "AMBUJACEM.NS", "ACC.NS", "GRASIM.NS",
    # Telecom & Media
    "BHARTIARTL.NS", "IDEA.NS", "ZEEL.NS",
    # Chemicals
    "PIDILITIND.NS", "SRF.NS", "ATUL.NS", "NAVINFLUOR.NS", "DEEPAKNTR.NS",
    # Diversified / Others
    "TITAN.NS", "ASIANPAINT.NS", "BERGEPAINT.NS", "HAVELLS.NS", "VOLTAS.NS",
    "SIEMENS.NS", "ABB.NS", "CUMMINSIND.NS", "BEL.NS", "HAL.NS",
    "IRCTC.NS", "INDHOTEL.NS", "PAGEIND.NS", "TRENT.NS", "PIIND.NS",
    "DMART.NS", "NAUKRI.NS", "POLICYBZR.NS", "PAYTM.NS", "ETERNAL.NS",
    "JSWENERGY.NS", "TORNTPOWER.NS", "INDIGO.NS", "MAXHEALTH.NS",
    "OBEROIRLTY.NS", "PRESTIGE.NS", "PHOENIXLTD.NS",
    "MFSL.NS", "CANFINHOME.NS", "RECLTD.NS", "PFC.NS",
    "GLAND.NS", "IPCALAB.NS", "ALKEM.NS",
    "CONCOR.NS", "POLYCAB.NS", "ASTRAL.NS", "SUPREMEIND.NS",
    "TATACOMM.NS", "STARHEALTH.NS", "UPL.NS", "ESCORTS.NS",
]

# Lookback
START_DATE = "2019-01-01"
END_DATE = None  # None = latest available data (today)

TRAIN_RATIO = 0.8

# Environment
WINDOW_SIZE = 20
INITIAL_BALANCE = 100000.0
TRANSACTION_COST_PCT = 0.001

# PCA — retain 95% of variance, compressing ~85 TA columns
PCA_VARIANCE_RATIO = 0.95

# Training
TOTAL_TIMESTEPS = 5_000_000
LEARNING_RATE = 5e-5
BATCH_SIZE = 512
N_STEPS = 512
N_EPOCHS = 5
NET_ARCH = [128, 128]
LSTM_HIDDEN_SIZE = 128
CHECKPOINT_FREQ = 250_000

# PPO tuning
GAMMA = 0.995            # longer discount horizon (~1yr effective) for daily trading
GAE_LAMBDA = 0.98        # longer-horizon credit assignment
CLIP_RANGE = 0.1         # more conservative policy updates for noisy financial data
ENT_COEF = 0.01          # encourage exploration, prevent early convergence to flat

# Curriculum learning — Phase 1 tickers selected dynamically by training-period momentum
CURRICULUM_PHASE1_FRACTION = 0.4   # 40% of timesteps on top momentum tickers
CURRICULUM_PHASE1_TOP_N = 15       # number of tickers to select for Phase 1

# Validation & early stopping
VAL_SPLIT = 0.9                    # 90% train / 10% validation (within training window)
VAL_EVAL_FREQ = 500_000            # evaluate on validation set every N timesteps
VAL_EARLY_STOP_PATIENCE = 5        # stop if no improvement for N consecutive evals
