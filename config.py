import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

N_SAMPLE_USERS = 5000
RANDOM_SEED = 42
WATCHED_THRESHOLD = 0.5

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

MANUAL_TIME_WINDOWS = [7, 30, 90]

AUTO_TIME_WINDOWS = [1, 3, 7, 14, 30, 60, 90, 180, 365]
AUTO_AGGREGATIONS = ["count", "mean", "std", "max", "min"]

TOP_N_AUTO_FEATURES = 50

CATBOOST_PARAMS = {
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 6,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "random_seed": RANDOM_SEED,
    "verbose": 100,
    "early_stopping_rounds": 50,
}

RANKING_K = [5, 10, 20]
