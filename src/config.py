"""Shared constants. Every value here is fixed in PROTOCOL.md; change it there first."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
SNAPSHOT_MD = DATA / "SNAPSHOT.md"
DATASET = DATA / "adult.csv.gz"
RESULTS = ROOT / "results"

SOURCE_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
RAW_ZIP = DATA_RAW / "adult.zip"
EXPECTED_ROWS = 48842

# Columns in the order of the UCI files.
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income",
]
TARGET = "income_gt_50k"
DROPPED = ["fnlwgt", "education"]
NUMERIC = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL = ["workclass", "marital_status", "occupation", "relationship", "race", "sex",
               "native_country"]
FEATURES = NUMERIC + CATEGORICAL
SENSITIVE = ["sex", "race"]
MISSING_LABEL = "Unknown"

# Evaluation: stratified 5-fold CV, repeated over 5 split seeds (25 folds).
N_SPLITS = 5
SPLIT_SEEDS = [0, 1, 2, 3, 4]
MODEL_SEED = 42
THRESHOLD = 0.5

# Race groups with at least this many people in the full data enter the
# primary race disparity metrics; smaller groups are reported descriptively.
RACE_MIN_GROUP = 1000

# SHAP background: rows sampled from the training fold, shared by both explainers.
BACKGROUND_ROWS = 200

# Deletion test.
DELETION_ROWS = 500
DELETION_RANDOM_ORDERS = 5
