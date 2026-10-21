"""Model pipelines with hyperparameters fixed in PROTOCOL.md (Models).

Every pipeline is fitted on a training fold only, so the encoder and scaler
learn nothing from held-out rows.
"""

import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL, MODEL_SEED, NUMERIC

LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": MODEL_SEED,
    "deterministic": True,
    "force_row_wise": True,  # required by LightGBM for deterministic results
    "n_jobs": 1,
    "verbose": -1,
}
# l1_ratio = 0 is a pure L2 penalty (the spelling scikit-learn 1.8+ asks for).
LOGREG_PARAMS = {"C": 1.0, "l1_ratio": 0.0, "solver": "lbfgs", "max_iter": 5000}


def encoder(numeric, categorical, scale_numeric):
    num = StandardScaler() if scale_numeric else "passthrough"
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([("num", num, numeric), ("cat", onehot, categorical)],
                             verbose_feature_names_out=False)


def lightgbm(numeric=NUMERIC, categorical=CATEGORICAL):
    return Pipeline([("prep", encoder(numeric, categorical, scale_numeric=False)),
                     ("model", lgb.LGBMClassifier(**LGBM_PARAMS))])


def logistic(numeric=NUMERIC, categorical=CATEGORICAL):
    return Pipeline([("prep", encoder(numeric, categorical, scale_numeric=True)),
                     ("model", LogisticRegression(**LOGREG_PARAMS))])


def majority():
    return DummyClassifier(strategy="prior")
