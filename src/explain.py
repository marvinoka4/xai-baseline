"""SHAP for both models, as fixed in PROTOCOL.md (Explanations).

Both explainers are interventional and use the same background: 200 rows
sampled from the training fold, encoded by that fold's fitted preprocessing.
Values are on the log-odds scale and are summed back from one-hot columns to
the original features.
"""

import numpy as np
import shap

from src.attributions import sum_to_features
from src.config import BACKGROUND_ROWS


def background_seed(split_seed, fold):
    return 2000 + 10 * split_seed + fold


def background(X_train, split_seed, fold, n=BACKGROUND_ROWS):
    """n rows sampled without replacement from the training fold only."""
    rng = np.random.default_rng(background_seed(split_seed, fold))
    positions = rng.choice(len(X_train), size=min(n, len(X_train)), replace=False)
    return X_train.iloc[np.sort(positions)]


def explainer(pipe, kind, background_rows):
    """Interventional explainer for a fitted pipeline on its encoded background."""
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    bg = prep.transform(background_rows)
    # An explicit masker sized to the whole background: passing a bare array
    # lets SHAP subsample it to 100 rows.
    masker = shap.maskers.Independent(bg, max_samples=len(bg))
    if kind == "lightgbm":
        return shap.TreeExplainer(model, data=masker, feature_perturbation="interventional",
                                  model_output="raw")
    if kind == "logistic":
        return shap.LinearExplainer(model, masker)
    raise ValueError(f"Unknown model kind {kind!r}")


def background_size(exp):
    """Number of background rows an explainer actually uses."""
    data = getattr(exp, "data", None)
    if data is None and hasattr(exp, "masker"):
        data = exp.masker.data
    return int(np.asarray(data).shape[0])


def feature_shap(pipe, kind, background_rows, rows, numeric, categorical):
    """SHAP values per original feature for rows, and the explainer's expected value."""
    prep = pipe.named_steps["prep"]
    exp = explainer(pipe, kind, background_rows)
    encoded = prep.transform(rows)
    values = exp.shap_values(encoded)
    if isinstance(values, list):  # older shap returns one array per class
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    expected = np.ravel(exp.expected_value)[-1]
    summed = sum_to_features(values, prep.get_feature_names_out(), numeric, categorical)
    summed.index = rows.index
    return summed, float(expected)
