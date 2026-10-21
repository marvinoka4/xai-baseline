"""Deletion test, as fixed in PROTOCOL.md (Faithfulness check)."""

import numpy as np
import pandas as pd


def replacement_values(train, numeric, categorical):
    """Training-fold median for numeric features, most frequent value for categorical ones."""
    values = {c: train[c].median() for c in numeric}
    values.update({c: train[c].mode(dropna=True).iloc[0] for c in categorical})
    return values


def deletion_curves(predict_proba, rows, orders, replace):
    """Confidence curves as features are removed in the given orders.

    predict_proba: function mapping a DataFrame to P(class 1) for each row.
    rows: DataFrame of n rows. orders: array (n, F) of column positions, the
    order in which each row's features are removed. replace: dict of
    replacement values by column name.

    Returns an array (n, F + 1): the probability of the class predicted with
    nothing removed, after removing 0, 1, ..., F features.
    """
    rows = rows.reset_index(drop=True)
    n, n_feat = orders.shape
    p0 = np.asarray(predict_proba(rows), dtype=float)
    predicted_one = p0 >= 0.5
    steps = [rows.copy()]
    current = rows.copy()
    columns = list(rows.columns)
    for step in range(n_feat):
        for col_pos in np.unique(orders[:, step]):
            mask = orders[:, step] == col_pos
            col = columns[col_pos]
            current.loc[mask, col] = replace[col]
        steps.append(current.copy())
    stacked = pd.concat(steps, ignore_index=True)
    for col in columns:
        if isinstance(rows[col].dtype, pd.CategoricalDtype):
            stacked[col] = pd.Categorical(stacked[col], categories=rows[col].cat.categories)
    probs = np.asarray(predict_proba(stacked), dtype=float).reshape(n_feat + 1, n).T
    return np.where(predicted_one[:, None], probs, 1 - probs)


def curve_area(curve):
    """Trapezoidal area under a mean curve, with removed features scaled to [0, 1]."""
    curve = np.asarray(curve, dtype=float)
    x = np.linspace(0, 1, len(curve))
    return float(np.sum((curve[1:] + curve[:-1]) / 2 * np.diff(x)))


def signed_orders(feature_shap, predicted_one):
    """Primary order: each feature's signed SHAP contribution toward the class the
    model predicted with nothing removed, largest first.

    feature_shap is on the log-odds scale of class 1, so the contribution toward
    class 1 is the SHAP value and toward class 0 its negative. Features that push
    away from the predicted class come last. Ties are broken by column position.
    """
    values = np.asarray(feature_shap, dtype=float)
    toward = np.where(np.asarray(predicted_one)[:, None], values, -values)
    return np.argsort(-toward, axis=1, kind="stable")


def abs_orders(feature_shap):
    """Secondary order: decreasing absolute SHAP (ties broken by column position)."""
    values = np.abs(np.asarray(feature_shap, dtype=float))
    return np.argsort(-values, axis=1, kind="stable")


def random_orders(n_rows, n_feat, rng):
    return np.array([rng.permutation(n_feat) for _ in range(n_rows)])
