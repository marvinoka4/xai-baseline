"""Attribution helpers, as fixed in PROTOCOL.md (Explanations, Stability, Attribution share)."""

import itertools

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def column_owner(encoded_names, numeric, categorical):
    """Map each encoded column to its original feature.

    One-hot columns are named "<feature>_<category>" by scikit-learn. Longer
    feature names are matched first so that a prefix of another feature name
    cannot claim its columns.
    """
    owners = []
    by_length = sorted(categorical, key=len, reverse=True)
    for name in encoded_names:
        if name in numeric:
            owners.append(name)
            continue
        match = next((f for f in by_length if name.startswith(f + "_")), None)
        if match is None:
            raise ValueError(f"Cannot map encoded column {name!r} to a feature")
        owners.append(match)
    return owners


def sum_to_features(values, encoded_names, numeric, categorical):
    """Sum per-column attributions (rows x encoded columns) back to original features."""
    owners = column_owner(encoded_names, numeric, categorical)
    frame = pd.DataFrame(np.asarray(values), columns=encoded_names)
    summed = frame.T.groupby(pd.Index(owners)).sum().T
    return summed[list(numeric) + list(categorical)]


def global_importance(feature_shap):
    """Mean absolute SHAP per feature over rows."""
    return feature_shap.abs().mean(axis=0)


def ranking_agreement(importances, top=5):
    """Kendall's tau-b and top-k overlap for every pair of importance vectors."""
    taus, overlaps = [], []
    for a, b in itertools.combinations(importances, 2):
        a, b = a.sort_index(), b.sort_index()
        taus.append(kendalltau(a.values, b.values).statistic)
        top_a, top_b = set(a.nlargest(top).index), set(b.nlargest(top).index)
        overlaps.append(len(top_a & top_b) / top)
    return np.array(taus), np.array(overlaps)


def attribution_share(feature_shap, features):
    """Share of total absolute SHAP falling on the given features."""
    total = feature_shap.abs().to_numpy().sum()
    return {f: float(feature_shap[f].abs().sum() / total) for f in features}
