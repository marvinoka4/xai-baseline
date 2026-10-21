"""Group fairness metrics, as fixed in PROTOCOL.md (Fairness)."""

import numpy as np
import pandas as pd
from fairlearn.metrics import (MetricFrame, count, demographic_parity_difference,
                               equalized_odds_difference, false_positive_rate,
                               selection_rate, true_positive_rate)


def disparities(y_true, y_pred, groups):
    """Demographic parity difference and equalized odds difference across groups."""
    y_true, y_pred, groups = np.asarray(y_true), np.asarray(y_pred), np.asarray(groups)
    return {
        "demographic_parity_difference": float(demographic_parity_difference(
            y_true, y_pred, sensitive_features=groups)),
        "equalized_odds_difference": float(equalized_odds_difference(
            y_true, y_pred, sensitive_features=groups)),
    }


def group_rates(y_true, y_pred, groups):
    """Selection rate, true and false positive rates, rows and positives for each group."""
    frame = MetricFrame(
        metrics={"selection_rate": selection_rate, "true_positive_rate": true_positive_rate,
                 "false_positive_rate": false_positive_rate, "n": count},
        y_true=np.asarray(y_true), y_pred=np.asarray(y_pred),
        sensitive_features=pd.Series(np.asarray(groups), name="group"))
    table = frame.by_group.copy()
    table["positives"] = pd.Series(np.asarray(y_true)).groupby(np.asarray(groups)).sum()
    table["n"] = table["n"].astype(int)
    return table


def restrict(y_true, y_pred, groups, keep):
    """Rows whose group is in keep (used for the primary race metrics)."""
    mask = np.isin(np.asarray(groups), list(keep))
    return np.asarray(y_true)[mask], np.asarray(y_pred)[mask], np.asarray(groups)[mask]
