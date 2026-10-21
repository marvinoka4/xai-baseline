"""Everything PROTOCOL.md asks for on one fold, for one model.

fold_aware: fit, metrics, SHAP, global importance, attribution share, deletion
test and fairness. fold_unaware: fit without sex and race, metrics and fairness.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src import deletion, explain, fairness
from src.attributions import attribution_share, global_importance
from src.config import (CATEGORICAL, DELETION_RANDOM_ORDERS, DELETION_ROWS, NUMERIC,
                        RACE_MIN_GROUP, SENSITIVE, THRESHOLD)
from src.models import lightgbm, logistic

FACTORIES = {"lightgbm": lightgbm, "logistic": logistic}


def deletion_rows_seed(split_seed, fold):
    """Stated in PROTOCOL.md (Faithfulness check)."""
    return 1000 * split_seed + fold


def random_orders_seed(split_seed, fold):
    """The protocol says the random orders are seeded; this is the seed used."""
    return 3000 + 10 * split_seed + fold


def primary_race_groups(race):
    """Race groups with at least RACE_MIN_GROUP people in the full data."""
    counts = pd.Series(race).value_counts()
    return sorted(counts[counts >= RACE_MIN_GROUP].index)


def scores(y_true, prob):
    pred = (prob >= THRESHOLD).astype(int)
    return {"accuracy": accuracy_score(y_true, pred),
            "f1": f1_score(y_true, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, prob)}, pred


def fairness_rows(y_true, pred, X_test, race_primary):
    """Disparity metrics and per-group rates for sex, race (primary) and race (all)."""
    metrics, groups = [], []
    scopes = [("sex", "sex", None), ("race", "race_primary", race_primary),
              ("race", "race_all", None)]
    for attr, scope, keep in scopes:
        g = X_test[attr].astype(str).to_numpy()
        yt, yp, gg = (fairness.restrict(y_true, pred, g, keep) if keep
                      else (np.asarray(y_true), pred, g))
        metrics.append({"attribute": attr, "scope": scope, **fairness.disparities(yt, yp, gg)})
        if scope != "race_primary":
            table = fairness.group_rates(yt, yp, gg).reset_index()
            table.insert(0, "attribute", attr)
            groups.append(table.rename(columns={"index": "group"}))
    return metrics, pd.concat(groups, ignore_index=True)


def fold_aware(kind, X, y, split_seed, fold, train, test, race_primary,
               deletion_n=DELETION_ROWS, random_n=DELETION_RANDOM_ORDERS, bg_rows=None):
    X_train, X_test = X.iloc[train], X.iloc[test]
    y_train, y_test = y.iloc[train], y.iloc[test]
    pipe = FACTORIES[kind]().fit(X_train, y_train)
    prob = pipe.predict_proba(X_test)[:, 1]
    metric, pred = scores(y_test, prob)

    bg = explain.background(X_train, split_seed, fold,
                            **({"n": bg_rows} if bg_rows else {}))
    shap_values, _ = explain.feature_shap(pipe, kind, bg, X_test, NUMERIC, CATEGORICAL)
    importance = global_importance(shap_values)
    share = attribution_share(shap_values, SENSITIVE)

    rng_rows = np.random.default_rng(deletion_rows_seed(split_seed, fold))
    pick = np.sort(rng_rows.choice(len(X_test), size=min(deletion_n, len(X_test)),
                                   replace=False))
    rows = X_test.iloc[pick]
    rows_shap = shap_values.iloc[pick].to_numpy()
    replace = deletion.replacement_values(X_train, NUMERIC, CATEGORICAL)

    def predict(frame):
        return pipe.predict_proba(frame)[:, 1]

    predicted_one = predict(rows) >= THRESHOLD
    curves = {
        "signed": deletion.deletion_curves(
            predict, rows, deletion.signed_orders(rows_shap, predicted_one), replace),
        "absolute": deletion.deletion_curves(
            predict, rows, deletion.abs_orders(rows_shap), replace),
    }
    rng_orders = np.random.default_rng(random_orders_seed(split_seed, fold))
    random_curves = [deletion.deletion_curves(
        predict, rows, deletion.random_orders(len(rows), rows.shape[1], rng_orders), replace)
        for _ in range(random_n)]
    curves["random"] = np.mean(random_curves, axis=0)
    mean_curves = {name: c.mean(axis=0) for name, c in curves.items()}
    areas = {name: deletion.curve_area(c) for name, c in mean_curves.items()}

    fair, groups = fairness_rows(y_test.to_numpy(), pred, X_test, race_primary)
    return {"metrics": metric, "importance": importance, "share": share,
            "curves": mean_curves, "areas": areas, "fairness": fair, "groups": groups,
            "pred": pd.Series(pred, index=X_test.index)}


def fold_unaware(kind, X, y, train, test, race_primary):
    numeric = [c for c in NUMERIC if c not in SENSITIVE]
    categorical = [c for c in CATEGORICAL if c not in SENSITIVE]
    X_train = X.iloc[train][numeric + categorical]
    X_test_full = X.iloc[test]
    pipe = FACTORIES[kind](numeric=numeric, categorical=categorical).fit(X_train, y.iloc[train])
    prob = pipe.predict_proba(X_test_full[numeric + categorical])[:, 1]
    metric, pred = scores(y.iloc[test], prob)
    fair, groups = fairness_rows(y.iloc[test].to_numpy(), pred, X_test_full, race_primary)
    return {"metrics": metric, "fairness": fair, "groups": groups,
            "pred": pd.Series(pred, index=X_test_full.index), "inputs": numeric + categorical}
