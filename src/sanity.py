"""Sanity outputs: class balance, group sizes and the majority-class baseline.

No model beyond the majority-class baseline is fitted here.
Writes results/class_balance.csv, results/group_sizes.csv,
results/majority_baseline_folds.csv and results/majority_baseline.csv.

Usage: python -m src.sanity
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.config import RACE_MIN_GROUP, RESULTS, THRESHOLD
from src.cv import folds
from src.data import load
from src.models import majority


def main():
    X, y = load()
    RESULTS.mkdir(parents=True, exist_ok=True)

    balance = pd.DataFrame({"label": ["income <=50K", "income >50K"],
                            "n": [int((y == 0).sum()), int((y == 1).sum())]})
    balance["share"] = (balance["n"] / len(y)).round(4)
    balance.to_csv(RESULTS / "class_balance.csv", index=False, lineterminator="\n")

    rows = []
    for attr in ["sex", "race"]:
        for group, idx in X.groupby(attr, observed=True).groups.items():
            n = len(idx)
            pos = int(y.loc[idx].sum())
            rows.append({"attribute": attr, "group": group, "n": n, "positives": pos,
                         "positive_rate": round(pos / n, 4),
                         "in_primary_race_metrics": (n >= RACE_MIN_GROUP) if attr == "race"
                         else ""})
    sizes = pd.DataFrame(rows).sort_values(["attribute", "n"], ascending=[True, False])
    sizes.to_csv(RESULTS / "group_sizes.csv", index=False, lineterminator="\n")

    fold_rows = []
    for seed, k, train, test in folds(y):
        model = majority().fit(X.iloc[train], y.iloc[train])
        score = model.predict_proba(X.iloc[test])[:, 1]
        pred = (score >= THRESHOLD).astype(int)
        fold_rows.append({"seed": seed, "fold": k, "n_test": len(test),
                          "accuracy": accuracy_score(y.iloc[test], pred),
                          "f1": f1_score(y.iloc[test], pred, zero_division=0),
                          "roc_auc": roc_auc_score(y.iloc[test], score)})
    per_fold = pd.DataFrame(fold_rows)
    per_fold.round(6).to_csv(RESULTS / "majority_baseline_folds.csv", index=False,
                             lineterminator="\n")
    summary = pd.DataFrame({
        "metric": ["accuracy", "f1", "roc_auc"],
        "mean": [per_fold[m].mean() for m in ["accuracy", "f1", "roc_auc"]],
        "sd": [per_fold[m].std(ddof=1) for m in ["accuracy", "f1", "roc_auc"]],
        "n_folds": len(per_fold),
    }).round(6)
    summary.to_csv(RESULTS / "majority_baseline.csv", index=False, lineterminator="\n")

    print(balance.to_string(index=False))
    print(sizes.to_string(index=False))
    print(summary.to_string(index=False))
    assert np.all(per_fold["n_test"].groupby(per_fold["seed"]).sum() == len(y))


if __name__ == "__main__":
    main()
