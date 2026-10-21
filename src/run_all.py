"""Run the full analysis in PROTOCOL.md and write results/.

Usage: python -m src.run_all

Writes small CSVs under results/ and figures under results/figures/. Timing
goes only to results/run_log.md, so every other output is byte-reproducible.
"""

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src import figures, fairness
from src.attributions import ranking_agreement
from src.config import RESULTS, SENSITIVE, THRESHOLD
from src.cv import folds
from src.data import load
from src.models import majority
from src.pipeline import fold_aware, fold_unaware, primary_race_groups
from src.runlog import append, logged

KINDS = ["lightgbm", "logistic"]
ROUND = 6


def save(frame, name):
    frame.round(ROUND).to_csv(RESULTS / name, index=False, lineterminator="\n")


def summarise(frame, by, cols):
    agg = frame.groupby(by, sort=False)[cols].agg(["mean", "std", "min", "max"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    return agg.reset_index()


def main():
    X, y = load()
    race_primary = primary_race_groups(X["race"].astype(str))
    RESULTS.mkdir(parents=True, exist_ok=True)
    metric_rows, importance_rows, share_rows, area_rows = [], [], [], []
    curve_rows, fair_rows, group_frames = [], [], []
    preds = {(k, v): {} for k in KINDS for v in ["aware", "unaware"]}
    timing = {}

    with logged("Phase 2 full run (25 folds; aware and unaware models; majority baseline)") as rec:
        for split_seed, fold, train, test in folds(y):
            ident = {"seed": split_seed, "fold": fold}
            base = majority().fit(X.iloc[train], y.iloc[train])
            score = base.predict_proba(X.iloc[test])[:, 1]
            pred = (score >= THRESHOLD).astype(int)
            metric_rows.append({"model": "majority", "variant": "aware", **ident,
                                "accuracy": accuracy_score(y.iloc[test], pred),
                                "f1": f1_score(y.iloc[test], pred, zero_division=0),
                                "roc_auc": roc_auc_score(y.iloc[test], score)})
            for kind in KINDS:
                start = datetime.now(timezone.utc)
                t0 = time.perf_counter()
                out = fold_aware(kind, X, y, split_seed, fold, train, test, race_primary)
                timing.setdefault((kind, "aware"), [start, None, 0.0])
                timing[(kind, "aware")][1] = datetime.now(timezone.utc)
                timing[(kind, "aware")][2] += time.perf_counter() - t0
                tag = {"model": kind, "variant": "aware", **ident}
                metric_rows.append({**tag, **out["metrics"]})
                importance_rows += [{"model": kind, **ident, "feature": f, "mean_abs_shap": v}
                                    for f, v in out["importance"].items()]
                share_rows.append({"model": kind, **ident,
                                   **{f"share_{f}": v for f, v in out["share"].items()}})
                a = out["areas"]
                area_rows.append({"model": kind, **ident, "area_signed": a["signed"],
                                  "area_absolute": a["absolute"], "area_random": a["random"],
                                  "random_minus_signed": a["random"] - a["signed"],
                                  "random_minus_absolute": a["random"] - a["absolute"]})
                for order, curve in out["curves"].items():
                    curve_rows += [{"model": kind, **ident, "order": order, "removed": i,
                                    "confidence": v} for i, v in enumerate(curve)]
                fair_rows += [{**tag, **m} for m in out["fairness"]]
                group_frames.append(out["groups"].assign(**tag))
                preds[(kind, "aware")].setdefault(split_seed, []).append(out["pred"])

                start = datetime.now(timezone.utc)
                t0 = time.perf_counter()
                un = fold_unaware(kind, X, y, train, test, race_primary)
                timing.setdefault((kind, "unaware"), [start, None, 0.0])
                timing[(kind, "unaware")][1] = datetime.now(timezone.utc)
                timing[(kind, "unaware")][2] += time.perf_counter() - t0
                if set(SENSITIVE) & set(un["inputs"]):
                    raise RuntimeError("Unaware model received sex or race")
                tag = {"model": kind, "variant": "unaware", **ident}
                metric_rows.append({**tag, **un["metrics"]})
                fair_rows += [{**tag, **m} for m in un["fairness"]]
                group_frames.append(un["groups"].assign(**tag))
                preds[(kind, "unaware")].setdefault(split_seed, []).append(un["pred"])
            print(f"seed {split_seed} fold {fold} done", flush=True)

        metrics = pd.DataFrame(metric_rows)
        save(metrics, "metrics_folds.csv")
        save(summarise(metrics, ["model", "variant"], ["accuracy", "f1", "roc_auc"]),
             "metrics_summary.csv")

        importance = pd.DataFrame(importance_rows)
        save(importance, "importance_folds.csv")
        save(summarise(importance, ["model", "feature"], ["mean_abs_shap"]),
             "importance_summary.csv")

        stab = []
        for kind in KINDS:
            part = importance[importance["model"] == kind]
            vectors = [g.set_index("feature")["mean_abs_shap"]
                       for _, g in part.groupby(["seed", "fold"], sort=True)]
            taus, overlaps = ranking_agreement(vectors)
            stab.append({"model": kind, "n_folds": len(vectors), "n_pairs": len(taus),
                         "kendall_tau_mean": taus.mean(), "kendall_tau_sd": taus.std(ddof=1),
                         "kendall_tau_min": taus.min(), "top5_overlap_mean": overlaps.mean(),
                         "top5_overlap_sd": overlaps.std(ddof=1),
                         "top5_overlap_min": overlaps.min()})
        save(pd.DataFrame(stab), "stability.csv")

        shares = pd.DataFrame(share_rows)
        save(shares, "attribution_share_folds.csv")
        save(summarise(shares, ["model"], [f"share_{f}" for f in SENSITIVE]),
             "attribution_share_summary.csv")

        areas = pd.DataFrame(area_rows)
        save(areas, "deletion_folds.csv")
        save(summarise(areas, ["model"], ["area_signed", "area_absolute", "area_random",
                                          "random_minus_signed", "random_minus_absolute"]),
             "deletion_summary.csv")
        curves = pd.DataFrame(curve_rows)
        save(curves.groupby(["model", "order", "removed"], sort=False)["confidence"]
             .mean().reset_index(), "deletion_curves.csv")

        fair = pd.DataFrame(fair_rows)
        save(fair, "fairness_folds.csv")
        save(summarise(fair, ["model", "variant", "attribute", "scope"],
                       ["demographic_parity_difference", "equalized_odds_difference"]),
             "fairness_summary.csv")
        groups = pd.concat(group_frames, ignore_index=True)
        save(groups[["model", "variant", "seed", "fold", "attribute", "group", "n",
                     "positives", "selection_rate", "true_positive_rate",
                     "false_positive_rate"]], "fairness_groups_folds.csv")

        pooled = []
        race = X["race"].astype(str)
        for (kind, variant), by_seed in preds.items():
            for split_seed, parts in by_seed.items():
                p = pd.concat(parts).sort_index()
                if not p.index.equals(y.index):
                    raise RuntimeError("Pooled predictions do not cover every row once")
                table = fairness.group_rates(y.to_numpy(), p.to_numpy(), race.to_numpy())
                table = table.reset_index().rename(columns={"index": "group"})
                pooled.append(table.assign(model=kind, variant=variant, seed=split_seed))
        pooled = pd.concat(pooled, ignore_index=True)
        pooled_mean = (pooled.groupby(["model", "variant", "group"], sort=False)
                       [["n", "positives", "selection_rate", "true_positive_rate",
                         "false_positive_rate"]].mean().reset_index())
        pooled_mean["in_primary_race_metrics"] = pooled_mean["group"].isin(race_primary)
        save(pooled_mean, "fairness_race_pooled.csv")

        figures.make_all(RESULTS)
        rec["outcome"] = (f"{len(metrics)} fold-model results (25 folds, 5 model variants); primary race groups "
                          f"{', '.join(race_primary)}")

    for (kind, variant), (start, end, secs) in timing.items():
        append(f"{kind} {variant}: 25 folds", start, end,
               f"completed; {secs:.1f} s of compute inside the folds")


if __name__ == "__main__":
    main()
