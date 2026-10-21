"""Figures for the Phase 2 results, drawn from the saved CSVs."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LABELS = {"lightgbm": "LightGBM", "logistic": "Logistic regression"}
META = {"Software": None}


def importance(results):
    data = pd.read_csv(results / "importance_summary.csv")
    order = (data[data["model"] == "lightgbm"].sort_values("mean_abs_shap_mean")["feature"]
             .tolist())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, kind in zip(axes, ["lightgbm", "logistic"]):
        part = data[data["model"] == kind].set_index("feature").loc[order]
        ax.barh(order, part["mean_abs_shap_mean"], xerr=part["mean_abs_shap_std"],
                color="#1f4e79", ecolor="#555555", capsize=2)
        ax.set_title(LABELS[kind], fontsize=10)
        ax.set_xlabel("Mean absolute SHAP (log-odds), mean and SD over 25 folds")
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Global feature importance, interventional SHAP on a shared background",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(results / "figures" / "global_importance.png", dpi=120, metadata=META)
    plt.close(fig)


def deletion(results):
    data = pd.read_csv(results / "deletion_curves.csv")
    styles = {"signed": ("-o", "Signed SHAP order (primary)"),
              "absolute": ("-s", "Absolute SHAP order (secondary)"),
              "random": ("--", "Random order (mean of 5)")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, kind in zip(axes, ["lightgbm", "logistic"]):
        for order, (style, label) in styles.items():
            part = data[(data["model"] == kind) & (data["order"] == order)]
            ax.plot(part["removed"], part["confidence"], style, ms=3, label=label)
        ax.set_title(LABELS[kind], fontsize=10)
        ax.set_xlabel("Features removed (replaced by training-fold median or mode)")
        ax.set_xticks(range(0, 13))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Mean probability of the originally predicted class")
    axes[1].legend(fontsize=8)
    fig.suptitle("Deletion curves, mean over 25 folds (500 held-out rows per fold)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(results / "figures" / "deletion_curves.png", dpi=120, metadata=META)
    plt.close(fig)


def fairness(results):
    data = pd.read_csv(results / "fairness_summary.csv")
    scopes = [("sex", "Sex"), ("race_primary", "Race (3 largest groups)"),
              ("race_all", "Race (all 5 groups)")]
    metrics = [("demographic_parity_difference", "Demographic parity difference"),
               ("equalized_odds_difference", "Equalized odds difference")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    bars = [("lightgbm", "aware"), ("lightgbm", "unaware"), ("logistic", "aware"),
            ("logistic", "unaware")]
    colours = ["#1f4e79", "#8fb3d9", "#b5651d", "#e3b98b"]
    width = 0.2
    x = np.arange(len(scopes))
    for ax, (metric, title) in zip(axes, metrics):
        for i, ((kind, variant), colour) in enumerate(zip(bars, colours)):
            part = data[(data["model"] == kind) & (data["variant"] == variant)]
            part = part.set_index("scope").loc[[s for s, _ in scopes]]
            ax.bar(x + (i - 1.5) * width, part[f"{metric}_mean"], width,
                   yerr=part[f"{metric}_std"], color=colour, capsize=2,
                   label=f"{LABELS[kind]}, {variant}")
        ax.set_xticks(x, [label for _, label in scopes], fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Mean over 25 folds (error bars: SD)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Fairness metrics, models with and without sex and race as inputs",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(results / "figures" / "fairness_aware_unaware.png", dpi=120, metadata=META)
    plt.close(fig)


def make_all(results):
    (results / "figures").mkdir(parents=True, exist_ok=True)
    importance(results)
    deletion(results)
    fairness(results)
