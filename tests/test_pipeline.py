import numpy as np
import pandas as pd
import pytest

from src import runlog
from src.cv import folds
from src.pipeline import (deletion_rows_seed, fold_aware, fold_unaware, primary_race_groups,
                          random_orders_seed)
from src.run_all import summarise


def toy(n=400, seed=3):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "age": rng.integers(18, 70, n).astype(float),
        "education_num": rng.integers(1, 16, n).astype(float),
        "capital_gain": rng.choice([0.0, 0.0, 0.0, 5000.0], n),
        "capital_loss": rng.choice([0.0, 0.0, 0.0, 1500.0], n),
        "hours_per_week": rng.integers(10, 60, n).astype(float),
    })
    cats = {"workclass": ["Private", "Self", "Gov"], "marital_status": ["Married", "Never"],
            "occupation": ["A", "B", "C"], "relationship": ["Husband", "Wife", "Own-child"],
            "race": ["White", "Black", "Asian-Pac-Islander", "Other"],
            "sex": ["Male", "Female"], "native_country": ["US", "Other"]}
    for col, levels in cats.items():
        X[col] = pd.Categorical(rng.choice(levels, n), categories=levels)
    logit = (0.05 * (X["age"] - 40) + 0.3 * (X["education_num"] - 9)
             + 1.0 * (X["marital_status"] == "Married"))
    y = pd.Series((logit + rng.normal(0, 1, n) > 0).astype(int))
    return X, y


@pytest.fixture(scope="module")
def outputs():
    X, y = toy()
    s, k, train, test = next(folds(y))
    primary = ["Black", "White"]
    run = {kind: fold_aware(kind, X, y, s, k, train, test, primary, deletion_n=30,
                            random_n=2, bg_rows=40) for kind in ["lightgbm", "logistic"]}
    return X, y, train, test, primary, run


def test_aware_outputs_are_complete(outputs):
    X, _, _, test, _, run = outputs
    for out in run.values():
        assert set(out["metrics"]) == {"accuracy", "f1", "roc_auc"}
        assert list(out["importance"].index) == list(X.columns)
        assert set(out["share"]) == {"sex", "race"}
        assert set(out["curves"]) == {"signed", "absolute", "random"}
        assert all(len(c) == X.shape[1] + 1 for c in out["curves"].values())
        assert all(0 <= a <= 1 for a in out["areas"].values())
        assert out["pred"].index.equals(X.iloc[test].index)
        assert [m["scope"] for m in out["fairness"]] == ["sex", "race_primary", "race_all"]


def test_curves_start_at_the_confidence_of_the_original_prediction(outputs):
    for out in outputs[5].values():
        starts = {name: c[0] for name, c in out["curves"].items()}
        assert starts["signed"] == pytest.approx(starts["absolute"])
        assert starts["signed"] == pytest.approx(starts["random"])
        assert starts["signed"] >= 0.5


def test_aware_run_is_deterministic(outputs):
    X, y, train, test, primary, run = outputs
    s, k, _, _ = next(folds(y))
    again = fold_aware("lightgbm", X, y, s, k, train, test, primary, deletion_n=30,
                       random_n=2, bg_rows=40)
    assert again["metrics"] == run["lightgbm"]["metrics"]
    assert again["areas"] == run["lightgbm"]["areas"]
    pd.testing.assert_series_equal(again["importance"], run["lightgbm"]["importance"])


def test_unaware_model_never_sees_sex_or_race(outputs):
    X, y, train, test, primary, _ = outputs
    un = fold_unaware("logistic", X, y, train, test, primary)
    assert "sex" not in un["inputs"] and "race" not in un["inputs"]
    assert [m["scope"] for m in un["fairness"]] == ["sex", "race_primary", "race_all"]
    shuffled = X.copy()
    shuffled["sex"] = shuffled["sex"].sample(frac=1, random_state=0).to_numpy()
    shuffled["race"] = shuffled["race"].sample(frac=1, random_state=1).to_numpy()
    un2 = fold_unaware("logistic", shuffled, y, train, test, primary)
    assert un2["metrics"] == un["metrics"]  # predictions do not depend on sex or race


def test_primary_race_groups_use_the_size_threshold():
    race = pd.Series(["White"] * 1500 + ["Black"] * 1000 + ["Other"] * 999)
    assert primary_race_groups(race) == ["Black", "White"]


def test_seeds_are_distinct_per_fold():
    seeds = {deletion_rows_seed(s, k) for s in range(5) for k in range(5)}
    assert len(seeds) == 25
    rseeds = {random_orders_seed(s, k) for s in range(5) for k in range(5)}
    assert len(rseeds) == 25
    # Within a fold the row sample and the random orders use different streams.
    assert all(deletion_rows_seed(s, k) != random_orders_seed(s, k)
               for s in range(5) for k in range(5))


def test_summarise_gives_mean_sd_min_max():
    frame = pd.DataFrame({"m": ["a", "a", "b"], "x": [1.0, 3.0, 5.0]})
    out = summarise(frame, ["m"], ["x"]).set_index("m")
    assert out.loc["a", "x_mean"] == 2.0
    assert out.loc["a", "x_std"] == pytest.approx(np.sqrt(2))
    assert out.loc["a", "x_min"] == 1.0 and out.loc["a", "x_max"] == 3.0


def test_run_log_keeps_timing_out_of_result_files(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RESULTS", tmp_path)
    monkeypatch.setattr(runlog, "RUN_LOG", tmp_path / "run_log.md")
    with runlog.logged("demo") as rec:
        rec["outcome"] = "ok | fine"
    text = (tmp_path / "run_log.md").read_text(encoding="utf-8")
    assert "| demo |" in text and "ok / fine" in text and "\r" not in text
    assert [p.name for p in tmp_path.iterdir()] == ["run_log.md"]
