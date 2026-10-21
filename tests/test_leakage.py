import numpy as np
import pandas as pd

from src.cv import folds
from src.deletion import replacement_values
from src.models import lightgbm, logistic


def toy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "num": rng.normal(0, 1, n),
        "cat": pd.Categorical(rng.choice(["a", "b", "c"], n)),
    })
    y = pd.Series((X["num"] + rng.normal(0, 1, n) > 0).astype(int))
    return X, y


def test_folds_partition_rows_per_seed_and_stay_stratified():
    _, y = toy(503)
    seen = {}
    for seed, k, train, test in folds(y):
        assert not set(train) & set(test)
        assert len(train) + len(test) == len(y)
        seen.setdefault(seed, []).extend(test)
        assert abs(y.iloc[test].mean() - y.mean()) < 0.02
    assert sorted(seen) == [0, 1, 2, 3, 4]
    for rows in seen.values():
        assert sorted(rows) == list(range(len(y)))


def test_scaler_and_encoder_learn_from_the_training_fold_only():
    X, y = toy()
    train, test = np.arange(0, 150), np.arange(150, 200)
    X = X.copy()
    X.loc[test, "num"] += 1000.0  # held-out rows far from the training rows
    X["cat"] = X["cat"].cat.add_categories(["only_in_test"])
    X.loc[test[:5], "cat"] = "only_in_test"
    pipe = logistic(numeric=["num"], categorical=["cat"]).fit(X.iloc[train], y.iloc[train])
    scaler = pipe.named_steps["prep"].named_transformers_["num"]
    assert np.isclose(scaler.mean_[0], X.iloc[train]["num"].mean())
    onehot = pipe.named_steps["prep"].named_transformers_["cat"]
    assert "only_in_test" not in list(onehot.categories_[0])
    encoded = pipe.named_steps["prep"].transform(X.iloc[test[:5]])
    cat_cols = [i for i, n in enumerate(pipe.named_steps["prep"].get_feature_names_out())
                if n.startswith("cat_")]
    assert np.all(encoded[:, cat_cols] == 0)


def test_lightgbm_pipeline_does_not_see_held_out_rows():
    X, y = toy()
    train = np.arange(0, 150)
    a = lightgbm(numeric=["num"], categorical=["cat"]).fit(X.iloc[train], y.iloc[train])
    X2 = X.copy()
    X2.loc[150:, "num"] = 1e6
    b = lightgbm(numeric=["num"], categorical=["cat"]).fit(X2.iloc[train], y.iloc[train])
    np.testing.assert_array_equal(a.predict_proba(X.iloc[:150]), b.predict_proba(X.iloc[:150]))


def test_replacement_values_use_the_training_fold_only():
    X, _ = toy()
    train = X.iloc[:150]
    full = X.copy()
    full.loc[150:, "num"] = 1e6
    full.loc[150:, "cat"] = "c"
    assert replacement_values(train, ["num"], ["cat"]) == replacement_values(
        full.iloc[:150], ["num"], ["cat"])
    assert replacement_values(train, ["num"], ["cat"])["num"] == train["num"].median()
