import numpy as np
import pandas as pd
import pytest

from src.explain import background, background_seed, background_size, explainer, feature_shap
from src.models import lightgbm, logistic


def toy(n=600, seed=7):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"num": rng.normal(0, 1, n),
                      "num2": rng.normal(0, 1, n),
                      "cat": pd.Categorical(rng.choice(["a", "b", "c"], n))})
    logit = X["num"] - 0.5 * X["num2"] + 1.2 * (X["cat"] == "b") + rng.normal(0, 1, n)
    return X, pd.Series((logit > 0.3).astype(int))


def test_background_is_drawn_from_the_training_fold_with_a_fixed_seed():
    X, _ = toy()
    train = X.iloc[100:500]
    a = background(train, split_seed=3, fold=2)
    b = background(train, split_seed=3, fold=2)
    assert len(a) == 200
    assert set(a.index) <= set(train.index)
    pd.testing.assert_frame_equal(a, b)
    assert not background(train, split_seed=3, fold=1).index.equals(a.index)
    assert background_seed(3, 2) == 2032


@pytest.mark.parametrize("kind,factory", [("lightgbm", lightgbm), ("logistic", logistic)])
def test_interventional_shap_summed_back_is_additive(kind, factory):
    X, y = toy()
    train, test = X.iloc[:450], X.iloc[450:]
    pipe = factory(numeric=["num", "num2"], categorical=["cat"]).fit(train, y.iloc[:450])
    bg = background(train, split_seed=0, fold=0)
    values, expected = feature_shap(pipe, kind, bg, test, ["num", "num2"], ["cat"])
    assert list(values.columns) == ["num", "num2", "cat"]
    assert values.index.equals(test.index)
    assert background_size(explainer(pipe, kind, bg)) == 200  # no silent subsampling
    margin = pipe.named_steps["model"].predict_proba(pipe.named_steps["prep"].transform(test))
    log_odds = np.log(margin[:, 1] / margin[:, 0])
    np.testing.assert_allclose(values.sum(axis=1) + expected, log_odds, atol=1e-5)
