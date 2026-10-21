import numpy as np
import pandas as pd
import pytest
import shap

from src.attributions import (attribution_share, column_owner, global_importance,
                              ranking_agreement, sum_to_features)
from src.models import logistic


def test_one_hot_columns_sum_back_to_their_feature():
    names = ["age", "sex_Female", "sex_Male", "race_Black", "race_White"]
    values = np.array([[1.0, 0.5, -0.25, 0.1, 0.2],
                       [-2.0, 0.0, 0.75, -0.3, 0.0]])
    out = sum_to_features(values, names, ["age"], ["sex", "race"])
    expected = pd.DataFrame({"age": [1.0, -2.0], "sex": [0.25, 0.75], "race": [0.3, -0.3]})
    pd.testing.assert_frame_equal(out, expected, check_names=False)
    np.testing.assert_allclose(out.sum(axis=1), values.sum(axis=1))


def test_longest_feature_name_wins_when_names_share_a_prefix():
    names = ["status_x", "status_group_y", "status_group_z"]
    assert column_owner(names, [], ["status", "status_group"]) == [
        "status", "status_group", "status_group"]


def test_unknown_encoded_column_is_rejected():
    with pytest.raises(ValueError):
        column_owner(["mystery_1"], ["age"], ["sex"])


def test_linear_shap_summed_back_keeps_additivity():
    rng = np.random.default_rng(42)
    n = 300
    X = pd.DataFrame({"num": rng.normal(0, 1, n),
                      "cat": pd.Categorical(rng.choice(["a", "b", "c"], n))})
    y = pd.Series(((X["num"] + (X["cat"] == "b") * 1.5 + rng.normal(0, 1, n)) > 0.5).astype(int))
    pipe = logistic(numeric=["num"], categorical=["cat"]).fit(X, y)
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    Z = prep.transform(X)
    explainer = shap.LinearExplainer(model, Z)
    values = explainer.shap_values(Z)
    per_feature = sum_to_features(values, prep.get_feature_names_out(), ["num"], ["cat"])
    margin = model.decision_function(Z)
    np.testing.assert_allclose(per_feature.sum(axis=1) + explainer.expected_value, margin,
                               atol=1e-6)


def test_ranking_agreement_on_identical_and_reversed_rankings():
    a = pd.Series({"f1": 5.0, "f2": 4.0, "f3": 3.0, "f4": 2.0, "f5": 1.0, "f6": 0.5})
    b = pd.Series({"f1": 0.5, "f2": 1.0, "f3": 2.0, "f4": 3.0, "f5": 4.0, "f6": 5.0})
    taus, overlaps = ranking_agreement([a, a.copy()])
    assert taus[0] == pytest.approx(1.0) and overlaps[0] == 1.0
    taus, overlaps = ranking_agreement([a, b])
    assert taus[0] == pytest.approx(-1.0)
    assert overlaps[0] == pytest.approx(4 / 5)  # top 5 of a: f1..f5; of b: f2..f6


def test_global_importance_and_attribution_share():
    shap_values = pd.DataFrame({"sex": [1.0, -1.0], "race": [0.5, 0.5], "age": [-2.5, 0.5]})
    np.testing.assert_allclose(global_importance(shap_values).values, [1.0, 0.5, 1.5])
    share = attribution_share(shap_values, ["sex", "race"])
    assert share == {"sex": pytest.approx(2 / 6), "race": pytest.approx(1 / 6)}
