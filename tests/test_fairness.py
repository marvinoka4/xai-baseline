import numpy as np
import pytest

from src.fairness import disparities, group_rates, restrict

# Group A: 4 rows. y_true 1,1,0,0; y_pred 1,0,1,0 -> selection 2/4, TPR 1/2, FPR 1/2.
# Group B: 4 rows. y_true 1,1,0,0; y_pred 1,1,0,0 -> selection 2/4, TPR 1,   FPR 0.
# Group C: 2 rows. y_true 1,0;     y_pred 0,0     -> selection 0,   TPR 0,   FPR 0.
Y_TRUE = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 0])
Y_PRED = np.array([1, 0, 1, 0, 1, 1, 0, 0, 0, 0])
GROUPS = np.array(["A"] * 4 + ["B"] * 4 + ["C"] * 2)


def test_disparities_by_hand():
    out = disparities(Y_TRUE, Y_PRED, GROUPS)
    assert out["demographic_parity_difference"] == pytest.approx(0.5)  # 0.5 - 0
    # TPR gap: 1 - 0 = 1; FPR gap: 0.5 - 0 = 0.5; equalized odds difference = 1.
    assert out["equalized_odds_difference"] == pytest.approx(1.0)


def test_disparities_two_groups_by_hand():
    y_true, y_pred, groups = restrict(Y_TRUE, Y_PRED, GROUPS, ["A", "B"])
    out = disparities(y_true, y_pred, groups)
    assert out["demographic_parity_difference"] == pytest.approx(0.0)
    assert out["equalized_odds_difference"] == pytest.approx(0.5)  # TPR 1 vs 0.5, FPR 0 vs 0.5


def test_group_rates_by_hand():
    table = group_rates(Y_TRUE, Y_PRED, GROUPS)
    assert table.loc["A", "selection_rate"] == pytest.approx(0.5)
    assert table.loc["A", "true_positive_rate"] == pytest.approx(0.5)
    assert table.loc["A", "false_positive_rate"] == pytest.approx(0.5)
    assert table.loc["B", "true_positive_rate"] == pytest.approx(1.0)
    assert table.loc["C", "selection_rate"] == pytest.approx(0.0)
    assert table["n"].to_dict() == {"A": 4, "B": 4, "C": 2}
    assert table["positives"].to_dict() == {"A": 2, "B": 2, "C": 1}


def test_restrict_keeps_only_listed_groups():
    _, _, groups = restrict(Y_TRUE, Y_PRED, GROUPS, ["A", "C"])
    assert sorted(set(groups)) == ["A", "C"] and len(groups) == 6
