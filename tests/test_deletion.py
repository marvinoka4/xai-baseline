import numpy as np
import pandas as pd
import pytest

from src.deletion import (abs_orders, curve_area, deletion_curves, random_orders,
                          replacement_values, signed_orders)


def predict(frame):
    """P(class 1) = 0.1 + 0.4 a + 0.2 b, for hand-checked curves."""
    return 0.1 + 0.4 * frame["a"].to_numpy() + 0.2 * frame["b"].to_numpy()


ROWS = pd.DataFrame({"a": [1.0, 0.0], "b": [1.0, 0.0]})
REPLACE = {"a": 0.0, "b": 0.0}


def test_curves_by_hand():
    # Row 0: p = 0.7, predicted class 1. Removing a then b: 0.7, 0.3, 0.1.
    # Row 1: p = 0.1, predicted class 0, confidence 0.9; removal changes nothing.
    curves = deletion_curves(predict, ROWS, np.array([[0, 1], [0, 1]]), REPLACE)
    np.testing.assert_allclose(curves, [[0.7, 0.3, 0.1], [0.9, 0.9, 0.9]])
    curves = deletion_curves(predict, ROWS, np.array([[1, 0], [1, 0]]), REPLACE)
    np.testing.assert_allclose(curves, [[0.7, 0.5, 0.1], [0.9, 0.9, 0.9]])


def test_area_by_hand():
    # x = 0, 0.5, 1: (0.7 + 0.3) / 2 * 0.5 + (0.3 + 0.1) / 2 * 0.5 = 0.35
    assert curve_area([0.7, 0.3, 0.1]) == pytest.approx(0.35)
    assert curve_area([0.7, 0.5, 0.1]) == pytest.approx(0.45)
    assert curve_area([1.0, 1.0]) == pytest.approx(1.0)


def test_signed_order_removes_largest_support_for_the_predicted_class_first():
    values = np.array([[0.1, -0.9, 0.5],    # predicted class 1: support 0.1, -0.9, 0.5
                       [0.1, -0.9, 0.5],    # predicted class 0: support -0.1, 0.9, -0.5
                       [0.3, 0.3, -0.2]])   # predicted class 1, tie broken by position
    predicted_one = np.array([True, False, True])
    np.testing.assert_array_equal(signed_orders(values, predicted_one),
                                  [[2, 0, 1], [1, 0, 2], [0, 1, 2]])


def test_abs_order_removes_largest_absolute_attribution_first():
    values = np.array([[0.1, -0.9, 0.5], [0.3, 0.3, -0.2]])
    np.testing.assert_array_equal(abs_orders(values), [[1, 2, 0], [0, 1, 2]])


def test_signed_and_abs_orders_give_different_curves_by_hand():
    # One row, predicted class 1 (p = 0.1 + 0.4 + 0.2 = 0.7). SHAP-like values:
    # a supports class 1 by 0.2, b opposes it by 0.6 (so |b| is largest).
    values = np.array([[0.2, -0.6]])
    row = pd.DataFrame({"a": [1.0], "b": [1.0]})
    signed = signed_orders(values, np.array([True]))   # a first: 0.7, 0.3, 0.1
    absolute = abs_orders(values)                       # b first: 0.7, 0.5, 0.1
    np.testing.assert_allclose(deletion_curves(predict, row, signed, REPLACE),
                               [[0.7, 0.3, 0.1]])
    np.testing.assert_allclose(deletion_curves(predict, row, absolute, REPLACE),
                               [[0.7, 0.5, 0.1]])


def test_random_orders_are_permutations():
    orders = random_orders(50, 4, np.random.default_rng(0))
    assert orders.shape == (50, 4)
    assert all(sorted(row) == [0, 1, 2, 3] for row in orders)


def test_categorical_replacement_and_curves_keep_categories():
    train = pd.DataFrame({"n": [1.0, 2.0, 10.0], "c": pd.Categorical(["x", "y", "y"])})
    assert replacement_values(train, ["n"], ["c"]) == {"n": 2.0, "c": "y"}
    rows = pd.DataFrame({"n": [5.0], "c": pd.Categorical(["x"], categories=["x", "y"])})

    def model(frame):
        return np.where(frame["c"].astype(str) == "x", 0.8, 0.4) + 0.0 * frame["n"].to_numpy()
    curves = deletion_curves(model, rows, np.array([[1, 0]]), {"n": 2.0, "c": "y"})
    np.testing.assert_allclose(curves, [[0.8, 0.4, 0.4]])
