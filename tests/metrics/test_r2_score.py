import numpy as np
import pytest

from mlite.metrics import r2_score


def test_r2_perfect_prediction():
    y_true = np.array([1, 2, 3, 4], dtype=np.float64)
    y_pred = y_true.copy()

    result = r2_score(y_true, y_pred)

    assert result == 1.0


def test_r2_manual_calculation():
    y_true = np.array([3, -0.5, 2, 7], dtype=np.float64)
    y_pred = np.array([2.5, 0.0, 2, 8], dtype=np.float64)

    mean = np.mean(y_true)

    ss_total = np.sum((y_true - mean) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)

    expected = 1 - (ss_residual / ss_total)

    result = r2_score(y_true, y_pred)

    assert result == pytest.approx(expected)


def test_r2_zero_score():
    y_true = np.array([1, 2, 3, 4], dtype=np.float64)
    y_pred = np.array([2.5, 2.5, 2.5, 2.5], dtype=np.float64)

    result = r2_score(y_true, y_pred)

    assert result == pytest.approx(0.0)


def test_r2_negative_score():
    y_true = np.array([1, 2, 3, 4], dtype=np.float64)
    y_pred = np.array([10, 10, 10, 10], dtype=np.float64)

    result = r2_score(y_true, y_pred)

    assert result < 0.0


def test_r2_constant_target():
    y_true = np.array([5, 5, 5, 5], dtype=np.float64)
    y_pred = np.array([5, 5, 5, 5], dtype=np.float64)

    result = r2_score(y_true, y_pred)

    assert result == 0.0


def test_r2_single_sample():
    y_true = np.array([1], dtype=np.float64)
    y_pred = np.array([1], dtype=np.float64)

    result = r2_score(y_true, y_pred)

    assert result == 0.0


def test_r2_small_error():
    y_true = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1], dtype=np.float64)

    result = r2_score(y_true, y_pred)

    assert result > 0.95


def test_r2_large_dataset():
    rng = np.random.default_rng(42)

    y_true = rng.normal(size=1000)
    y_pred = y_true + rng.normal(scale=0.1, size=1000)

    result = r2_score(y_true, y_pred)

    assert result > 0.9


def test_r2_dtype_conversion():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])

    result = r2_score(y_true, y_pred)

    assert result == 1.0