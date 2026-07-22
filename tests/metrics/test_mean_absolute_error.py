import numpy as np
import pytest

from mlite.metrics import mean_absolute_error


def test_mean_absolute_error_perfect_prediction():
    y_true = np.array([1, 2, 3, 4], dtype=np.float64)

    assert mean_absolute_error(y_true, y_true.copy()) == 0.0


def test_mean_absolute_error_manual_calculation():
    y_true = np.array([3, -0.5, 2, 7], dtype=np.float64)
    y_pred = np.array([2.5, 0.0, 2, 8], dtype=np.float64)

    expected = np.mean(np.abs(y_true - y_pred))

    assert mean_absolute_error(y_true, y_pred) == pytest.approx(expected)


def test_mean_absolute_error_converts_integer_arrays():
    assert mean_absolute_error(np.array([1, 2, 3]), np.array([2, 2, 2])) == pytest.approx(2 / 3)


def test_mean_absolute_error_rejects_empty_arrays():
    with pytest.raises(ValueError, match="cannot be empty"):
        mean_absolute_error(np.array([]), np.array([]))


def test_mean_absolute_error_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        mean_absolute_error(np.array([1, 2]), np.array([1]))
