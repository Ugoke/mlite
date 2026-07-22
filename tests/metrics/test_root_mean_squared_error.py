import numpy as np
import pytest

from mlite.metrics import root_mean_squared_error


def test_root_mean_squared_error_perfect_prediction():
    y_true = np.array([1, 2, 3, 4], dtype=np.float64)

    assert root_mean_squared_error(y_true, y_true.copy()) == 0.0


def test_root_mean_squared_error_manual_calculation():
    y_true = np.array([3, -0.5, 2, 7], dtype=np.float64)
    y_pred = np.array([2.5, 0.0, 2, 8], dtype=np.float64)

    expected = np.sqrt(np.mean((y_true - y_pred) ** 2))

    assert root_mean_squared_error(y_true, y_pred) == pytest.approx(expected)


def test_root_mean_squared_error_converts_integer_arrays():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([2, 2, 2])

    assert root_mean_squared_error(y_true, y_pred) == pytest.approx(np.sqrt(2 / 3))


def test_root_mean_squared_error_rejects_empty_arrays():
    with pytest.raises(ValueError, match="cannot be empty"):
        root_mean_squared_error(np.array([]), np.array([]))


def test_root_mean_squared_error_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        root_mean_squared_error(np.array([1, 2]), np.array([1]))
