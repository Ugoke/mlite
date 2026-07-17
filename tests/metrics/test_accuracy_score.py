import numpy as np
import pytest

from mlite.metrics import accuracy_score


def test_accuracy_perfect_prediction():
    y_true = np.array([0, 1, 0, 1], dtype=np.float64)
    y_pred = np.array([0, 1, 0, 1], dtype=np.float64)

    result = accuracy_score(y_true, y_pred)

    assert result == 1.0


def test_accuracy_all_wrong_prediction():
    y_true = np.array([0, 1, 0, 1], dtype=np.float64)
    y_pred = np.array([1, 0, 1, 0], dtype=np.float64)

    result = accuracy_score(y_true, y_pred)

    assert result == 0.0


def test_accuracy_half_correct():
    y_true = np.array([0, 1, 1, 0], dtype=np.float64)
    y_pred = np.array([0, 0, 1, 1], dtype=np.float64)

    result = accuracy_score(y_true, y_pred)

    assert result == 0.5


def test_accuracy_multiclass():
    y_true = np.array([0, 1, 2, 2, 1], dtype=np.float64)

    y_pred = np.array([0, 2, 2, 1, 1], dtype=np.float64)

    result = accuracy_score(y_true, y_pred)

    assert result == 3 / 5


def test_accuracy_single_sample_correct():
    y_true = np.array([1], dtype=np.float64)
    y_pred = np.array([1], dtype=np.float64)

    assert accuracy_score(y_true, y_pred) == 1.0


def test_accuracy_single_sample_wrong():
    y_true = np.array([1], dtype=np.float64)
    y_pred = np.array([0], dtype=np.float64)

    assert accuracy_score(y_true, y_pred) == 0.0


def test_accuracy_empty_arrays():
    y_true = np.array([], dtype=np.float64)
    y_pred = np.array([], dtype=np.float64)

    with pytest.raises(ValueError, match="y cannot be empty"):
        accuracy_score(y_true, y_pred)


def test_accuracy_float_labels():
    y_true = np.array([0., 1., 1.])
    y_pred = np.array([0., 1., 0.])

    result = accuracy_score(y_true, y_pred)

    assert result == pytest.approx(2 / 3)


def test_accuracy_large_dataset():
    rng = np.random.default_rng(42)

    y_true = rng.integers(0, 2, size=1000).astype(np.float64)
    y_pred = y_true.copy()

    result = accuracy_score(y_true, y_pred)

    assert result == 1.0