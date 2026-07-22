import numpy as np
import pytest

from mlite.metrics import recall_score


def test_recall_score_perfect_prediction():
    y_true = np.array([1, 0, 1, 0], dtype=np.float64)

    assert recall_score(y_true, y_true.copy()) == 1.0


def test_recall_score_manual_calculation():
    y_true = np.array([1, 1, 1, 0], dtype=np.float64)
    y_pred = np.array([1, 0, 1, 1], dtype=np.float64)

    assert recall_score(y_true, y_pred) == pytest.approx(2 / 3)


def test_recall_score_returns_zero_without_positive_samples():
    y_true = np.array([0, 0, 0], dtype=np.float64)
    y_pred = np.array([0, 1, 0], dtype=np.float64)

    assert recall_score(y_true, y_pred) == 0.0


def test_recall_score_rejects_empty_arrays():
    with pytest.raises(ValueError, match="cannot be empty"):
        recall_score(np.array([]), np.array([]))


def test_recall_score_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        recall_score(np.array([1, 0]), np.array([1]))


def test_recall_score_rejects_non_binary_labels():
    with pytest.raises(ValueError, match="binary labels"):
        recall_score(np.array([1, 2]), np.array([1, 1]))
