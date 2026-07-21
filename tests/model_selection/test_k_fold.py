import numpy as np
import pytest

from mlite.model_selection import k_fold


def test_k_fold_basic():
    X = np.arange(20, dtype=np.float64).reshape(10, 2)
    y = np.arange(10, dtype=np.float64)

    folds = k_fold(X, y)

    assert isinstance(folds, list)
    assert len(folds) == 5

    for X_train, X_test, y_train, y_test in folds:
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)

        assert X_train.dtype == np.float64
        assert X_test.dtype == np.float64
        assert y_train.dtype == np.float64
        assert y_test.dtype == np.float64

        assert X_train.shape[1] == 2
        assert X_test.shape[1] == 2

        assert len(X_train) + len(X_test) == 10
        assert len(y_train) + len(y_test) == 10


def test_k_fold_one_dimensional_X():
    X = np.arange(10, dtype=np.float64)
    y = np.arange(10, dtype=np.float64)

    folds = k_fold(X, y)

    for X_train, X_test, _, _ in folds:
        assert X_train.ndim == 2
        assert X_test.ndim == 2
        assert X_train.shape[1] == 1
        assert X_test.shape[1] == 1


def test_k_fold_custom_n_splits():
    X = np.arange(24, dtype=np.float64).reshape(12, 2)
    y = np.arange(12, dtype=np.float64)

    folds = k_fold(X, y, n_splits=3)

    assert len(folds) == 3

    for X_train, X_test, _, _ in folds:
        assert len(X_test) == 4
        assert len(X_train) == 8


def test_k_fold_reproducible_shuffle():
    X = np.arange(40, dtype=np.float64).reshape(20, 2)
    y = np.arange(20, dtype=np.float64)

    folds1 = k_fold(X, y, shuffle=True, random_state=42)
    folds2 = k_fold(X, y, shuffle=True, random_state=42)

    for f1, f2 in zip(folds1, folds2):
        for a, b in zip(f1, f2):
            np.testing.assert_array_equal(a, b)


def test_k_fold_different_seed():
    X = np.arange(40, dtype=np.float64).reshape(20, 2)
    y = np.arange(20, dtype=np.float64)

    folds1 = k_fold(X, y, shuffle=True, random_state=1)
    folds2 = k_fold(X, y, shuffle=True, random_state=2)

    different = False

    for f1, f2 in zip(folds1, folds2):
        if not np.array_equal(f1[1], f2[1]):
            different = True
            break

    assert different


def test_k_fold_shuffle_false_is_deterministic():
    X = np.arange(20, dtype=np.float64).reshape(10, 2)
    y = np.arange(10, dtype=np.float64)

    folds = k_fold(X, y, shuffle=False)

    _, X_test, _, y_test = folds[0]

    np.testing.assert_array_equal(X_test, X[:2])
    np.testing.assert_array_equal(y_test, y[:2])


def test_invalid_sample_count():
    X = np.ones((10, 2))
    y = np.ones(9)

    with pytest.raises(ValueError):
        k_fold(X, y)


def test_invalid_n_splits_too_small():
    X = np.ones((10, 2))
    y = np.ones(10)

    with pytest.raises(ValueError):
        k_fold(X, y, n_splits=1)


def test_invalid_n_splits_too_large():
    X = np.ones((5, 2))
    y = np.ones(5)

    with pytest.raises(ValueError):
        k_fold(X, y, n_splits=6)


def test_invalid_n_splits_type():
    X = np.ones((10, 2))
    y = np.ones(10)

    with pytest.raises(TypeError):
        k_fold(X, y, n_splits=5.0)


def test_invalid_shuffle_type():
    X = np.ones((10, 2))
    y = np.ones(10)

    with pytest.raises(TypeError):
        k_fold(X, y, shuffle=1)


def test_empty_input():
    X = np.empty((0, 2))
    y = np.empty((0,))

    with pytest.raises(ValueError):
        k_fold(X, y)


def test_three_dimensional_X():
    X = np.ones((2, 2, 2))
    y = np.ones(2)

    with pytest.raises(ValueError):
        k_fold(X, y)


def test_returned_arrays_are_independent():
    X = np.arange(20, dtype=np.float64).reshape(10, 2)
    y = np.arange(10, dtype=np.float64)

    folds = k_fold(X, y)

    X_train, _, _, _ = folds[0]

    X_train[0, 0] = -999

    assert X[0, 0] != -999