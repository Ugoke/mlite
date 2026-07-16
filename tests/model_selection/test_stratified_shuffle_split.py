import numpy as np
import pytest

from mlite.model_selection import stratified_shuffle_split


def test_output_shapes():
    X = np.arange(40, dtype=np.float64).reshape(20, 2)
    y = np.array([0] * 10 + [1] * 10, dtype=np.float64)

    X_train, X_test, y_train, y_test = stratified_shuffle_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    assert X_train.shape == (16, 2)
    assert X_test.shape == (4, 2)
    assert y_train.shape == (16,)
    assert y_test.shape == (4,)


def test_output_dtype():
    X = np.random.rand(20, 3)
    y = np.array([0] * 10 + [1] * 10)

    X_train, X_test, y_train, y_test = stratified_shuffle_split(X, y)

    assert X_train.dtype == np.float64
    assert X_test.dtype == np.float64
    assert y_train.dtype == np.float64
    assert y_test.dtype == np.float64


def test_reproducibility():
    X = np.random.rand(100, 4)
    y = np.array([0] * 50 + [1] * 50, dtype=np.float64)

    split1 = stratified_shuffle_split(
        X,
        y,
        random_state=123,
    )

    split2 = stratified_shuffle_split(
        X,
        y,
        random_state=123,
    )

    for a, b in zip(split1, split2):
        np.testing.assert_array_equal(a, b)


def test_different_random_state():
    X = np.random.rand(100, 4)
    y = np.array([0] * 50 + [1] * 50, dtype=np.float64)

    split1 = stratified_shuffle_split(
        X,
        y,
        random_state=1,
    )

    split2 = stratified_shuffle_split(
        X,
        y,
        random_state=2,
    )

    assert not np.array_equal(split1[0], split2[0])


def test_class_distribution():
    X = np.random.rand(100, 2)

    y = np.array(
        [0] * 80 +
        [1] * 20,
        dtype=np.float64,
    )

    _, _, y_train, y_test = stratified_shuffle_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    train_ratio = np.mean(y_train)
    test_ratio = np.mean(y_test)

    assert abs(train_ratio - 0.2) < 0.05
    assert abs(test_ratio - 0.2) < 0.05


def test_invalid_sample_count():
    X = np.random.rand(10, 2)
    y = np.random.rand(9)

    with pytest.raises(ValueError):
        stratified_shuffle_split(X, y)


def test_class_too_small():
    X = np.random.rand(3, 2)

    y = np.array(
        [0, 0, 1],
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        stratified_shuffle_split(
            X,
            y,
            test_size=0.5,
        )


def test_invalid_test_size():
    X = np.random.rand(10, 2)
    y = np.array([0] * 5 + [1] * 5, dtype=np.float64)

    with pytest.raises(ValueError):
        stratified_shuffle_split(
            X,
            y,
            test_size=0.0,
        )

    with pytest.raises(ValueError):
        stratified_shuffle_split(
            X,
            y,
            test_size=1.0,
        )


def test_returns_all_samples():
    X = np.random.rand(100, 3)
    y = np.array([0] * 60 + [1] * 40, dtype=np.float64)

    X_train, X_test, y_train, y_test = stratified_shuffle_split(X, y)

    assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
    assert y_train.shape[0] + y_test.shape[0] == y.shape[0]


def test_original_arrays_not_modified():
    X = np.random.rand(20, 2)
    y = np.array([0] * 10 + [1] * 10, dtype=np.float64)

    X_copy = X.copy()
    y_copy = y.copy()

    stratified_shuffle_split(X, y)

    np.testing.assert_array_equal(X, X_copy)
    np.testing.assert_array_equal(y, y_copy)