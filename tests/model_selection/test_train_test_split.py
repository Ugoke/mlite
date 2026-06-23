import numpy as np
import pytest

from mlite.model_selection import train_test_split


def test_train_test_split_shapes():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)
    y = np.array([0, 0, 1, 1], dtype=np.float64)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)

    assert X_train.shape[1] == X.shape[1]
    assert X_test.shape[1] == X.shape[1]


def test_train_test_split_content_preserved():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)
    y = np.array([0, 0, 1, 1], dtype=np.float64)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

    X_all = np.vstack([X_train, X_test])
    y_all = np.hstack([y_train, y_test])

    assert set(map(tuple, X_all)) == set(map(tuple, X))
    assert set(y_all) == set(y)


def test_train_test_split_deterministic():
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 2, 20).astype(np.float64)

    out1 = train_test_split(X, y, test_size=0.3, random_state=123)
    out2 = train_test_split(X, y, test_size=0.3, random_state=123)

    for a, b in zip(out1, out2):
        assert np.allclose(a, b)


def test_train_test_split_different_random_state():
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 2, 20).astype(np.float64)

    out1 = train_test_split(X, y, test_size=0.3, random_state=1)
    out2 = train_test_split(X, y, test_size=0.3, random_state=999)

    assert not np.array_equal(out1[0], out2[0])


def test_invalid_test_size():
    X = np.array([[1, 2], [3, 4]], dtype=np.float64)
    y = np.array([0, 1], dtype=np.float64)

    with pytest.raises(ValueError):
        train_test_split(X, y, test_size=1.5)


def test_invalid_random_state_type():
    X = np.array([[1, 2], [3, 4]], dtype=np.float64)
    y = np.array([0, 1], dtype=np.float64)

    with pytest.raises(TypeError):
        train_test_split(X, y, random_state="bad")