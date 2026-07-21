import numpy as np
import pytest

from mlite.model_selection import cross_val_score
from mlite.linear_models import LinearRegression


def test_cross_val_score_basic():
    X = np.arange(40, dtype=np.float64).reshape(20, 2)
    y = np.arange(20, dtype=np.float64)

    model = LinearRegression()

    scores = cross_val_score(model, X, y, n_splits=5)

    assert isinstance(scores, np.ndarray)
    assert scores.dtype == np.float64
    assert scores.shape == (5,)


def test_cross_val_score_with_shuffle():
    X = np.arange(100, dtype=np.float64).reshape(50, 2)
    y = np.arange(50, dtype=np.float64)

    model = LinearRegression()

    scores = cross_val_score(model, X, y, n_splits=5, shuffle=True, random_state=42)

    assert scores.shape == (5,)


def test_cross_val_score_reproducible():
    X = np.arange(60, dtype=np.float64).reshape(30, 2)
    y = np.arange(30, dtype=np.float64)

    model = LinearRegression()

    scores1 = cross_val_score(model, X, y, shuffle=True, random_state=42)
    scores2 = cross_val_score(model, X, y, shuffle=True, random_state=42)

    np.testing.assert_array_equal(scores1, scores2)


def test_cross_val_score_different_seed():
    X = np.arange(60, dtype=np.float64).reshape(30, 2)
    y = np.arange(30, dtype=np.float64)

    model = LinearRegression()

    scores1 = cross_val_score(model, X, y, shuffle=True, random_state=1)
    scores2 = cross_val_score(model, X, y, shuffle=True, random_state=2)

    assert not np.array_equal(scores1, scores2)


def test_cross_val_score_does_not_fit_original_model():
    X = np.arange(40, dtype=np.float64).reshape(20, 2)
    y = np.arange(20, dtype=np.float64)

    model = LinearRegression()

    cross_val_score(model, X, y, n_splits=5)

    with pytest.raises(RuntimeError):
        _ = model.coef_


def test_cross_val_score_custom_splits():
    X = np.arange(80, dtype=np.float64).reshape(40, 2)
    y = np.arange(40, dtype=np.float64)

    model = LinearRegression()

    scores = cross_val_score(model, X, y, n_splits=4)

    assert len(scores) == 4


def test_cross_val_score_sample_mismatch():
    X = np.ones((10, 2), dtype=np.float64)
    y = np.ones(9, dtype=np.float64)

    model = LinearRegression()

    with pytest.raises(ValueError):
        cross_val_score(model, X, y)


def test_cross_val_score_invalid_n_splits():
    X = np.ones((10, 2), dtype=np.float64)
    y = np.ones(10, dtype=np.float64)

    model = LinearRegression()

    with pytest.raises(ValueError):
        cross_val_score(model, X, y, n_splits=1)


def test_cross_val_score_invalid_X_dimension():
    X = np.ones((2, 2, 2), dtype=np.float64)
    y = np.ones(2, dtype=np.float64)

    model = LinearRegression()

    with pytest.raises(ValueError):
        cross_val_score(model, X, y)


def test_cross_val_score_invalid_shuffle_type():
    X = np.ones((10, 2), dtype=np.float64)
    y = np.ones(10, dtype=np.float64)

    model = LinearRegression()

    with pytest.raises(TypeError):
        cross_val_score(model, X, y, shuffle=1)


def test_cross_val_score_invalid_random_state():
    X = np.ones((10, 2), dtype=np.float64)
    y = np.ones(10, dtype=np.float64)

    model = LinearRegression()

    with pytest.raises(ValueError):
        cross_val_score(model, X, y, random_state=-1)