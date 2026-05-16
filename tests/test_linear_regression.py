import numpy as np

import pytest

from mlite import LinearRegression


def test_fit_and_predict() -> None:
    model = LinearRegression()

    X = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
        ]
    )

    y = np.array(
        [
            2.0,
            4.0,
            6.0,
        ]
    )

    model.fit(X, y)

    pred = model.predict(
        np.array(
            [
                [4.0],
                [5.0],
            ]
        )
    )

    assert pred.shape == (2,)
    assert np.allclose(pred, np.array([8.0, 10.0]), atol=1.0)


def test_fit_accepts_1d_input() -> None:

    model = LinearRegression()

    X = np.array([1.0, 2.0, 3.0])

    y = np.array([2.0, 4.0, 6.0])

    model.fit(X, y)

    pred = model.predict(np.array([4.0]))

    assert pred.shape == (1,)


def test_predict_accepts_1d_input() -> None:

    model = LinearRegression()

    X = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
        ]
    )

    y = np.array(
        [
            2.0,
            4.0,
            6.0,
        ]
    )

    model.fit(X, y)

    pred = model.predict(np.array([4.0, 5.0]))

    assert pred.shape == (2,)


def test_invalid_y_dimension() -> None:
    model = LinearRegression()

    X = np.array(
        [
            [1.0],
            [2.0],
        ]
    )

    y = np.array(
        [
            [1.0],
            [2.0],
        ]
    )

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_mismatched_shapes() -> None:
    model = LinearRegression()

    X = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
        ]
    )

    y = np.array(
        [
            1.0,
            2.0,
        ]
    )

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_invalid_learning_rate_type() -> None:
    with pytest.raises(TypeError):
        LinearRegression(learning_rate="0.1")


def test_invalid_learning_rate_value() -> None:
    with pytest.raises(ValueError):
        LinearRegression(learning_rate=-1.0)


def test_invalid_epochs_type() -> None:
    with pytest.raises(TypeError):
        LinearRegression(epochs=1.5)


def test_invalid_epochs_value() -> None:
    with pytest.raises(ValueError):
        LinearRegression(epochs=0)
