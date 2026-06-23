import numpy as np
import pytest

from mlite.linear_models import LinearRegression


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


def test_score_returns_r2() -> None:
    model = LinearRegression()

    X = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
        ]
    )

    y = np.array([2.0, 4.0, 6.0, 8.0])

    model.fit(X, y)

    score = model.score(X, y)

    assert 0.99 <= score <= 1.0


def test_model_attributes_after_fit() -> None:
    model = LinearRegression()

    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 6.0])

    model.fit(X, y)

    assert hasattr(model, "coef_")
    assert hasattr(model, "intercept_")
    assert hasattr(model, "n_features_in_")

    assert len(model.coef_) == 1
    assert model.n_features_in_ == 1


def test_predict_before_fit_raises() -> None:
    model = LinearRegression()

    with pytest.raises(Exception):
        model.predict(np.array([[1.0]]))


def test_multifeature_regression() -> None:
    model = LinearRegression()

    X = np.array(
        [
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
        ]
    )

    y = np.array([3.0, 5.0, 7.0])

    model.fit(X, y)

    pred = model.predict(np.array([[4.0, 5.0]]))

    assert pred.shape == (1,)


def test_1d_input_normalization() -> None:
    model = LinearRegression()

    X = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])

    model.fit(X, y)

    pred = model.predict(np.array([4.0, 5.0, 6.0]))

    assert pred.shape == (3,)


def test_predict_is_deterministic() -> None:
    model = LinearRegression()

    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 6.0])

    model.fit(X, y)

    p1 = model.predict(np.array([[4.0], [5.0]]))
    p2 = model.predict(np.array([[4.0], [5.0]]))

    assert np.allclose(p1, p2)


def test_fit_empty_dataset_raises() -> None:
    model = LinearRegression()

    with pytest.raises(ValueError):
        model.fit(np.empty((0, 1)), np.empty((0,)))


def test_predict_feature_mismatch_raises() -> None:
    model = LinearRegression()

    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 6.0])

    model.fit(X, y)

    with pytest.raises(ValueError):
        model.predict(np.array([[1.0, 2.0]]))


def test_predict_invalid_dimension_raises() -> None:
    model = LinearRegression()

    X = np.array([[1.0], [2.0]])
    y = np.array([2.0, 4.0])

    model.fit(X, y)

    with pytest.raises(ValueError):
        model.predict(np.array([[[1.0]]]))


def test_score_constant_target_returns_finite() -> None:
    model = LinearRegression()

    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([5.0, 5.0, 5.0])

    model.fit(X, y)

    score = model.score(X, y)

    assert np.isfinite(score)


def test_learning_rate_nan_rejected() -> None:
    with pytest.raises(ValueError, match="learning_rate must be finite and positive"):
        LinearRegression(learning_rate=np.nan, epochs=100)


def test_state_dict():
    X = np.array([[1], [2], [3], [4]], dtype=np.float64)
    y = np.array([2, 4, 6, 8], dtype=np.float64)

    model = LinearRegression()
    model.fit(X, y)

    state = model.state_dict()

    assert "coef" in state
    assert "intercept" in state
    assert "n_features_in" in state
    assert "learning_rate" in state
    assert "epochs" in state

    assert isinstance(state["coef"], np.ndarray)


def test_load_state_dict():
    model1 = LinearRegression()
    X = np.array([[1], [2], [3], [4]], dtype=np.float64)
    y = np.array([2, 4, 6, 8], dtype=np.float64)

    model1.fit(X, y)
    state = model1.state_dict()

    model2 = LinearRegression()
    model2.load_state_dict(state)

    preds1 = model1.predict(X)
    preds2 = model2.predict(X)

    assert np.allclose(preds1, preds2)