import numpy as np
import pytest

from mlite.linear_models import LogisticRegression


@pytest.fixture
def binary_dataset():
    X = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0],
        [5.0, 6.0],
        [6.0, 7.0],
    ], dtype=np.float64)

    y = np.array([
        0,
        0,
        0,
        1,
        1,
        1,
    ], dtype=np.float64)

    return X, y


def test_default_initialization():
    model = LogisticRegression()

    assert model.learning_rate > 0
    assert model.epochs > 0


def test_custom_initialization():
    model = LogisticRegression(learning_rate=0.01, epochs=500)

    assert model.learning_rate == 0.01
    assert model.epochs == 500


def test_fit(binary_dataset):
    X, y = binary_dataset

    model = LogisticRegression()
    model.fit(X, y)

    assert model._fitted is True


def test_fit_invalid_sample_count():
    X = np.array([[1.0], [2.0]])
    y = np.array([1.0])

    model = LogisticRegression()

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_fit_invalid_labels():
    X = np.array([
        [1.0],
        [2.0],
        [3.0],
    ])

    y = np.array([
        0.0,
        2.0,
        1.0,
    ])

    model = LogisticRegression()

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_predict(binary_dataset):
    X, y = binary_dataset

    model = LogisticRegression(learning_rate=0.01, epochs=5000)
    model.fit(X, y)

    predictions = model.predict(X)

    assert predictions.shape == (6,)
    assert predictions.dtype == np.int64


def test_predict_values(binary_dataset):
    X, y = binary_dataset

    model = LogisticRegression(learning_rate=0.01, epochs=5000)
    model.fit(X, y)

    predictions = model.predict(X)

    assert np.all(np.isin(predictions, [0, 1]))


def test_predict_before_fit(binary_dataset):
    X, _ = binary_dataset

    model = LogisticRegression()

    with pytest.raises(RuntimeError):
        model.predict(X)


def test_predict_proba(binary_dataset):
    X, y = binary_dataset

    model = LogisticRegression()
    model.fit(X, y)

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (6,)


def test_predict_proba_range(binary_dataset):
    X, y = binary_dataset

    model = LogisticRegression()
    model.fit(X, y)

    probabilities = model.predict_proba(X)

    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)


def test_score(binary_dataset):
    X, y = binary_dataset

    model = LogisticRegression(learning_rate=0.01, epochs=5000)
    model.fit(X, y)

    score = model.score(X, y)

    assert isinstance(score, float)


def test_score_range(binary_dataset):
    X, y = binary_dataset

    model = LogisticRegression()
    model.fit(X, y)

    score = model.score(X, y)

    assert 0.0 <= score <= 1.0


def test_score_high_accuracy(binary_dataset):
    X, y = binary_dataset

    model = LogisticRegression(learning_rate=0.01, epochs=10000)
    model.fit(X, y)

    score = model.score(X, y)

    assert score > 0.9


def test_state_dict(binary_dataset):
    X, y = binary_dataset

    model = LogisticRegression()
    model.fit(X, y)

    state = model.state_dict()

    assert isinstance(state, dict)

    assert "coef" in state
    assert "intercept" in state
    assert "n_features_in" in state
    assert "learning_rate" in state
    assert "epochs" in state


def test_load_state_dict(binary_dataset):
    X, y = binary_dataset

    model1 = LogisticRegression()
    model1.fit(X, y)

    state = model1.state_dict()

    model2 = LogisticRegression()
    model2.load_state_dict(state)

    pred1 = model1.predict(X)
    pred2 = model2.predict(X)

    assert np.array_equal(pred1, pred2)


def test_coef_property(binary_dataset):
    X, y = binary_dataset

    model = LogisticRegression()
    model.fit(X, y)
    coef = model.coef_

    assert isinstance(coef, np.ndarray)


def test_intercept_property(binary_dataset):
    X, y = binary_dataset

    model = LogisticRegression()
    model.fit(X, y)
    intercept = model.intercept_

    assert isinstance(intercept, float)


def test_n_features_property(binary_dataset):
    X, y = binary_dataset

    model = LogisticRegression()
    model.fit(X, y)

    assert model.n_features_in_ == 2


def test_invalid_learning_rate():
    with pytest.raises(ValueError):
        LogisticRegression(learning_rate=-1.0)


def test_invalid_epochs():
    with pytest.raises(ValueError):
        LogisticRegression(epochs=0)


def test_invalid_threshold(binary_dataset):
    X, y = binary_dataset

    model = LogisticRegression()
    model.fit(X, y)

    with pytest.raises(ValueError):
        model.predict(X, threshold=2.0)


def test_empty_X():
    X = np.array([], dtype=np.float64).reshape(0, 1)
    y = np.array([], dtype=np.float64)

    model = LogisticRegression()

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_invalid_X_dimensions():
    X = np.array([[[1.0]]])
    y = np.array([1.0])

    model = LogisticRegression()

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_invalid_y_dimensions():
    X = np.array([[1.0]])
    y = np.array([[1.0]])

    model = LogisticRegression()

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_predict_feature_mismatch(binary_dataset):
    X, y = binary_dataset

    model = LogisticRegression()
    model.fit(X, y)

    X_bad = np.array([
        [1.0, 2.0, 3.0]
    ])

    with pytest.raises(ValueError):
        model.predict(X_bad)


def test_score_feature_mismatch(binary_dataset):
    X, y = binary_dataset

    model = LogisticRegression()
    model.fit(X, y)

    X_bad = np.array([
        [1.0, 2.0, 3.0]
    ])

    y_bad = np.array([1.0])

    with pytest.raises(ValueError):
        model.score(X_bad, y_bad)