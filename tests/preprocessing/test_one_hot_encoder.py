import numpy as np
import pytest

from mlite.preprocessing import OneHotEncoder


def test_fit_transform_strings():
    X = np.array([
        ["red", "S"],
        ["blue", "M"],
        ["red", "M"],
    ])
    encoder = OneHotEncoder()

    result = encoder.fit_transform(X)

    expected = np.array([
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
    ])
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.float64
    assert encoder.n_features_in_ == 2
    assert encoder.n_features_out_ == 4
    np.testing.assert_array_equal(encoder.categories_[0], ["blue", "red"])
    np.testing.assert_array_equal(encoder.categories_[1], ["M", "S"])


def test_one_dimensional_input_becomes_one_feature():
    encoder = OneHotEncoder()

    result = encoder.fit_transform(np.array([3, 1, 3]))

    np.testing.assert_array_equal(result, [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    assert encoder.n_features_in_ == 1


def test_fit_returns_none():
    encoder = OneHotEncoder()

    assert encoder.fit([["a"], ["b"]]) is None


def test_transform_rejects_unknown_category_by_default():
    encoder = OneHotEncoder()
    encoder.fit([["red"], ["blue"]])

    with pytest.raises(ValueError, match="unknown category"):
        encoder.transform([["green"]])


def test_transform_can_ignore_unknown_category():
    encoder = OneHotEncoder(handle_unknown="ignore")
    encoder.fit([
        ["red", "S"],
        ["blue", "M"],
    ])

    result = encoder.transform([["green", "S"]])

    np.testing.assert_array_equal(result, [[0.0, 0.0, 0.0, 1.0]])


def test_inverse_transform_round_trip():
    X = np.array([
        ["red", "S"],
        ["blue", "M"],
    ])
    encoder = OneHotEncoder()

    encoded = encoder.fit_transform(X)
    decoded = encoder.inverse_transform(encoded)

    np.testing.assert_array_equal(decoded, X)


def test_inverse_transform_unknown_group_returns_none_when_ignored():
    encoder = OneHotEncoder(handle_unknown="ignore")
    encoder.fit([["red"], ["blue"]])

    decoded = encoder.inverse_transform([[0.0, 0.0]])

    assert decoded.shape == (1, 1)
    assert decoded[0, 0] is None


def test_inverse_transform_unknown_group_raises_by_default():
    encoder = OneHotEncoder()
    encoder.fit([["red"], ["blue"]])

    with pytest.raises(ValueError, match="unknown one-hot category"):
        encoder.inverse_transform([[0.0, 0.0]])


def test_state_dict_round_trip():
    X = np.array([
        ["red", 1],
        ["blue", 2],
    ], dtype=object)
    original = OneHotEncoder(handle_unknown="ignore")
    expected = original.fit_transform(X)

    restored = OneHotEncoder()
    restored.load_state_dict(original.state_dict())

    np.testing.assert_array_equal(restored.transform(X), expected)
    assert restored.handle_unknown == "ignore"
    assert restored.n_features_in_ == 2


def test_categories_property_returns_copies():
    encoder = OneHotEncoder()
    encoder.fit([["a"], ["b"]])

    categories = encoder.categories_
    categories[0][0] = "changed"

    assert encoder.categories_[0][0] == "a"


@pytest.mark.parametrize("attribute", ["categories_", "n_features_in_", "n_features_out_"])
def test_learned_attributes_require_fit(attribute):
    encoder = OneHotEncoder()

    with pytest.raises(RuntimeError):
        getattr(encoder, attribute)


def test_transform_requires_fit():
    with pytest.raises(RuntimeError):
        OneHotEncoder().transform([["a"]])


def test_transform_rejects_feature_mismatch():
    encoder = OneHotEncoder()
    encoder.fit([["a", "b"]])

    with pytest.raises(ValueError, match="different number of features"):
        encoder.transform([["a"]])


@pytest.mark.parametrize("X", [np.array([]), np.empty((0, 2)), np.empty((2, 0))])
def test_fit_rejects_empty_input(X):
    with pytest.raises(ValueError):
        OneHotEncoder().fit(X)


def test_fit_rejects_three_dimensional_input():
    with pytest.raises(ValueError, match="1D or 2D"):
        OneHotEncoder().fit(np.zeros((1, 1, 1)))


def test_fit_rejects_missing_values():
    with pytest.raises(ValueError, match="missing values"):
        OneHotEncoder().fit(np.array([["a"], [None]], dtype=object))


def test_invalid_handle_unknown():
    with pytest.raises(ValueError, match="handle_unknown"):
        OneHotEncoder(handle_unknown="encode")
