import numpy as np
import pytest

from mlite.preprocessing import OrdinalEncoder


def test_fit_transform_strings():
    X = np.array([
        ["red", "S"],
        ["blue", "M"],
        ["red", "M"],
    ])
    encoder = OrdinalEncoder()

    result = encoder.fit_transform(X)

    expected = np.array([
        [1.0, 1.0],
        [0.0, 0.0],
        [1.0, 0.0],
    ])
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.float64
    assert encoder.n_features_in_ == 2
    np.testing.assert_array_equal(encoder.categories_[0], ["blue", "red"])
    np.testing.assert_array_equal(encoder.categories_[1], ["M", "S"])


def test_one_dimensional_input_becomes_one_feature():
    encoder = OrdinalEncoder()

    result = encoder.fit_transform(np.array([3, 1, 3]))

    np.testing.assert_array_equal(result, [[1.0], [0.0], [1.0]])
    assert encoder.n_features_in_ == 1


def test_fit_returns_none():
    encoder = OrdinalEncoder()

    assert encoder.fit([["a"], ["b"]]) is None


def test_transform_rejects_unknown_category_by_default():
    encoder = OrdinalEncoder()
    encoder.fit([["red"], ["blue"]])

    with pytest.raises(ValueError, match="unknown category"):
        encoder.transform([["green"]])


def test_transform_uses_configured_unknown_value():
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-7.0)
    encoder.fit([["red"], ["blue"]])

    result = encoder.transform([["green"], ["red"]])

    np.testing.assert_array_equal(result, [[-7.0], [1.0]])


def test_inverse_transform_round_trip():
    X = np.array([
        ["red", "S"],
        ["blue", "M"],
    ])
    encoder = OrdinalEncoder()

    encoded = encoder.fit_transform(X)
    decoded = encoder.inverse_transform(encoded)

    np.testing.assert_array_equal(decoded, X)


def test_inverse_transform_unknown_value_returns_none():
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1.0)
    encoder.fit([["red"], ["blue"]])

    decoded = encoder.inverse_transform([[-1.0], [1.0]])

    assert decoded[0, 0] is None
    assert decoded[1, 0] == "red"


def test_inverse_transform_rejects_invalid_code():
    encoder = OrdinalEncoder()
    encoder.fit([["red"], ["blue"]])

    with pytest.raises(ValueError, match="invalid ordinal code"):
        encoder.inverse_transform([[2.0]])


def test_unknown_value_must_not_collide_with_known_codes():
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=0.0)

    with pytest.raises(ValueError, match="collides"):
        encoder.fit([["red"], ["blue"]])


def test_state_dict_round_trip():
    X = np.array([
        ["red", 1],
        ["blue", 2],
    ], dtype=object)
    original = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-7.0)
    expected = original.fit_transform(X)

    restored = OrdinalEncoder()
    restored.load_state_dict(original.state_dict())

    np.testing.assert_array_equal(restored.transform(X), expected)
    assert restored.handle_unknown == "use_encoded_value"
    assert restored.unknown_value == -7.0


def test_categories_property_returns_copies():
    encoder = OrdinalEncoder()
    encoder.fit([["a"], ["b"]])

    categories = encoder.categories_
    categories[0][0] = "changed"

    assert encoder.categories_[0][0] == "a"


@pytest.mark.parametrize("attribute", ["categories_", "n_features_in_"])
def test_learned_attributes_require_fit(attribute):
    encoder = OrdinalEncoder()

    with pytest.raises(RuntimeError):
        getattr(encoder, attribute)


def test_transform_requires_fit():
    with pytest.raises(RuntimeError):
        OrdinalEncoder().transform([["a"]])


def test_transform_rejects_feature_mismatch():
    encoder = OrdinalEncoder()
    encoder.fit([["a", "b"]])

    with pytest.raises(ValueError, match="different number of features"):
        encoder.transform([["a"]])


@pytest.mark.parametrize("X", [np.array([]), np.empty((0, 2)), np.empty((2, 0))])
def test_fit_rejects_empty_input(X):
    with pytest.raises(ValueError):
        OrdinalEncoder().fit(X)


def test_fit_rejects_three_dimensional_input():
    with pytest.raises(ValueError, match="1D or 2D"):
        OrdinalEncoder().fit(np.zeros((1, 1, 1)))


def test_fit_rejects_missing_values():
    with pytest.raises(ValueError, match="missing values"):
        OrdinalEncoder().fit(np.array([["a"], [np.nan]], dtype=object))


def test_invalid_handle_unknown():
    with pytest.raises(ValueError, match="handle_unknown"):
        OrdinalEncoder(handle_unknown="ignore")


def test_unknown_value_must_be_finite():
    with pytest.raises(ValueError, match="finite"):
        OrdinalEncoder(unknown_value=np.nan)
