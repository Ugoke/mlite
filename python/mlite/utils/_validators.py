from numbers import Real
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate_X(X: NDArray[np.float64]) -> NDArray[np.float64]:
    X = np.asarray(X, dtype=np.float64)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if X.ndim != 2:
        raise ValueError("X must be 2D")
    
    if X.shape[0] == 0:
        raise ValueError("X cannot be empty")
    
    return X


def _validate_y(y: NDArray[np.float64]) -> NDArray[np.float64]:
    y = np.asarray(y, dtype=np.float64)

    if y.ndim != 1:
        raise ValueError("y must be 1D")
    
    if y.shape[0] == 0:
        raise ValueError("y cannot be empty")
    
    return y


def _validate_learning_rate(learning_rate: float) -> float:
    if not isinstance(learning_rate, float):
        raise TypeError("learning_rate must be float")
    
    if learning_rate <= 0 or not np.isfinite(learning_rate):
        raise ValueError("learning_rate must be finite and positive")
    
    return learning_rate


def _validate_epochs(epochs: int) -> int:
    if not isinstance(epochs, int):
        raise TypeError("epochs must be int")
    
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    
    return epochs


def _validate_test_size(test_size: float) -> float:
    if not isinstance(test_size, float):
        raise TypeError("test_size must be float")

    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be in range (0, 1)")
    
    return test_size


def _validate_random_state(random_state: int) -> int:
    if not isinstance(random_state, int):
        raise TypeError("random_state must be int")
    
    if random_state < 0:
        raise ValueError("random_state must be non-negative")
    
    return random_state


def _validate_n_splits(n_splits: int) -> int:
    if type(n_splits) is not int:
        raise TypeError("n_splits must be int")

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    return n_splits


def _validate_shuffle(shuffle: bool) -> bool:
    if type(shuffle) is not bool:
        raise TypeError("shuffle must be bool")

    return shuffle


def _validate_categorical_X(X: ArrayLike) -> NDArray[Any]:
    array = np.asarray(X)

    if array.ndim == 1:
        array = array.reshape(-1, 1)

    if array.ndim != 2:
        raise ValueError("X must be 1D or 2D")

    if array.shape[0] == 0:
        raise ValueError("X cannot be empty")

    if array.shape[1] == 0:
        raise ValueError("X must contain at least one feature")

    return array


def _validate_state_categories(value: Any) -> list[NDArray[Any]]:
    from ._categorical import _category_array, _category_key, _normalize_category

    if not isinstance(value, (list, tuple)) or len(value) == 0:
        raise ValueError("state['categories'] must be a non-empty list")

    categories: list[NDArray[Any]] = []

    for feature, raw_categories in enumerate(value):
        array = np.asarray(raw_categories)
        if array.ndim != 1 or array.size == 0:
            raise ValueError(f"state categories for feature {feature} must be a non-empty 1D array")

        normalized = [_normalize_category(item) for item in array]
        keys = [_category_key(item) for item in normalized]
        if len(set(keys)) != len(keys):
            raise ValueError(f"state categories for feature {feature} must be unique")

        categories.append(_category_array(normalized))

    return categories


def _validate_encoded_X(X: ArrayLike, expected_features: int, description: str) -> NDArray[np.float64]:
    try:
        array = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{description} must contain numeric values") from error

    if array.ndim != 2:
        raise ValueError(f"{description} must be 2D")

    if array.shape[0] == 0:
        raise ValueError(f"{description} cannot be empty")

    if array.shape[1] != expected_features:
        raise ValueError(f"{description} has different number of features")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{description} must contain only finite values")

    return np.ascontiguousarray(array)


def _validate_handle_unknown(value: Any, allowed: Sequence[str]) -> str:
    if not isinstance(value, str):
        raise TypeError("handle_unknown must be str")

    if value not in allowed:
        choices = ", ".join(repr(choice) for choice in allowed)
        raise ValueError(f"handle_unknown must be one of: {choices}")

    return value


def _validate_unknown_value(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("unknown_value must be a real number")

    result = float(value)
    if not np.isfinite(result):
        raise ValueError("unknown_value must be finite")

    return result


def _validate_unknown_value_does_not_collide(unknown_value: float, categories: Sequence[NDArray[Any]]) -> None:
    for feature, feature_categories in enumerate(categories):
        if (unknown_value >= 0.0 and unknown_value.is_integer() and int(unknown_value) < len(feature_categories)):
            raise ValueError(f"unknown_value collides with an encoded category in feature {feature}")