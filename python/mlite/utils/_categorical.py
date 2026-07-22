from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


CategoryKey = Tuple[type, Any]
CategoryMap = Dict[CategoryKey, int]


def _normalize_category(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()

    if value is None:
        raise ValueError("X cannot contain missing values")

    if isinstance(value, (float, complex)) and np.isnan(value):
        raise ValueError("X cannot contain missing values")

    try:
        hash(value)
    except TypeError as error:
        raise TypeError("Categories must be scalar and hashable") from error

    return value


def _category_key(value: Any) -> CategoryKey:
    normalized = _normalize_category(value)
    return type(normalized), normalized


def _category_array(values: Sequence[Any]) -> NDArray[Any]:
    value_types = {type(value) for value in values}

    if len(value_types) == 1:
        return np.asarray(values)

    return np.asarray(values, dtype=object)


def _discover_categories(X: NDArray[Any]) -> List[NDArray[Any]]:
    categories: List[NDArray[Any]] = []

    for feature in range(X.shape[1]):
        unique: Dict[CategoryKey, Any] = {}

        for raw_value in X[:, feature]:
            value = _normalize_category(raw_value)
            unique.setdefault(_category_key(value), value)

        try:
            ordered = sorted(unique.values())
        except TypeError as error:
            raise TypeError(f"Categories in feature {feature} must be mutually comparable") from error

        categories.append(_category_array(ordered))

    return categories


def _build_category_maps(categories: Sequence[NDArray[Any]]) -> List[CategoryMap]:
    return [
        {_category_key(value): index for index, value in enumerate(feature_categories)}
        for feature_categories in categories
    ]


def _encode_categories(X: NDArray[Any], category_maps: Sequence[CategoryMap], allow_unknown: bool) -> NDArray[np.float64]:
    encoded = np.empty(X.shape, dtype=np.float64)

    for feature, category_map in enumerate(category_maps):
        for sample, raw_value in enumerate(X[:, feature]):
            key = _category_key(raw_value)
            category = category_map.get(key)

            if category is None:
                if not allow_unknown:
                    raise ValueError(f"Found unknown category {raw_value!r} in feature {feature}")
                encoded[sample, feature] = -1.0
            else:
                encoded[sample, feature] = float(category)

    return encoded


def _native_categories(categories: Sequence[NDArray[Any]]) -> List[List[float]]:
    return [
        [float(index) for index in range(len(feature_categories))]
        for feature_categories in categories
    ]


def _decode_categories(tokens: NDArray[np.float64], categories: Sequence[NDArray[Any]]) -> NDArray[Any]:
    decoded = np.empty(tokens.shape, dtype=object)
    has_unknown = False

    for feature, feature_categories in enumerate(categories):
        for sample, token in enumerate(tokens[:, feature]):
            if np.isnan(token):
                decoded[sample, feature] = None
                has_unknown = True
                continue

            category = int(token)
            if token != category or category < 0 or category >= len(feature_categories):
                raise RuntimeError("Native encoder returned an invalid category code")

            decoded[sample, feature] = feature_categories[category]

    if has_unknown:
        return decoded

    dtypes = [feature_categories.dtype for feature_categories in categories]
    if dtypes and all(dtype == dtypes[0] for dtype in dtypes):
        try:
            return decoded.astype(dtypes[0])
        except (TypeError, ValueError):
            pass

    return decoded
