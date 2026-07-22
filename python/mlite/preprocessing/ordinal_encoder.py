from __future__ import annotations

from typing import Any, List

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._core import OrdinalEncoder as _OrdinalEncoder  # type: ignore
from ..utils._categorical import _build_category_maps, _decode_categories, _discover_categories, _encode_categories, _native_categories
from ..utils._decorators import _ensure_fitted
from ..utils._validators import _validate_categorical_X, _validate_encoded_X, _validate_handle_unknown, _validate_state_categories, _validate_unknown_value, _validate_unknown_value_does_not_collide


class OrdinalEncoder:
    """Encode each categorical feature with integer-valued ``float64`` codes."""

    def __init__(self, handle_unknown: str = "error", unknown_value: float = -1.0) -> None:
        self.handle_unknown = _validate_handle_unknown(handle_unknown, ("error", "use_encoded_value"))
        self.unknown_value = _validate_unknown_value(unknown_value)
        self._model = _OrdinalEncoder()
        self._categories: List[NDArray[Any]] = []
        self._category_maps = []
        self._fitted = False

    def fit(self, X: ArrayLike) -> None:
        """Learn the sorted categories of every input feature."""

        X = _validate_categorical_X(X)
        categories = _discover_categories(X)

        if self.handle_unknown == "use_encoded_value":
            _validate_unknown_value_does_not_collide(self.unknown_value, categories)

        category_maps = _build_category_maps(categories)
        encoded = _encode_categories(X, category_maps, allow_unknown=False)

        self._model.fit(encoded)
        self._categories = categories
        self._category_maps = category_maps
        self._fitted = True

    @_ensure_fitted
    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Replace categories with their zero-based ordinal codes."""

        X = _validate_categorical_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has different number of features")

        use_encoded_unknown = self.handle_unknown == "use_encoded_value"
        encoded = _encode_categories(X, self._category_maps, allow_unknown=use_encoded_unknown)

        return np.asarray(
            self._model.transform(
                encoded,
                use_encoded_unknown,
                self.unknown_value,
            ),
            dtype=np.float64,
        )

    def fit_transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Learn categories and transform ``X`` in one call."""

        self.fit(X)
        return self.transform(X)

    @_ensure_fitted
    def inverse_transform(self, X: ArrayLike) -> NDArray[Any]:
        """Convert ordinal codes back to the original categories."""

        X = _validate_encoded_X(X, self.n_features_in_, "X")
        use_encoded_unknown = self.handle_unknown == "use_encoded_value"
        tokens = np.asarray(
            self._model.inverse_transform(
                X,
                use_encoded_unknown,
                self.unknown_value,
            ),
            dtype=np.float64,
        )
        return _decode_categories(tokens, self._categories)

    @_ensure_fitted
    def state_dict(self) -> dict:
        """Return all state required to restore the fitted encoder."""

        return {
            "categories": [categories.copy() for categories in self._categories],
            "handle_unknown": self.handle_unknown,
            "unknown_value": self.unknown_value,
            "n_features_in": self.n_features_in_,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore a fitted encoder from ``state_dict`` output."""

        if not isinstance(state, dict):
            raise TypeError("state must be dict")

        required = {
            "categories",
            "handle_unknown",
            "unknown_value",
            "n_features_in",
        }
        missing = required.difference(state)
        if missing:
            raise KeyError(f"Missing state keys: {', '.join(sorted(missing))}")

        categories = _validate_state_categories(state["categories"])
        n_features_in = int(state["n_features_in"])
        if n_features_in != len(categories):
            raise ValueError("n_features_in does not match categories")

        handle_unknown = _validate_handle_unknown(state["handle_unknown"], ("error", "use_encoded_value"))
        unknown_value = _validate_unknown_value(state["unknown_value"])
        if handle_unknown == "use_encoded_value":
            _validate_unknown_value_does_not_collide(unknown_value, categories)

        self._model.load_state(_native_categories(categories))
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self._categories = categories
        self._category_maps = _build_category_maps(categories)
        self._fitted = True

    @property
    @_ensure_fitted
    def categories_(self) -> List[NDArray[Any]]:
        return [categories.copy() for categories in self._categories]

    @property
    @_ensure_fitted
    def n_features_in_(self) -> int:
        return int(self._model.n_features_in_)
