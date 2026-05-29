from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .._core import LogisticRegression as _LogisticRegression  # type: ignore
from ..utils._decorators import _ensure_fitted
from ..utils._constants import DEFAULT_LEARNING_RATE, DEFAULT_EPOCHS


class LogisticRegression:
    """
    Logistic Regression classifier.

    Parameters
    ----------
    learning_rate : float
        Optimizer learning rate.

    epochs : int
        Number of training epochs.
    """

    def __init__(self, learning_rate: float = DEFAULT_LEARNING_RATE, epochs: int = DEFAULT_EPOCHS) -> None:
        self._model = _LogisticRegression()
        self.learning_rate = self._validate_learning_rate(learning_rate)
        self.epochs = self._validate_epochs(epochs)
        self._fitted = False

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """
        Train classifier.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix with shape
            (n_samples, n_features)

        y : np.ndarray
            Binary labels with shape
            (n_samples,)
        """

        X = self._validate_X(X)
        y = self._validate_y(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")

        unique = np.unique(y)

        if not np.all(np.isin(unique, [0.0, 1.0])):
            raise ValueError("y must contain only binary labels 0 and 1")

        self._model.fit(X, y, self.learning_rate, self.epochs)
        self._fitted = True

    @_ensure_fitted
    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predict probabilities.

        Parameters
        ----------
        X : np.ndarray
            Input matrix.

        Returns
        -------
        np.ndarray
            Probabilities in range [0, 1].
        """

        X = self._validate_X(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has different number of features")

        return np.asarray(self._model.predict_proba(X), dtype=np.float64)

    @_ensure_fitted
    def predict(self, X: NDArray[np.float64], threshold: float = 0.5) -> NDArray[np.int64]:
        """
        Predict binary classes.

        Parameters
        ----------
        X : np.ndarray
            Input matrix.

        threshold : float
            Decision threshold.

        Returns
        -------
        np.ndarray
            Predicted classes.
        """

        X = self._validate_X(X)
        threshold = self._validate_threshold(threshold)

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has different number of features")

        return np.asarray(self._model.predict(X, threshold), dtype=np.int64)

    @_ensure_fitted
    def score(self, X: NDArray[np.float64], y: NDArray[np.float64], threshold: float = 0.5) -> float:
        """
        Compute classification accuracy.

        Parameters
        ----------
        X : np.ndarray
            Test samples.

        y : np.ndarray
            True labels.

        threshold : float
            Decision threshold.

        Returns
        -------
        float
            Accuracy score.
        """

        X = self._validate_X(X)
        y = self._validate_y(y)

        threshold = self._validate_threshold(threshold)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has different number of features")

        return float(self._model.score(X, y, threshold))

    @_ensure_fitted
    def state_dict(self) -> dict:
        """
        Return model state dictionary.

        Returns
        -------
        dict
            Serialized model state.
        """

        return {
            "coef": self.coef_,
            "intercept": self.intercept_,
            "n_features_in": self.n_features_in_,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
        }

    def load_state_dict(self, state: dict) -> None:
        """
        Restore model from state dictionary.

        Parameters
        ----------
        state : dict
            Serialized model state.
        """

        self._model.load_state(
            np.asarray(state["coef"],dtype=np.float64), float(state["intercept"]), int(state["n_features_in"])
        )

        self.learning_rate = float(state["learning_rate"])
        self.epochs = int(state["epochs"])
        self._fitted = True

    @property
    @_ensure_fitted
    def coef_(self) -> NDArray[np.float64]:
        return np.asarray(self._model.coef_, dtype=np.float64)

    @property
    @_ensure_fitted
    def intercept_(self) -> float:
        return float(self._model.intercept_)

    @property
    @_ensure_fitted
    def n_features_in_(self) -> int:
        return int(self._model.n_features_in_)

    @staticmethod
    def _validate_X(X: NDArray[np.float64]) -> NDArray[np.float64]:
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("X must be 2D")

        if X.shape[0] == 0:
            raise ValueError("X cannot be empty")

        return X

    @staticmethod
    def _validate_y(y: NDArray[np.float64]) -> NDArray[np.float64]:
        y = np.asarray(y, dtype=np.float64)

        if y.ndim != 1:
            raise ValueError("y must be 1D")

        if y.shape[0] == 0:
            raise ValueError("y cannot be empty")

        return y

    @staticmethod
    def _validate_learning_rate(learning_rate: float) -> float:
        if not isinstance(learning_rate, float):
            raise TypeError("learning_rate must be float")

        if (learning_rate <= 0 or not np.isfinite(learning_rate)):
            raise ValueError("learning_rate must be finite and positive")

        return learning_rate

    @staticmethod
    def _validate_epochs(epochs: int) -> int:
        if not isinstance(epochs, int):
            raise TypeError("epochs must be int")

        if epochs <= 0:
            raise ValueError("epochs must be positive")

        return epochs

    @staticmethod
    def _validate_threshold(threshold: float,) -> float:
        if not isinstance(threshold, float):
            raise TypeError("threshold must be float")

        if (threshold < 0.0 or threshold > 1.0 or not np.isfinite(threshold)):
            raise ValueError("threshold must be in range [0, 1]")

        return threshold