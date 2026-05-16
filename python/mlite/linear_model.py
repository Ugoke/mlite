from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._core import LinearRegression as _LinearRegression  # type: ignore

DEFAULT_LEARNING_RATE = 0.01
DEFAULT_EPOCHS = 1000


class LinearRegression:
    """
    Linear Regression model.

    Parameters
    ----------
    learning_rate : float
        Gradient descent learning rate.

    epochs : int
        Number of training iterations.
    """

    def __init__(
        self, learning_rate: float = DEFAULT_LEARNING_RATE, epochs: int = DEFAULT_EPOCHS
    ) -> None:
        self._model = _LinearRegression()

        self.learning_rate = self._validate_learning_rate(learning_rate)
        self.epochs = self._validate_epochs(epochs)

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """
        Train the model.

        Parameters
        ----------
        X : np.ndarray
            Training features with shape (n_samples, n_features)

        y : np.ndarray
            Target values with shape (n_samples,)
        """

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("X must be 2D")

        if y.ndim != 1:
            raise ValueError("y must be 1D")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")

        self._model.fit(
            X,
            y,
            self.learning_rate,
            self.epochs,
        )

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predict values.

        Parameters
        ----------
        X : np.ndarray
            Input matrix.

        Returns
        -------
        np.ndarray
            Predictions.
        """

        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("X must be 2D")

        return self._model.predict(X)

    @staticmethod
    def _validate_learning_rate(learning_rate: float) -> float:
        if not isinstance(learning_rate, float):
            raise TypeError("learning_rate must be float")

        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        return learning_rate

    @staticmethod
    def _validate_epochs(epochs: int) -> int:
        if not isinstance(epochs, int):
            raise TypeError("epochs must be int")

        if epochs <= 0:
            raise ValueError("epochs must be positive")

        return epochs
