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

        self._fitted = False

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

        X = self._validate_X(X)
        y = self._validate_y(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")

        self._model.fit(
            X,
            y,
            self.learning_rate,
            self.epochs,
        )

        self._fitted = True

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

        self._check_is_fitted()

        X = self._validate_X(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has different number of features")

        return self._model.predict(X)

    def score(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> float:
        """
        Compute R² score.

        Parameters
        ----------
        X : np.ndarray
            Test samples.

        y : np.ndarray
            True values.

        Returns
        -------
        float
            R² score.
        """

        self._check_is_fitted()

        X = self._validate_X(X)
        y = self._validate_y(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has different number of features")

        return self._model.score(X, y)

    def state_dict(self) -> dict:
        """
        Return model state as a dictionary.
    
        This method extracts all parameters required to fully
        reconstruct the trained model.
    
        Returns
        -------
        dict
            Dictionary containing model parameters:
            - coef: learned weights
            - intercept: bias term
            - n_features_in: number of input features
            - learning_rate: training learning rate
            - epochs: number of training epochs
        """
    
        self._check_is_fitted()
    
        return {
            "coef": self.coef_,
            "intercept": self.intercept_,
            "n_features_in": self.n_features_in_,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
        }
    
    
    def load_state_dict(self, state: dict) -> None:
        """
        Load model state from a dictionary.
    
        This method restores a previously saved model state.
        It overwrites internal parameters and marks the model
        as fitted.
    
        Parameters
        ----------
        state : dict
            Dictionary containing model parameters. Expected keys:
            - coef
            - intercept
            - n_features_in
            - learning_rate
            - epochs
        """

        self._model.load_state(
            np.asarray(state["coef"], dtype=np.float64),
            float(state["intercept"]),
            int(state["n_features_in"]),
        )

        self.learning_rate = float(state["learning_rate"])
        self.epochs = int(state["epochs"])

        self._fitted = True

    @property
    def coef_(self) -> NDArray[np.float64]:
        self._check_is_fitted()
        return np.asarray(self._model.coef_)

    @property
    def intercept_(self) -> float:
        self._check_is_fitted()
        return float(self._model.intercept_)

    @property
    def n_features_in_(self) -> int:
        self._check_is_fitted()
        return int(self._model.n_features_in_)

    def _check_is_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model is not fitted")

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

        if learning_rate <= 0 or not np.isfinite(learning_rate):
            raise ValueError("learning_rate must be finite and positive")

        return learning_rate

    @staticmethod
    def _validate_epochs(epochs: int) -> int:
        if not isinstance(epochs, int):
            raise TypeError("epochs must be int")

        if epochs <= 0:
            raise ValueError("epochs must be positive")

        return epochs
