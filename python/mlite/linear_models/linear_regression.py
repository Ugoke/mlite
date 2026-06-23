from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .._core import LinearRegression as _LinearRegression  # type: ignore
from ..utils._decorators import _ensure_fitted
from ..utils._constants import DEFAULT_LEARNING_RATE, DEFAULT_EPOCHS
from ..utils._validators import _validate_X, _validate_y, _validate_epochs, _validate_learning_rate


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
        self, learning_rate: float = DEFAULT_LEARNING_RATE, epochs: int = DEFAULT_EPOCHS) -> None:
        self._model = _LinearRegression()

        self.learning_rate = _validate_learning_rate(learning_rate)
        self.epochs = _validate_epochs(epochs)

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

        X = _validate_X(X)
        y = _validate_y(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")

        self._model.fit(
            X,
            y,
            self.learning_rate,
            self.epochs,
        )

        self._fitted = True

    @_ensure_fitted
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

        X = _validate_X(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has different number of features")

        return self._model.predict(X)

    @_ensure_fitted
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

        X = _validate_X(X)
        y = _validate_y(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has different number of features")

        return self._model.score(X, y)

    @_ensure_fitted
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

        state = {
            "coef": self.coef_,
            "intercept": self.intercept_,
            "n_features_in": self.n_features_in_,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
        }

        return state
    
    
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
    @_ensure_fitted
    def coef_(self) -> NDArray[np.float64]:
        return np.asarray(self._model.coef_)

    @property
    @_ensure_fitted
    def intercept_(self) -> float:
        return float(self._model.intercept_)

    @property
    @_ensure_fitted
    def n_features_in_(self) -> int:
        return int(self._model.n_features_in_)