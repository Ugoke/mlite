import numpy as np
from numpy.typing import NDArray

from ..utils._validators import _validate_y
from .._core import accuracy_score as _accuracy_score  # type: ignore


def accuracy_score(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """
    Calculate the classification accuracy score.

    Parameters
    ----------
    y_true : NDArray[np.float64]
        True class labels.

    y_pred : NDArray[np.float64]
        Predicted class labels.

    Returns
    -------
    float
        Accuracy score.

        1.0 means all predictions are correct.
        0.0 means all predictions are incorrect.
    """

    y_true = _validate_y(y_true)
    y_pred = _validate_y(y_pred)

    return _accuracy_score(y_true, y_pred)