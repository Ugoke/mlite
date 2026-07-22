import numpy as np
from numpy.typing import NDArray

from .._core import mean_absolute_error as _mean_absolute_error  # type: ignore
from ..utils._validators import _validate_y


def mean_absolute_error(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """
    Calculate the mean absolute error (MAE).

    Parameters
    ----------
    y_true : NDArray[np.float64]
        True target values.
    y_pred : NDArray[np.float64]
        Predicted target values.

    Returns
    -------
    float
        Mean absolute error.
    """

    y_true = _validate_y(y_true)
    y_pred = _validate_y(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")

    return _mean_absolute_error(y_true, y_pred)
