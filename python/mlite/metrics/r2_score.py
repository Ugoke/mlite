import numpy as np
from numpy.typing import NDArray

from ..utils._validators import _validate_y
from .._core import r2_score as _r2_score # type: ignore


def r2_score(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """
    Calculate the coefficient of determination (R² score).

    Parameters
    ----------
    y_true : NDArray[np.float64]
        True target values.

    y_pred : NDArray[np.float64]
        Predicted target values.

    Returns
    -------
    float
        R² score.
    """

    y_true = _validate_y(y_true)
    y_pred = _validate_y(y_pred)

    return _r2_score(y_true, y_pred)