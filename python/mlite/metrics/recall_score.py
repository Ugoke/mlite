import numpy as np
from numpy.typing import NDArray

from .._core import recall_score as _recall_score  # type: ignore
from ..utils._validators import _validate_y


def recall_score(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """
    Calculate recall for binary classification with positive label 1.0.

    Parameters
    ----------
    y_true : NDArray[np.float64]
        True binary class labels.
    y_pred : NDArray[np.float64]
        Predicted binary class labels.

    Returns
    -------
    float
        Recall score. Returns 0.0 when there are no positive true labels.
    """

    y_true = _validate_y(y_true)
    y_pred = _validate_y(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")

    if not np.all(np.isin(y_true, [0.0, 1.0])) or not np.all(np.isin(y_pred, [0.0, 1.0])):
        raise ValueError("y_true and y_pred must contain only binary labels 0.0 and 1.0")

    return _recall_score(y_true, y_pred)
