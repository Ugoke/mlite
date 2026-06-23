import numpy as np
from numpy.typing import NDArray

from ..utils._validators import _validate_X, _validate_y, _validate_test_size, _validate_random_state
from .._core import train_test_split as _train_test_split # type: ignore


def train_test_split(X: NDArray[np.float64], y: NDArray[np.float64], test_size: float = 0.2, random_state: int = 42) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Split arrays into random train and test subsets.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix with shape
        (n_samples, n_features).

    y : np.ndarray
        Target values with shape
        (n_samples,).

    test_size : float
        Proportion of the dataset
        to include in the test split.

    random_state : int
        Seed used for random shuffling.

    Returns
    -------
    tuple
        Tuple containing:

        - X_train : np.ndarray
        - X_test : np.ndarray
        - y_train : np.ndarray
        - y_test : np.ndarray
    """

    X = _validate_X(X)
    y = _validate_y(y)
    test_size = _validate_test_size(test_size)
    random_state = _validate_random_state(random_state)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of samples")

    return _train_test_split(X, y, test_size, random_state)