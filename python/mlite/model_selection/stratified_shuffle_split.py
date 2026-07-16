import numpy as np
from numpy.typing import NDArray

from .._core import stratified_shuffle_split as _stratified_shuffle_split # type: ignore
from ..utils._validators import _validate_X, _validate_y, _validate_test_size, _validate_random_state


def stratified_shuffle_split(X: NDArray[np.float64], y: NDArray[np.float64], test_size: float = 0.2, random_state: int = 42) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Split arrays into random train and test subsets while preserving
    the class distribution.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.

    y : array-like of shape (n_samples,)
        Target labels.

    test_size : float, default=0.2
        Fraction of samples assigned to the test set.

    random_state : int, default=42
        Seed used for reproducible shuffling.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """

    X = _validate_X(X)
    y = _validate_y(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of samples.")

    test_size = _validate_test_size(test_size)
    random_state = _validate_random_state(random_state)

    classes, counts = np.unique(y, return_counts=True)

    if classes.size < 2:
        raise ValueError("y must contain at least two classes.")

    if np.any(counts < 2):
        raise ValueError("Each class must contain at least two samples for stratified splitting.")

    return _stratified_shuffle_split(X, y, test_size, random_state)