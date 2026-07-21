import numpy as np
from numpy.typing import NDArray

from .._core import k_fold as _k_fold # type: ignore
from ..utils._validators import _validate_X, _validate_y, _validate_n_splits, _validate_shuffle, _validate_random_state


def k_fold(X: NDArray[np.float64], y: NDArray[np.float64], n_splits: int = 5, shuffle: bool = False, random_state: int = 42) -> list[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]]:
    """
    Split data into K consecutive train and test folds.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix with shape
        (n_samples, n_features).

    y : np.ndarray
        Target values with shape
        (n_samples,).

    n_splits : int
        Number of folds.
        Must be at least 2.

    shuffle : bool
        Whether to shuffle the
        data before splitting
        into folds.

    random_state : int
        Seed used for random
        shuffling when
        ``shuffle=True``.

    Returns
    -------
    list
        List of tuples, where each tuple
        contains one train/test split:

        - X_train : np.ndarray
        - X_test : np.ndarray
        - y_train : np.ndarray
        - y_test : np.ndarray
    """
    X = _validate_X(X)
    y = _validate_y(y)

    n_splits = _validate_n_splits(n_splits)
    shuffle = _validate_shuffle(shuffle)
    random_state = _validate_random_state(random_state)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of samples")

    if n_splits > X.shape[0]:
        raise ValueError("n_splits cannot be greater than number of samples")

    return _k_fold(X, y, n_splits, shuffle, random_state)