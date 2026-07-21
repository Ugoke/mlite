from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..utils._validators import _validate_X, _validate_y, _validate_n_splits, _validate_shuffle, _validate_random_state
from ..utils._estimator import _clone_estimator
from .k_fold import k_fold


def cross_val_score(estimator, X: NDArray[np.float64], y: NDArray[np.float64], n_splits: int = 5, shuffle: bool = False, random_state: int = 42) -> NDArray[np.float64]:
    """
    Evaluate an estimator using K-fold cross-validation.

    Parameters
    ----------
    estimator :
        Model implementing
        ``fit(X, y)`` and
        ``score(X, y)`` methods.

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
        Whether to shuffle data
        before splitting.

    random_state : int
        Seed used for shuffling.

    Returns
    -------
    np.ndarray
        Scores for each validation fold.
        Shape is ``(n_splits,)``.
    """

    X = _validate_X(X)
    y = _validate_y(y)

    n_splits = _validate_n_splits(n_splits)
    shuffle = _validate_shuffle(shuffle)
    random_state = _validate_random_state(random_state)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of samples")

    folds = k_fold(X, y, n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    scores = []

    for X_train, X_test, y_train, y_test in folds:
        model = _clone_estimator(estimator)
        model.fit(X_train, y_train)
        
        scores.append(model.score(X_test, y_test))

    return np.asarray(scores, dtype=np.float64)