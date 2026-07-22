import numpy as np
from numpy.typing import NDArray


def r2_score(
        y_true: NDArray[np.float64], 
        y_pred: NDArray[np.float64]
    ) -> float: ...


def accuracy_score(
        y_true: NDArray[np.float64], 
        y_pred: NDArray[np.float64]
    ) -> float: ...


def mean_squared_error(
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64]
    ) -> float: ...


def root_mean_squared_error(
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64]
    ) -> float: ...


def mean_absolute_error(
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64]
    ) -> float: ...


def recall_score(
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64]
    ) -> float: ...


def k_fold(
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        n_splits: int,
        shuffle: bool,
        random_state: int,
    ) -> list[
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ]
    ]: ...