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