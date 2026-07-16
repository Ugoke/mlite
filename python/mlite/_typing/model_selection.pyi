import numpy as np
from numpy.typing import NDArray


def train_test_split(    
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    test_size: float = 0.2,
    random_state: int = 42
    ) -> tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64]
            ]: ...


def stratified_shuffle_split(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    test_size: float = 0.2,
    random_state: int = 42
    ) -> tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64]
            ]: ...