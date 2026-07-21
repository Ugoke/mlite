import numpy as np
from numpy.typing import NDArray


def _validate_X(X: NDArray[np.float64]) -> NDArray[np.float64]:
    X = np.asarray(X, dtype=np.float64)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if X.ndim != 2:
        raise ValueError("X must be 2D")
    
    if X.shape[0] == 0:
        raise ValueError("X cannot be empty")
    
    return X


def _validate_y(y: NDArray[np.float64]) -> NDArray[np.float64]:
    y = np.asarray(y, dtype=np.float64)

    if y.ndim != 1:
        raise ValueError("y must be 1D")
    
    if y.shape[0] == 0:
        raise ValueError("y cannot be empty")
    
    return y


def _validate_learning_rate(learning_rate: float) -> float:
    if not isinstance(learning_rate, float):
        raise TypeError("learning_rate must be float")
    
    if learning_rate <= 0 or not np.isfinite(learning_rate):
        raise ValueError("learning_rate must be finite and positive")
    
    return learning_rate


def _validate_epochs(epochs: int) -> int:
    if not isinstance(epochs, int):
        raise TypeError("epochs must be int")
    
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    
    return epochs


def _validate_test_size(test_size: float) -> float:
    if not isinstance(test_size, float):
        raise TypeError("test_size must be float")

    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be in range (0, 1)")
    
    return test_size


def _validate_random_state(random_state: int) -> int:
    if not isinstance(random_state, int):
        raise TypeError("random_state must be int")
    
    if random_state < 0:
        raise ValueError("random_state must be non-negative")
    
    return random_state


def _validate_n_splits(n_splits: int) -> int:
    if type(n_splits) is not int:
        raise TypeError("n_splits must be int")

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    return n_splits


def _validate_shuffle(shuffle: bool) -> bool:
    if type(shuffle) is not bool:
        raise TypeError("shuffle must be bool")

    return shuffle