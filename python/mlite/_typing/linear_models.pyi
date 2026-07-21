import numpy as np
from numpy.typing import NDArray


class LinearRegression:
    coef_: NDArray[np.float64]
    intercept_: float
    n_features_in_: int

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        learning_rate: float,
        epochs: int,
    ) -> None: ...

    def predict(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]: ...
    
    def score(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> float: ...

    def state_dict(
        self,
    ) -> dict: ...

    def load_state_dict(
        self, 
        state: dict,
    ) -> None: ...


class LogisticRegression:
    coef_: NDArray[np.float64]
    intercept_: float
    n_features_in_: int

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        learning_rate: float,
        epochs: int,
    ) -> None: ...

    def predict(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]: ...
    
    def score(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> float: ...

    def state_dict(
        self,
    ) -> dict: ...

    def load_state_dict(
        self, 
        state: dict,
    ) -> None: ...

    def predict_proba(
        self,
        X: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...