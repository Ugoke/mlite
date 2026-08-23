from typing import Any, Optional, Sequence, Union

import numpy as np

from .dtype import DType
from .tensor import Tensor


def tensor(
    data: Any,
    dtype: Optional[DType] = ...,
    copy: Optional[bool] = ...,
) -> Tensor: ...

def empty(shape: Sequence[int], dtype: DType = ...) -> Tensor: ...
def zeros(shape: Sequence[int], dtype: DType = ...) -> Tensor: ...
def ones(shape: Sequence[int], dtype: DType = ...) -> Tensor: ...
def full(
    shape: Sequence[int],
    fill_value: Union[
        bool,
        int,
        float,
        np.bool_,
        np.integer[Any],
        np.floating[Any],
    ],
    dtype: DType = ...,
) -> Tensor: ...
def add(left: Tensor, right: Tensor) -> Tensor: ...
def subtract(left: Tensor, right: Tensor) -> Tensor: ...
def multiply(left: Tensor, right: Tensor) -> Tensor: ...
def divide(left: Tensor, right: Tensor) -> Tensor: ...
