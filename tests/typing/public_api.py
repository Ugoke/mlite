import numpy as np

import mlite


left: mlite.Tensor = mlite.ones((2, 3))
right: mlite.Tensor = mlite.full((2, 3), np.float32(2.0))

added: mlite.Tensor = mlite.add(left=left, right=right)
subtracted: mlite.Tensor = mlite.subtract(left=left, right=right)
multiplied: mlite.Tensor = mlite.multiply(left=left, right=right)
divided: mlite.Tensor = mlite.divide(left=left, right=right)

integer_filled: mlite.Tensor = mlite.full((1,), np.int64(2), dtype=mlite.int64)
boolean_filled: mlite.Tensor = mlite.full((1,), np.bool_(True), dtype=mlite.bool_)
