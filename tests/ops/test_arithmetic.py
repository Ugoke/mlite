import numpy as np
import pytest

import mlite


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (mlite.add, [[11, 22, 33], [14, 25, 36]]),
        (mlite.subtract, [[-9, -18, -27], [-6, -15, -24]]),
        (mlite.multiply, [[10, 40, 90], [40, 100, 180]]),
        (mlite.divide, [[0.1, 0.1, 0.1], [0.4, 0.25, 0.2]]),
    ],
)
def test_binary_operations_support_broadcasting(operation, expected):
    left = mlite.tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64))
    right = mlite.tensor(np.array([10, 20, 30], dtype=np.float64))

    result = operation(left, right)

    np.testing.assert_allclose(result.numpy(), expected)
    assert result.shape == (2, 3)
    assert result.is_contiguous


def test_operator_syntax_and_strided_inputs():
    left = mlite.tensor(np.arange(6, dtype=np.float32).reshape(2, 3)).T
    right = mlite.ones((3, 1))

    result = left + right

    np.testing.assert_array_equal(result.numpy(), left.numpy() + 1)


def test_high_rank_strided_broadcasting_advances_offsets_correctly():
    source = mlite.tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
    left = source.transpose(0, 2)
    right = mlite.tensor(np.arange(3, dtype=np.float32).reshape(1, 3, 1))

    result = left + right

    np.testing.assert_array_equal(result.numpy(), left.numpy() + right.numpy())


def test_empty_dimension_broadcasting():
    left = mlite.ones((0, 3))
    right = mlite.ones((1, 3))
    assert (left + right).shape == (0, 3)

    huge = np.iinfo(np.int64).max
    huge_empty = mlite.empty((huge, 0))
    broadcast_empty = mlite.empty((1, 0))
    assert (huge_empty + broadcast_empty).shape == (huge, 0)


def test_invalid_dtypes_shapes_and_integer_division():
    with pytest.raises(mlite.DTypeError):
        mlite.ones((2,), dtype=mlite.int32) + mlite.ones((2,), dtype=mlite.float32)
    with pytest.raises(mlite.ShapeError):
        mlite.ones((2, 3)) + mlite.ones((4,))
    with pytest.raises(mlite.TensorError):
        mlite.ones((2,), dtype=mlite.int32) / mlite.zeros((2,), dtype=mlite.int32)


def test_integer_overflow_is_reported():
    maximum = mlite.tensor(np.array([np.iinfo(np.int32).max], dtype=np.int32))
    one = mlite.ones((1,), dtype=mlite.int32)

    with pytest.raises(mlite.TensorError):
        maximum + one
