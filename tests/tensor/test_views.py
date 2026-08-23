import numpy as np
import pytest

import mlite


def test_reshape_is_zero_copy_and_supports_inference():
    source = mlite.tensor(np.arange(12, dtype=np.float32))
    reshaped = source.reshape((3, -1))

    reshaped.numpy()[1, 0] = 99

    assert reshaped.shape == (3, 4)
    assert source.numpy()[4] == 99


def test_transpose_is_a_strided_view():
    source = mlite.tensor(np.arange(6, dtype=np.int32).reshape(2, 3))
    transposed = source.T

    assert transposed.shape == (3, 2)
    assert transposed.strides == (1, 3)
    assert not transposed.is_contiguous
    np.testing.assert_array_equal(transposed.numpy(), source.numpy().T)


def test_clone_and_contiguous_materialize_storage():
    source = mlite.tensor(np.arange(6, dtype=np.float64).reshape(2, 3))
    cloned = source.clone()
    contiguous = source.T.contiguous()

    cloned.numpy()[0, 0] = -1
    contiguous.numpy()[0, 0] = -2

    assert source.numpy()[0, 0] == 0
    assert contiguous.is_contiguous
    np.testing.assert_array_equal(contiguous.numpy()[1:], source.numpy().T[1:])


def test_invalid_reshape_and_dimensions_are_rejected():
    source = mlite.ones((2, 3))

    with pytest.raises(mlite.ShapeError):
        source.reshape((5,))
    with pytest.raises(mlite.ShapeError):
        source.transpose(0, 2)
    with pytest.raises(mlite.ShapeError):
        source.T.reshape((6,))
