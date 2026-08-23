import numpy as np
import pytest

import mlite


@pytest.mark.parametrize(
    ("dtype", "numpy_dtype"),
    [
        (mlite.bool_, np.bool_),
        (mlite.int32, np.int32),
        (mlite.int64, np.int64),
        (mlite.float32, np.float32),
        (mlite.float64, np.float64),
    ],
)
def test_zeros_supports_all_public_dtypes(dtype, numpy_dtype):
    value = mlite.zeros((2, 3), dtype=dtype)

    assert value.shape == (2, 3)
    assert value.strides == (3, 1)
    assert value.ndim == 2
    assert value.size == 6
    assert value.nbytes == 6 * np.dtype(numpy_dtype).itemsize
    assert value.dtype == dtype
    assert value.device == "cpu"
    assert value.is_contiguous
    np.testing.assert_array_equal(value.numpy(), np.zeros((2, 3), dtype=numpy_dtype))


def test_factories_support_scalar_and_empty_shapes():
    scalar = mlite.full((), 4.5, dtype=mlite.float64)
    empty = mlite.ones((2, 0, 3))

    assert scalar.shape == ()
    assert scalar.size == 1
    assert scalar.numpy().item() == 4.5
    assert empty.shape == (2, 0, 3)
    assert empty.size == 0
    assert empty.numpy().shape == (2, 0, 3)


def test_invalid_shapes_and_integer_fill_values_are_rejected():
    with pytest.raises(mlite.ShapeError):
        mlite.empty((2, -1))
    with pytest.raises(mlite.DTypeError):
        mlite.full((2,), 1.5, dtype=mlite.int32)


def test_int64_fill_values_are_not_rounded_through_float():
    values = (2**53 + 1, np.iinfo(np.int64).max, np.iinfo(np.int64).min)

    for fill_value in values:
        result = mlite.full((2,), int(fill_value), dtype=mlite.int64)
        np.testing.assert_array_equal(
            result.numpy(),
            np.full((2,), fill_value, dtype=np.int64),
        )


@pytest.mark.parametrize("fill_value", [2**63, -(2**63) - 1])
def test_python_integer_outside_int64_range_is_rejected(fill_value):
    with pytest.raises(mlite.DTypeError, match="outside the int64 range"):
        mlite.full((1,), fill_value, dtype=mlite.int64)


def test_floating_int64_upper_boundary_is_rejected_without_wrapping():
    # INT64_MAX is not exactly representable as a double and rounds to 2**63.
    # The conversion must reject that half-open upper bound before the C++ cast.
    for fill_value in (float(2**63), float(np.iinfo(np.int64).max)):
        with pytest.raises(mlite.DTypeError, match="outside dtype range"):
            mlite.full((1,), fill_value, dtype=mlite.int64)


def test_numpy_real_scalars_are_supported_without_precision_loss():
    integer = np.int64(2**53 + 1)
    floating = np.float32(1.25)

    assert mlite.full((1,), integer, dtype=mlite.int64).numpy()[0] == integer
    assert mlite.full((1,), floating, dtype=mlite.float32).numpy()[0] == floating
    assert mlite.full((1,), np.bool_(True), dtype=mlite.bool_).numpy()[0]


def test_default_factory_dtype_is_float32():
    assert mlite.ones((1,)).dtype == mlite.float32
