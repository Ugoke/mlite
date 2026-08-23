import gc

import numpy as np
import pytest

import mlite


def test_contiguous_numpy_input_is_zero_copy_and_kept_alive():
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    value = mlite.tensor(array, copy=False)

    array[0, 1] = 42
    assert value.numpy()[0, 1] == 42

    del array
    gc.collect()
    np.testing.assert_array_equal(value.numpy(), [[0, 42, 2], [3, 4, 5]])


def test_fortran_numpy_input_is_zero_copy():
    array = np.asfortranarray(np.arange(6, dtype=np.float64).reshape(2, 3))
    value = mlite.tensor(array, copy=False)

    assert value.is_fortran_contiguous
    assert value.strides == (1, 2)
    value.numpy()[1, 2] = 77
    assert array[1, 2] == 77


def test_noncontiguous_input_obeys_copy_policy():
    source = np.arange(12, dtype=np.int64).reshape(3, 4)[:, ::2]
    copied = mlite.tensor(source)
    source[0, 0] = -1

    assert copied.is_contiguous
    assert copied.numpy()[0, 0] == 0
    with pytest.raises(mlite.TensorError):
        mlite.tensor(source, copy=False)


def test_copy_false_never_materializes_or_converts_input():
    with pytest.raises(mlite.TensorError, match="NumPy ndarray"):
        mlite.tensor([1, 2, 3], copy=False)

    source = np.arange(4, dtype=np.int32)
    with pytest.raises(mlite.DTypeError, match="dtype conversion"):
        mlite.tensor(source, dtype=mlite.float32, copy=False)

    value = mlite.tensor(source, dtype=mlite.int32, copy=False)
    assert np.shares_memory(value.numpy(), source)


def test_explicit_copy_is_independent():
    source = np.arange(4, dtype=np.float32)
    copied = mlite.tensor(source, copy=True)
    source[0] = 100
    assert copied.numpy()[0] == 0


def test_explicit_copy_with_dtype_is_independent_and_converted():
    source = np.arange(4, dtype=np.int32)
    copied = mlite.tensor(source, dtype=mlite.float64, copy=True)

    source[0] = 100
    assert copied.dtype == mlite.float64
    np.testing.assert_array_equal(copied.numpy(), np.arange(4, dtype=np.float64))


def test_tensor_input_dtype_conversion_obeys_copy_policy():
    source = mlite.tensor(np.arange(4, dtype=np.int32), copy=False)
    converted = mlite.tensor(source, dtype=mlite.float64)

    assert converted.dtype == mlite.float64
    np.testing.assert_array_equal(converted.numpy(), source.numpy())
    with pytest.raises(mlite.DTypeError, match="copy=False"):
        mlite.tensor(source, dtype=mlite.float64, copy=False)


def test_readonly_input_remains_readonly():
    source = np.arange(4, dtype=np.float32)
    source.flags.writeable = False
    value = mlite.tensor(source, copy=False)
    array = value.numpy()

    assert not value.writable
    assert not array.flags.writeable
    with pytest.raises(ValueError):
        array[0] = 1


def test_buffer_protocol_and_unsupported_dtype():
    value = mlite.ones((2, 2), dtype=mlite.float64)
    array = np.asarray(value)
    array[0, 0] = 9
    assert value.numpy()[0, 0] == 9

    with pytest.raises(mlite.DTypeError):
        mlite.tensor(np.arange(3, dtype=np.uint32))


def test_array_protocol_honors_copy_policy_and_dtype():
    value = mlite.tensor(np.arange(4, dtype=np.float32), copy=False)

    shared = value.__array__(dtype=np.dtype(np.float32), copy=False)
    assert np.shares_memory(shared, value.numpy())

    with pytest.raises(ValueError, match="requested dtype"):
        value.__array__(dtype=np.dtype(np.float64), copy=False)

    converted = value.__array__(dtype=np.dtype(np.float64), copy=None)
    copied = value.__array__(copy=True)
    assert converted.dtype == np.dtype(np.float64)
    assert not np.shares_memory(converted, value.numpy())
    assert not np.shares_memory(copied, value.numpy())


def test_copy_policy_requires_bool_or_none():
    source = np.arange(3, dtype=np.float32)
    with pytest.raises(TypeError, match="True, False, or None"):
        mlite.tensor(source, copy=0)
    with pytest.raises(TypeError, match="True, False, or None"):
        mlite.tensor(source).__array__(copy=0)
