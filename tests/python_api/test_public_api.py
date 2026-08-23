import numpy as np
import pytest

import mlite
from mlite import _core


def test_public_api_exports_tensor_surface():
    expected = {
        "Tensor",
        "DType",
        "tensor",
        "empty",
        "zeros",
        "ones",
        "full",
        "add",
        "subtract",
        "multiply",
        "divide",
        "set_num_threads",
        "get_num_threads",
    }

    assert expected.issubset(set(mlite.__all__))
    assert repr(mlite.ones((2, 3))) == "Tensor(shape=(2, 3), dtype=float32, device='cpu')"


def test_error_hierarchy_matches_the_public_contract():
    assert issubclass(mlite.Error, RuntimeError)
    assert issubclass(mlite.TensorError, mlite.Error)
    assert issubclass(mlite.ExecutionError, mlite.Error)


def test_production_extension_does_not_export_test_hooks():
    assert not hasattr(_core, "_testing")


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (mlite.add, [4.0, 6.0]),
        (mlite.subtract, [-2.0, -2.0]),
        (mlite.multiply, [3.0, 8.0]),
        (mlite.divide, [1.0 / 3.0, 0.5]),
    ],
)
def test_functional_arithmetic_accepts_public_keyword_names(operation, expected):
    left = mlite.tensor(np.array([1.0, 2.0], dtype=np.float32))
    right = mlite.tensor(np.array([3.0, 4.0], dtype=np.float32))

    result = operation(left=left, right=right)

    np.testing.assert_allclose(result.numpy(), expected)
    assert "left:" in operation.__doc__
    assert "right:" in operation.__doc__


def test_core_functions_and_methods_have_user_facing_docstrings():
    documented = (
        mlite.tensor,
        mlite.empty,
        mlite.zeros,
        mlite.ones,
        mlite.full,
        mlite.add,
        mlite.subtract,
        mlite.multiply,
        mlite.divide,
        mlite.Tensor.numpy,
        mlite.Tensor.clone,
        mlite.Tensor.contiguous,
        mlite.Tensor.reshape,
        mlite.Tensor.transpose,
    )

    for symbol in documented:
        assert symbol.__doc__ is not None
        assert "\n\n" in symbol.__doc__
