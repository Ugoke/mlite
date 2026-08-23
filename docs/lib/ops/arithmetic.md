# Arithmetic Operations

## 1. Description

Arithmetic operations implement elementwise addition, subtraction, multiplication, and division
for numeric tensors. They accept strided inputs, apply NumPy-style broadcasting, execute large
ranges in parallel, and return contiguous output.

## 2. Mathematics / Logic

For every broadcast output index `i`:

```text
output[i] = operation(left[broadcast(i)], right[broadcast(i)])
```

Dimensions are aligned from the right. Two dimensions are compatible when they are equal or one is
`1`. A broadcast dimension uses stride zero for the operand whose input dimension is `1`.

Integer operations perform explicit overflow checks. Integer division truncates toward zero and
rejects division by zero and `minimum / -1` overflow.

## 3. Creation

```python
import mlite

a = mlite.ones((2, 3))
b = mlite.full((3,), 2)

result = a + b
```

Functional forms are also provided: `mlite.add`, `subtract`, `multiply`, and `divide`.

## 4. Core Methods

| Function | Operator | Logic |
|---|---|---|
| `mlite.add(a, b)` | `a + b` | Elementwise addition. |
| `mlite.subtract(a, b)` | `a - b` | Elementwise subtraction. |
| `mlite.multiply(a, b)` | `a * b` | Elementwise multiplication. |
| `mlite.divide(a, b)` | `a / b` | Elementwise division. |

## 5. Parameters

| Parameter | Type | Requirements |
|---|---|---|
| `left` | `Tensor` | Numeric dtype; same dtype and device as right. |
| `right` | `Tensor` | Numeric dtype; shape broadcast-compatible with left. |

Implicit dtype promotion, scalar operands, and cross-device operations are not supported.

## 6. Output

Every operation returns a new writable C-contiguous tensor on the operands' device with their
shared dtype and broadcast shape.

Incompatible shapes raise `ShapeError`; dtype mismatch or bool arithmetic raises `DTypeError`;
device mismatch raises `DeviceError`; integer zero division or overflow raises `TensorError`.

## 7. Examples

### Minimal example

```python
import mlite

a = mlite.full((3,), 6)
b = mlite.full((3,), 2)
print((a / b).numpy())  # [3. 3. 3.]
```

### Extended example

```python
import numpy as np
import mlite

samples = mlite.tensor(
    np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
)
scale = mlite.tensor(np.array([10, 20, 30], dtype=np.float32))
bias = mlite.ones((2, 1))

normalized = samples * scale + bias

np.testing.assert_array_equal(
    normalized.numpy(),
    [[11, 41, 91], [41, 101, 181]],
)
```

## 8. Attributes

Arithmetic operations are stateless. Output attributes are derived from the broadcast shape and
input dtype/device; output strides are always C-contiguous.

## 9. Notes

- Both operands must be Tensor objects.
- Arithmetic accepts `int32`, `int64`, `float32`, and `float64` tensors.
- Floating-point division follows the platform's IEEE behavior.
- Operations release the Python GIL and use the shared CPU thread pool for sufficiently large
  outputs.
