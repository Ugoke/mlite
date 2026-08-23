# Creation Operations

## 1. Description

Creation operations allocate CPU storage and construct C-contiguous tensors. They provide
uninitialized, zero-filled, one-filled, constant-filled, cloned, and contiguous results.

## 2. Mathematics / Logic

For shape `S` and dtype `D`, allocation size is:

```text
bytes = product(S) × size(D)
```

`full()` dispatches once on dtype and fills independent index ranges through the execution
context. `clone()` copies logical row-major iteration order, so it can materialize a strided view.

## 3. Creation

```python
import mlite

a = mlite.empty((2, 3))
b = mlite.zeros((2, 3))
c = mlite.ones((2, 3), dtype=mlite.float64)
d = mlite.full((2, 3), 5, dtype=mlite.int32)
```

## 4. Core Methods

| Function or method | Purpose | Copies/initializes memory? |
|---|---|---|
| `empty(shape, dtype)` | Allocate a tensor without initializing values. | Allocates only. |
| `zeros(shape, dtype)` | Create a zero-filled tensor. | Allocates and initializes. |
| `ones(shape, dtype)` | Create a one-filled tensor. | Allocates and initializes. |
| `full(shape, value, dtype)` | Fill every element with a constant. | Allocates and initializes. |
| `Tensor.clone()` | Create independent contiguous storage. | Always copies. |
| `Tensor.contiguous()` | Return C-contiguous layout. | Conditional. |

## 5. Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `shape` | sequence of integers | — | Non-negative dimensions. |
| `dtype` | `DType` | `mlite.float32` | Element representation. |
| `fill_value` | number | — | Constant converted to the requested dtype. |

Integer fill values must be finite, integral, and inside the target dtype range.

## 6. Output

Every function returns a writable CPU tensor. Factory results are C-contiguous. `contiguous()` may
return a shallow alias when its input is already C-contiguous.

Invalid shapes raise `ShapeError`; invalid fill conversions raise `DTypeError`; allocation failure
propagates as a standard allocation exception.

## 7. Examples

### Minimal example

```python
import mlite

x = mlite.full((3,), 2.5)
print(x.numpy())  # [2.5 2.5 2.5]
```

### Extended example

```python
import numpy as np
import mlite

weights = mlite.ones((2, 3), dtype=mlite.float64)
transposed = weights.T
packed = transposed.contiguous()
backup = weights.clone()

packed.numpy()[0, 0] = 10
backup.numpy()[0, 0] = 20

assert weights.numpy()[0, 0] == 1
assert packed.is_contiguous
np.testing.assert_array_equal(backup.numpy()[0, 1:], [1, 1])
```

## 8. Attributes

Results expose the standard Tensor attributes. Creation functions add no mutable configuration or
global factory state.

## 9. Notes

- Reading values from `empty()` before writing them is invalid application behavior.
- Large fills and strided clones use the CPU thread pool automatically.
