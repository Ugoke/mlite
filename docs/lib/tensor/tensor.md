# Tensor

## 1. Description

`Tensor` is the central multidimensional data object in mlite. It combines shared `Storage` with a
`TensorSpec` describing dtype, shape, element strides, and storage offset. Copying a C++ `Tensor`
or returning a Python view is inexpensive because storage ownership is shared.

The public Python class is `mlite.Tensor`. Use factory functions or `mlite.tensor()` instead of
constructing storage manually.

## 2. Mathematics / Logic

For an index `(i₀, i₁, ..., iₙ)`, the element offset is:

```text
element_offset = storage_offset + Σ(iₖ × strideₖ)
byte_address   = storage_base + element_offset × dtype_size
```

Strides are measured in elements. A tensor may therefore represent contiguous data, a transposed
view, or Fortran-order data without moving the underlying values.

The tensor validates that the smallest and largest reachable offsets remain inside storage. There
is no implicit copy-on-write behavior: writable views observe the same memory.

## 3. Creation

```python
import mlite
import numpy as np

a = mlite.tensor([1, 2, 3], dtype=mlite.int64)
b = mlite.zeros((2, 3))
c = mlite.tensor(np.arange(6, dtype=np.float32).reshape(2, 3), copy=False)
```

`Tensor(data, copy=None)` accepts data and a copy policy. `mlite.tensor()` additionally accepts an
explicit dtype.

## 4. Core Methods

| Method | Purpose | Copies memory? |
|---|---|---|
| `numpy(copy=False)` | Return a NumPy representation. | No by default; yes when requested. |
| `clone()` | Create independent contiguous storage. | Always. |
| `contiguous()` | Obtain C-contiguous layout. | Only when the source is not contiguous. |
| `reshape(shape)` | Return a new contiguous view with another shape. | No. |
| `transpose(dim0, dim1)` | Swap two dimensions and their strides. | No. |
| `T` | Transpose a rank-2 tensor. | No. |

Arithmetic operators `+`, `-`, `*`, and `/` create new tensors and are documented in
[Arithmetic Operations](../ops/arithmetic.md).

## 5. Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | array-like or `Tensor` | — | Input values or an existing tensor. |
| `dtype` | `DType` or `None` | `None` | Requested dtype. A differing dtype converts Tensor input unless `copy=False`. |
| `copy` | `bool` or `None` | `None` | `False` requires zero-copy, `True` forces a copy, and `None` copies only when required. |
| `shape` | sequence of integers | — | New shape for `reshape`; one dimension may be `-1`. |
| `dim0`, `dim1` | integer | `0`, `1` | Dimensions exchanged by `transpose`; negative dimensions are supported. |

## 6. Output

Tensor factories and operations return a CPU `Tensor`. Views preserve the source dtype, device,
storage ownership, and writable state. Materializing operations return C-contiguous writable
storage.

Invalid metadata raises `ShapeError`; unsupported conversions raise `DTypeError`; mutation through
read-only storage raises `ReadOnlyError`.

## 7. Examples

### Minimal example

```python
import mlite

value = mlite.ones((2, 3))
print(value.shape)       # (2, 3)
print(value.dtype)       # DType.float32
print(value.numpy())
```

### Extended example

```python
import numpy as np
import mlite

source = np.arange(12, dtype=np.float32)
tensor = mlite.tensor(source, copy=False)
matrix = tensor.reshape((3, 4))
columns = matrix.T
materialized = columns.contiguous()

matrix.numpy()[1, 0] = 100
assert source[4] == 100             # reshape shares storage
assert not columns.is_contiguous
assert materialized.is_contiguous
```

## 8. Attributes

| Attribute | Type | Description |
|---|---|---|
| `shape` | `tuple[int, ...]` | Dimension sizes. |
| `strides` | `tuple[int, ...]` | Element strides, not byte strides. |
| `ndim` | `int` | Number of dimensions. |
| `size` | `int` | Logical number of elements. A scalar has size 1. |
| `nbytes` | `int` | Logical element count multiplied by dtype size. |
| `dtype` | `DType` | Element type. |
| `device` | `str` | Execution device (`"cpu"`). |
| `writable` | `bool` | Whether mutation through exported memory is permitted. |
| `is_contiguous` | `bool` | Whether the layout is C-contiguous. |
| `is_fortran_contiguous` | `bool` | Whether the layout is column-major contiguous. |

## 9. Notes

- `reshape()` requires C-contiguous input; call `contiguous()` first for a transposed tensor.
- `.T` works with rank-2 tensors; use `transpose()` to specify dimensions explicitly.
- See [NumPy Interoperability](../bindings/python/numpy.md) for lifetime and zero-copy guarantees.
