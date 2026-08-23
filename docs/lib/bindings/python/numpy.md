# NumPy Interoperability

## 1. Description

The NumPy adapter converts between `numpy.ndarray` and C++ `Tensor` while preserving dtype, shape,
strides, writable state, and object lifetime. Compatible arrays share memory in both directions.

## 2. Mathematics / Logic

NumPy reports byte strides; mlite stores element strides:

```text
mlite_stride = numpy_byte_stride / dtype_size
numpy_byte_stride = mlite_stride × dtype_size
```

Zero-copy NumPy input requires native-endian supported dtype, proper alignment, and either C- or
Fortran-contiguous layout. Otherwise `copy=None` creates a C-order copy and `copy=False` raises an
error.

The storage owner retains a Python reference under the GIL. NumPy exports retain a shallow Tensor
inside a capsule, keeping C++ storage alive.

## 3. Creation

```python
import numpy as np
import mlite

array = np.arange(6, dtype=np.float32).reshape(2, 3)
tensor = mlite.tensor(array, copy=False)
```

## 4. Core Methods

| API | Purpose | Copies memory? |
|---|---|---|
| `mlite.tensor(array, copy=False)` | Require NumPy-to-Tensor zero-copy. | No. |
| `mlite.tensor(array, copy=None)` | Use zero-copy when compatible. | Conditional. |
| `mlite.tensor(array, copy=True)` | Create independent Tensor storage. | Yes. |
| `mlite.tensor(tensor, dtype=...)` | Convert an existing Tensor dtype. | When dtype differs. |
| `tensor.numpy(copy=False)` | Export a NumPy view. | No. |
| `tensor.numpy(copy=True)` | Export an independent NumPy array. | Yes. |
| `numpy.asarray(tensor)` | Consume the Python buffer/array protocol. | Normally no. |

## 5. Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | `numpy.ndarray` or array-like | — | Source values. |
| `dtype` | `mlite.DType` or `None` | `None` | Optional dtype requested through `numpy.asarray`. |
| `copy` | `bool` or `None` | `None` on input | Copy policy. Tensor export defaults to `False`. |

Supported NumPy dtypes are native `bool`, `int32`, `int64`, `float32`, and `float64`.

## 6. Output

Input conversion returns a CPU Tensor. Export returns a NumPy array with matching shape, byte
strides, dtype, and writeability. Unsupported dtype raises `DTypeError`; an impossible
`copy=False` request raises `TensorError`.

## 7. Examples

### Minimal example

```python
import numpy as np
import mlite

array = np.array([1, 2, 3], dtype=np.float32)
tensor = mlite.tensor(array, copy=False)
array[0] = 10
assert tensor.numpy()[0] == 10
```

### Extended example

```python
import gc
import numpy as np
import mlite

source = np.asfortranarray(np.arange(6, dtype=np.float64).reshape(2, 3))
tensor = mlite.tensor(source, copy=False)

assert tensor.is_fortran_contiguous
assert tensor.strides == (1, 2)

exported = tensor.numpy()
del source
gc.collect()

exported[1, 2] = 99
assert tensor.numpy()[1, 2] == 99
```

## 8. Attributes

The adapter preserves `Tensor.shape`, `strides`, `dtype`, and `writable`. NumPy exposes the
corresponding `shape`, byte `strides`, `dtype`, and `flags.writeable`.

## 9. Notes

- Non-contiguous NumPy slices are copied under `copy=None` and rejected under `copy=False`.
- Python sequences require NumPy materialization and are therefore rejected under `copy=False`.
- Requesting a different dtype requires conversion and is rejected under `copy=False`, including
  through NumPy's `__array__(dtype=..., copy=False)` protocol.
- Converting an existing Tensor to a different dtype makes one independent C-contiguous
  allocation when copying is allowed.
- Read-only NumPy input remains read-only when exported again.
- NumPy unsigned integers, complex numbers, strings, objects, and non-native-endian types are not
  supported.
- `Tensor.numpy()` defaults to zero-copy; use `copy=True` when independent lifetime and mutation are
  required.
