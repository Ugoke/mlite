# Indexing and Views

## 1. Description

The indexing module creates zero-copy metadata views using `reshape()` and `transpose()`.

## 2. Mathematics / Logic

`reshape()` preserves the invariant:

```text
product(source.shape) = product(requested.shape)
```

One requested dimension may be `-1`; it is inferred by dividing the source element count by the
product of known dimensions. `transpose(a, b)` swaps both `shape[a]` with `shape[b]` and
`stride[a]` with `stride[b]`.

## 3. Creation

```python
import mlite

x = mlite.ones((2, 3, 4))
matrix = x.reshape((6, 4))
view = matrix.transpose(0, 1)
```

## 4. Core Methods

| Method | Purpose | Copies memory? |
|---|---|---|
| `reshape(shape)` | Change logical dimensions for contiguous data. | No. |
| `transpose(dim0, dim1)` | Exchange two dimensions. | No. |
| `T` | Exchange dimensions 0 and 1 on a matrix. | No. |

## 5. Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `shape` | sequence of integers | — | Target dimensions; at most one `-1`. |
| `dimension0` | integer | `0` | First dimension; negative indexing is accepted. |
| `dimension1` | integer | `1` | Second dimension; negative indexing is accepted. |

## 6. Output

The result is a Tensor sharing source storage, dtype, device, storage offset, and writable state.
Invalid element count, multiple inferred dimensions, non-contiguous reshape input, or an invalid
dimension raises `ShapeError`.

## 7. Examples

### Minimal example

```python
import mlite

x = mlite.ones((2, 6))
y = x.reshape((3, -1))
assert y.shape == (3, 4)
```

### Extended example

```python
import numpy as np
import mlite

source = mlite.tensor(np.arange(24, dtype=np.float32))
batch = source.reshape((2, 3, 4))
reordered = batch.transpose(0, 2)

assert reordered.shape == (4, 3, 2)
reordered.numpy()[0, 0, 1] = 99
assert source.numpy()[12] == 99
```

## 8. Attributes

Views expose changed `shape`, `strides`, and contiguity flags. Their `dtype`, `device`, `size`,
`writable`, and storage lifetime remain tied to the source.

## 9. Notes

- `reshape()` rejects a transposed source; call `.contiguous().reshape(...)` to materialize first.
- Inferring `-1` is rejected when zero dimensions make the answer ambiguous.
- `.T` supports only rank-2 tensors.
