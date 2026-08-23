# Creating New Modules

This workflow keeps implementation, Python typing, documentation, and tests synchronized.

## 1. Choose the Architectural Layer

Place the component in the narrowest valid layer:

- `cpp/foundation` for errors, metadata, and dependency-free utilities;
- `cpp/tensor` for tensor metadata and storage abstractions;
- `cpp/execution` for backend-independent execution interfaces;
- `cpp/backends/<backend>` for CPU, CUDA, or other device implementations;
- `cpp/ops` for tensor operations;
- `cpp/ml` for estimators, metrics, preprocessing, and training;
- `bindings/python` for Python and NumPy adapters;
- `python/mlite` for the public Python package surface.

Dependencies must point inward. Tensor code must not depend on Python, NumPy, ML models, or a
specific execution backend.

## 2. Add the Implementation

Create the public header and implementation in matching module directories. Reuse shared
abstractions such as `TensorIterator`, `ExecutionContext`, dtype dispatch, and `Allocator` instead
of duplicating shape, threading, or memory logic.

## 3. Add Bindings and Typing

If the component is public in Python:

1. add the pybind11 adapter in `bindings/python`;
2. release the GIL around CPU-only work;
3. define ownership and copy behavior explicitly;
4. export the symbol from `python/mlite/__init__.py`;
5. update the matching file in `python/mlite/_typing` and its public re-export.

Bindings must remain adapters. Algorithms belong in C++ operation or ML modules.

## 4. Write Documentation

Follow the [Documentation Standard](documentation_standard.md) and link the new page from the
documentation index. Document the public behavior, parameters, return values, errors, and examples.

## 5. Add pytest Coverage

Keep `pytest` as the single test runner. Place tests by responsibility:

```text
tests/cpp/          C++ contracts exposed through test-only bindings
tests/tensor/       shape, strides, storage, views, and dtype behavior
tests/ops/          operation semantics and broadcasting
tests/execution/    concurrency, cancellation, and exception propagation
tests/bindings/     NumPy ownership and Python/C++ conversion
tests/python_api/   exports, metadata, and user contracts
```

Test successful behavior, validation errors, empty/scalar tensors, non-contiguous layouts,
read-only memory, ownership, and multithreaded exception propagation where applicable.

## 6. Validate

Before review:

1. configure and build the extension;
2. run the entire `pytest` suite;
3. compile with project warnings enabled;
4. verify documentation links;
5. confirm that package metadata still comes only from `pyproject.toml`.
