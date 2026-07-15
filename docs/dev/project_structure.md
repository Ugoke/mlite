# Project Structure

`mlite` combines a C++17 core with a Python API. A public call passes through a Python wrapper, the single pybind11 extension `mlite._core`, and a C++ algorithm.

```text
python/mlite/<category>/<name>.py
        ↓
core/bindings/<category>/py_<name>.cpp
        ↓
core/include/mlite/<category>/<name>.hpp
core/src/<category>/<name>.cpp
```

| Path | Role |
| --- | --- |
| `core/include/mlite/core` | common C++ read-only views for NumPy buffers |
| `core/include/mlite/<category>` | public C++ declarations of algorithms |
| `core/src/<category>` | C++ computation implementations |
| `core/bindings` | pybind11 adapters and the `PYBIND11_MODULE(_core, m)` entry point |
| `python/mlite` | user-facing API, validation, typing, and serialization |
| `python/mlite/utils` | internal Python helpers; not public API |
| `tests` | pytest contracts for the public Python API |
| `docs/lib` | user documentation mirroring the public package structure |

For the complete, normative guide to creating and registering a module—including supported module types, templates, lifecycle, testing, and checklist—see [Creating New Modules](new_module_development.md).