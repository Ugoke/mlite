# Developer Guide: Creating Modules for mlite

> This document is intended to be the only resource needed to add a new public module to `mlite`.
>
> Labels used below:
>
> - **[Explicit]** — a rule expressed by configuration, public API, or existing documentation.
> - **[Inferred]** — a consistent convention derived from all existing implementations and tests. Follow it until the project adopts a different explicit rule.
> - **[Recommendation]** — a safe practice for new code; legacy modules may not implement it.

## 1. Project overview

`mlite` is a Python package with a C++17 computation core. Users interact with a pure Python API, whose wrappers validate and normalize input before delegating computation to the native `mlite._core` module.

```text
User
    │  from mlite.<category> import <PublicName>
    ▼
Python wrapper (python/mlite/<category>/<snake_name>.py)
    │  validation; NumPy conversion
    ▼
pybind11 binding (core/bindings/.../py_<snake_name>.cpp)
    │  NumPy buffer → MatrixView / VectorView or scalar
    ▼
C++ API (core/include/mlite/<category>/<snake_name>.hpp)
    ▼
C++ implementation (core/src/<category>/<snake_name>.cpp)
```

### Separation of responsibilities

| Layer | Responsibility | Excludes |
| --- | --- | --- |
| Python public API | Public API, validation, lifecycle, serialization, result conversion | Algorithms |
| pybind11 binding | Python/C++ conversion, exported interface, memory ownership | Validation and algorithms |
| C++ core | Algorithms, internal state, performance-critical code | Python-specific behavior and user-facing validation |
| `core` views | Read-only access to contiguous `double` buffers | Ownership and mutation |
| Stubs | Static typing | Runtime behavior |
| Tests | Public contract and regression coverage | Private implementation details |
| Docs | Public behavior | Internal implementation details |

### Current public categories

| Category | Public items | Implementation |
| --- | --- | --- |
| `linear_models` | `LinearRegression`, `LogisticRegression` | Trainable C++ classes with Python wrappers |
| `model_selection` | `train_test_split` | C++ free function with a Python wrapper |
| `metrics` | Evaluation metrics | Stateless C++ functions in `metrics/` with Python wrappers |
| `preprocessing` | `OneHotEncoder`, `OrdinalEncoder` | Stateful C++ transformers with Python wrappers |
| `utils` | Internal validators, constants, decorators | Python-only; not part of the public API |

## 2. Directory structure and file responsibilities

```text
.
├── CMakeLists.txt                 # builds the single C++ extension _core
├── pyproject.toml                 # packaging, dependencies, scikit-build-core, pytest
├── core/
│   ├── include/mlite/
│   │   ├── core/                  # MatrixView, VectorView, future shared C++ primitives
│   │   └── <category>/            # public C++ algorithm headers
│   ├── src/<category>/            # C++ algorithm implementations
│   └── bindings/
│       ├── module.cpp             # initialization point of the single _core module
│       └── <category>/            # one pybind11 binding per public item
├── python/mlite/
│   ├── _core.pyi                  # typing surface of compiled _core
│   ├── _typing/<category>.pyi     # declarations of category's public classes/functions
│   ├── <category>/
│   │   ├── __init__.py            # re-exports and exact __all__
│   │   └── <snake_name>.py        # user-facing Python wrapper
│   ├── utils/                     # common internal Python helpers
│   ├── __init__.py                # package root (currently empty)
│   └── py.typed                   # package ships typing information
├── tests/<category>/test_<snake_name>.py
├── docs/lib/<category>/<snake_name>.md
└── docs/dev/                      # maintainer documentation
```

**[Inferred]** One algorithm has one identically named file at each applicable layer. For example, `linear_regression` consists of five files: `.hpp`, `.cpp`, `py_*.cpp`, a Python wrapper, and a test; documentation is the sixth. Do not combine unrelated public capabilities in one binding or wrapper.

## 3. Fundamental data contracts

### Feature matrix `X`

- **[Explicit]** The public Python API accepts array-like input; `_validate_X` calls `np.asarray(X, dtype=np.float64)`.
- **[Explicit]** A one-dimensional `X` becomes a column of shape `(n_samples, 1)`; a two-dimensional input is retained; any other dimensionality raises `ValueError`.
- **[Explicit]** `X.shape[0]` cannot be zero.
- **[Inferred]** C++ assumes a row-major contiguous array: the address of a row is `data + row * cols`. Use `py::array_t<double, py::array::c_style | py::array::forcecast>` for every binding argument that receives `X`. It both casts the dtype and guarantees C layout.

### Target vector `y`

- **[Explicit]** `_validate_y` casts to `float64`, accepts only 1D input, and rejects empty arrays.
- **[Explicit]** The wrapper must check `X.shape[0] == y.shape[0]` before calling the core.
- **[Inferred]** Domain-specific semantic rules belong in the wrapper; for example, binary classification checks for values `0.0`/`1.0` before C++ is called.

### Scalar parameters

- **[Explicit]** Current shared validators are intentionally strict: `learning_rate` and `test_size` must be exactly `float`; `epochs` and `random_state` must be exactly `int`.
- **[Explicit]** `learning_rate` is finite and `> 0`; `epochs` is `> 0`; `test_size` is strictly between 0 and 1; `random_state` is `>= 0`.
- **[Recommendation]** Add a validator to `utils/_validators.py` when the rule is reusable or part of API consistency. Check finiteness for every float that affects computation.

### `MatrixView` and `VectorView`

Neither type owns memory. `MatrixView(const double*, rows, cols)` and `VectorView(const double*, size)` receive only a pointer and metadata, rejecting `nullptr` for non-zero sizes. Their index operators check bounds. Their data is valid only during the synchronous pybind11 call.

**Rule:** never store a `MatrixView`, `VectorView`, `data()` pointer, or NumPy-buffer pointer in a C++ object's fields, and never return one after the call. If a result must survive the call, copy it into a `std::vector` or scalar.

## 4. API and naming conventions

| Entity | Format | Example |
| --- | --- | --- |
| Directory and file name | `snake_case` in a plural/domain section | `linear_models`, `train_test_split.py` |
| C++ namespace | `mlite` | `mlite::LogisticRegression` |
| C++ class and Python public class | matching `PascalCase` | `LogisticRegression` |
| C++ and Python free function | matching `snake_case` | `train_test_split` |
| Binding file/function | `py_<snake_name>.cpp` / `bind_<snake_name>` | `py_train_test_split.cpp`, `bind_train_test_split` |
| C++ header/implementation | `<snake_name>.hpp` / `<snake_name>.cpp` | `linear_regression.hpp` |
| Core object in a wrapper | `_` + public name | `_LinearRegression` |
| Internal Python symbols/files | leading `_` | `_validate_X`, `_decorators.py` |
| Learned attributes | trailing `_` | `coef_`, `intercept_`, `n_features_in_` |

**[Inferred]** A public import must work from the category without exposing its file: `from mlite.linear_models import LinearRegression`. Add every public item to the category's `__init__.py` and `__all__`; do not direct users to import `_core`.

## 5. Project patterns

1. **Facade / Adapter.** A Python wrapper hides the pybind11 ABI and adapts the Python/NumPy contract to C++.
2. **Thin binding.** The binding only constructs views, calls C++, and converts results; shape validation, fitted state, and domain rules belong above it.
3. **Value-oriented core.** C++ returns `std::vector` or a struct containing `std::vector`; the binding creates an independent NumPy result.
4. **Shared metrics.** Model `score` methods should delegate to the shared `metrics` category rather than re-implementing evaluation formulas internally.
5. **Explicit registration.** New code is not discovered automatically. It must be added to CMake and called from `module.cpp`.
6. **Stateful estimator.** A model starts unfitted, `fit` replaces its learned parameters, read methods use `_ensure_fitted`, and `state_dict`/`load_state_dict` transfer complete state.
7. **Single public-validation source.** The Python wrapper validates input before C++ and applies equivalent checks in `fit`, prediction methods, and `score`.

## 6. Type A: trainable model class

### Required files

For `NewModel` in `linear_models`, create or change all of the following:

```text
core/include/mlite/linear_models/new_model.hpp       # public C++ declaration
core/src/linear_models/new_model.cpp                 # algorithm
core/bindings/linear_models/py_new_model.cpp         # pybind11 adapter
python/mlite/linear_models/new_model.py              # public wrapper
python/mlite/_typing/linear_models.pyi               # class declaration
tests/linear_models/test_new_model.py                # public-contract tests
docs/lib/linear_models/new_model.md                  # user documentation
core/bindings/module.cpp                             # declaration + bind_new_model(m)
CMakeLists.txt                                       # two new .cpp files in _core target
python/mlite/linear_models/__init__.py               # re-export + __all__
python/mlite/_core.pyi                               # domain-stub re-export, if necessary
README.md                                            # link if the feature is indexed publicly
```

Do not add a header to `CMakeLists.txt`: only `.cpp` files are compiled.

### Lifecycle and contract

```text
__init__ → unfitted
fit(X, y) → Python validation → C++ fit → fitted
predict / score / learned properties / state_dict → allowed only while fitted
load_state_dict(state) → C++ load_state → fitted
fit(...) again → replaces learned state with newly trained state
```

**[Inferred]** Existing model `fit` methods return `None`, not `self`; retain this convention. `_fitted` is held by the wrapper; C++ has no separate fitted flag.

### C++ API template

```cpp
#pragma once

#include <cstddef>
#include <vector>

#include "mlite/core/matrix_view.hpp"
#include "mlite/core/vector_view.hpp"

namespace mlite {
  class NewModel {
    private:
        std::vector<double> coef_;
        double intercept_;
        std::size_t n_features_in_;
    public:
        NewModel();
        void fit(const MatrixView& X, const VectorView& y,
                 double learning_rate, std::size_t epochs);
        std::vector<double> predict(const MatrixView& X) const;
        double score(const MatrixView& X, const VectorView& y) const;
        void load_state(const std::vector<double>& coef, double intercept,
                        std::size_t n_features_in);
        const std::vector<double>& get_coef() const;
        double get_intercept() const;
        std::size_t get_n_features_in() const;
    };
}
```
 
**[Inferred]** A trainable model stores only parameters needed for prediction and the feature count. Training settings such as `learning_rate` and `epochs` stay in the Python wrapper and are included in serialized state. A new algorithm may have different state, but everything needed for an identical later result must be included in both `state_dict` and `load_state`.
 
**[Recommendation]** If a model implements `score`, do not duplicate metric logic inside the estimator. Instead, delegate to a shared metric function in the `metrics` category, for example `mlite::r2_score` or `mlite::accuracy_score`, and document the shared metric in `docs/lib/metrics/metrics.md`. This keeps evaluation behavior consistent across models and public metric helpers.

### Binding template

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "mlite/linear_models/new_model.hpp"

namespace py = pybind11;

void bind_new_model(py::module_& m) {
    py::class_<mlite::NewModel>(m, "NewModel")
        .def(py::init<>())
        .def("fit", [](mlite::NewModel& self,
                         py::array_t<double, py::array::c_style | py::array::forcecast> X,
                         py::array_t<double, py::array::c_style | py::array::forcecast> y,
                         double learning_rate, std::size_t epochs) {
             const auto xb = X.request(); const auto yb = y.request();
             self.fit(mlite::MatrixView(static_cast<const double*>(xb.ptr),
                                        static_cast<std::size_t>(xb.shape[0]),
                                        static_cast<std::size_t>(xb.shape[1])),
                      mlite::VectorView(static_cast<const double*>(yb.ptr),
                                        static_cast<std::size_t>(yb.shape[0])),
                      learning_rate, epochs);
        }, py::arg("X"), py::arg("y"), py::arg("learning_rate"), py::arg("epochs"));
}
```

Extend the chain with `.def(...)` for every C++ method and `.def_property_readonly(...)` for learned state. Vector results may be returned as `std::vector` with `pybind11/stl.h`; the wrapper must cast the public result to its documented `np.ndarray`/dtype. For a raw NumPy result, create an owning array; never return an array that points to a temporary `std::vector`.

### Python wrapper template

```python
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from .._core import NewModel as _NewModel  # type: ignore
from ..utils._decorators import _ensure_fitted
from ..utils._validators import _validate_X, _validate_y

class NewModel:
    def __init__(self, *, setting: float = 1.0) -> None:
        self._model = _NewModel()
        self.setting = _validate_setting(setting)
        self._fitted = False

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        X, y = _validate_X(X), _validate_y(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        # Domain validation of y belongs here.
        self._model.fit(X, y, self.setting)
        self._fitted = True

    @_ensure_fitted
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        X = _validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has different number of features")
        return np.asarray(self._model.predict(X), dtype=np.float64)

    @_ensure_fitted
    def state_dict(self) -> dict:
        return {"coef": self.coef_, "intercept": self.intercept_,
                "n_features_in": self.n_features_in_, "setting": self.setting}

    def load_state_dict(self, state: dict) -> None:
        self._model.load_state(np.asarray(state["coef"], dtype=np.float64),
                               float(state["intercept"]), int(state["n_features_in"]))
        self.setting = _validate_setting(state["setting"])
        self._fitted = True
```

Use the exact same state keys in documentation, tests, and the loader. **[Recommendation]** Validate required keys, parameter shapes, and values in `load_state_dict` before mutating the native object. Existing modules use direct dictionary access, so this improves rather than changes the legacy guarantee.

## 7. Type B: stateless core function

Use this type for an operation that does not retain trained state: a split, metric, transform, or deterministic utility.

### Required files

```text
core/include/mlite/<category>/<function>.hpp
core/src/<category>/<function>.cpp
core/bindings/<category>/py_<function>.cpp
python/mlite/<category>/<function>.py
python/mlite/_typing/<category>.pyi
tests/<category>/test_<function>.py
docs/lib/<category>/<function>.md
CMakeLists.txt; core/bindings/module.cpp; python/mlite/<category>/__init__.py
```

### Contract and result

- C++ declares a free function in `namespace mlite` and accepts `const MatrixView&`, `const VectorView&`, and simple scalars.
- For multiple results, use a simple result struct containing values and shape metadata, as `TrainTestSplit` does.
- The binding creates independent NumPy arrays. When a temporary C++ buffer is the source, call `.copy()` before the struct is destroyed.
- The Python function runs all validators, checks input compatibility, and imports the native function as a private alias: `from .._core import name as _name`.
- The metrics category is a shared implementation location for evaluation helpers. A model's `score` method should call into this category when appropriate.
- The category re-exports only the public name through `__all__`.

### Lifecycle

```text
call → validate/normalize → native call → materialize independent Python result → return
```

This function has no `_fitted`, `state_dict`, or `load_state_dict`. A seed parameter must produce reproducible output: `train_test_split` uses `std::mt19937(random_state)`.

## 8. Type C: Python-only shared helper

This is not a public module type in the current architecture. Use it for recurring wrapper-layer logic that does not need C++ acceleration.

```text
python/mlite/utils/_<name>.py
tests/... (if the helper has an independent contract, otherwise through public tests)
```

- Give the file and symbols a leading `_`.
- Do not add it to `utils/__init__.py` or public `__all__` without a separate API decision.
- Do not import a helper from C++, and do not put an algorithm in it if it belongs in the core.
- If it validates data, it must return the normalized value or raise a clear `TypeError`/`ValueError`.

**[Inferred]** `utils` is internal because all its files/symbols use leading underscores and its `__init__.py` is empty.

## 9. Registration: mandatory sequence

A new file does not become part of the package automatically. For every public core-backed item, follow these steps in order:

1. Select an existing category, or create a new category symmetrically at every layer.
2. Declare the C++ API in `core/include/mlite/<category>/<name>.hpp`.
3. Implement it in `core/src/<category>/<name>.cpp`.
4. Create `core/bindings/<category>/py_<name>.cpp` with `bind_<name>(py::module_&)`.
5. In `core/bindings/module.cpp`, add the forward declaration `void bind_<name>(py::module_& m);` and call `bind_<name>(m);` inside `PYBIND11_MODULE(_core, m)`.
6. Ensure the C++ implementation (`core/src/<category>/<name>.cpp`) is listed in `pybind11_add_module(_core ...)` in `CMakeLists.txt`. Binding files under `core/bindings/` are discovered automatically.
7. Write the wrapper and import the C++ symbol under a private alias.
8. Add the public re-export and update `__all__` in `python/mlite/<category>/__init__.py`.
9. Add or update the domain stub at `python/mlite/_typing/<category>.pyi`, then ensure `python/mlite/_core.pyi` exposes it. The current `_core.pyi` scheme aggregates domain stubs.
10. Create tests and user documentation. Add a README link when the category is included in the feature index.
11. Rebuild the package and execute tests against the installed/editable package.

### Minimum registration changes

```cpp
// core/bindings/module.cpp
void bind_new_model(py::module_& m);
// ... inside PYBIND11_MODULE
bind_new_model(m);
```

```cmake
# CMakeLists.txt, inside pybind11_add_module(_core ...)
core/src/linear_models/new_model.cpp
```

```python
# python/mlite/linear_models/__init__.py
from .new_model import NewModel
__all__ = ["LinearRegression", "LogisticRegression", "NewModel"]
```

## 10. Typing, documentation, and tests

### Typing

- `py.typed` already includes typing information in the distributed package.
- Specify public signatures, public learned attributes, and precise NumPy dtypes in `python/mlite/_typing/<category>.pyi`.
- Do not declare the Python-private `_model` or other implementation details as public API.
- **[Inferred]** Stubs must describe the wrapper API, including `state_dict`/`load_state_dict`, not only the native binding.

> The current `python/mlite/_core.pyi` uses imports prefixed with `python.mlite`, although the expected package-relative form is `from ._typing...`. This does not affect runtime, but check a type checker and, where possible, correct the path in a separate compatible change when modifying typing. Do not copy the prefix without checking it.

### User documentation

**[Explicit]** Every public Python source has a corresponding `docs/lib/<category>/<name>.md` file mirroring the library structure. Each page includes:

1. purpose;
2. mathematical model or algorithm, where applicable;
3. construction or signature;
4. methods, parameters, and results;
5. a minimal and a practical example;
6. learned attributes or returned data;
7. limitations, exceptions, and reproducibility details.

Document actual public behavior: shape, dtype, defaults, output, error conditions, and state before/after `fit`.

### Minimum test coverage

Test imports from the public category, never `_core`.

| Model | Function |
| --- | --- |
| default and custom parameters | correct shape and dtype for every result |
| successful `fit`, predict, and score | preservation of data/semantics |
| 1D and ordinary 2D `X` | reproducibility with a fixed seed |
| predict/score/properties before fit → `RuntimeError` | changed seed changes the result when expected |
| invalid dimensionality, empty data, sample count, and feature count | invalid types, parameter boundaries, and sample mismatch |
| domain restrictions for y/threshold | returned arrays are independent of native temporary storage |
| `state_dict` contains all state; restored model gives equal output | — |

**[Explicit]** `pytest` discovers tests in `tests`; run `python -m pytest`. The project also declares `ruff` and `black` as development dependencies, but no rules are configured in `pyproject.toml`; do not claim a formatting standard that does not exist.

## 11. Build and local development

Packaging requirements are Python >= 3.9, NumPy >= 1.24, CMake >= 3.18, pybind11 >= 2.12, Ninja, and a C++17 compiler. `scikit-build-core` builds a wheel/editable installation; CMake creates `_core`, which installs into package `mlite`.

```powershell
python -m pip install -e ".[dev]"
python -m pytest
```

After changing `.cpp`, `CMakeLists.txt`, or a binding, reinstall/rebuild the package before testing. Otherwise Python can import an older compiled `_core`. CMake builds into `build/{wheel_tag}`; `build/` is an artifact and not part of the source API.

## 12. Best practices

- Design the Python contract first: signature, dtype, shape, errors, state, and serialization. Then move only the necessary work to C++.
- Keep the native API narrow: `MatrixView`, `VectorView`, scalar parameters, and value results.
- Validate dimensionality before a binding accesses `shape[1]`. A binding assumes the Python wrapper has already passed a valid 2D `X`.
- Return `float64` for continuous predictions and explicitly cast class labels to `int64` when that is the public contract.
- Use numerically stable formulas for sensitive functions; the existing logistic regression provides a stable-sigmoid example.
- Explicitly define behavior for constant targets, parameter limits, and empty data; add tests before fixing behavior.
- Keep argument order identical across Python wrapper, binding, C++ header, and stub.
- Keep errors stable and clear: Python type/domain errors are `TypeError`/`ValueError`; use of an unfitted model is `RuntimeError`.
- Document complexity, randomness, seeds, and meaningful data copies.

## 13. Frequent errors and prevention

| Error | Symptom | Prevention |
| --- | --- | --- |
| File created but CMake registration omitted | linker/import lacks the symbol | add implementation and binding `.cpp` files to the `_core` target |
| CMake updated but `bind_*` omitted from `module.cpp` | build succeeds but `_core` has no attribute | declare and call `bind_*` |
| Binding reads `shape[1]` for 1D/3D input | unsafe native exception | wrapper always calls `_validate_X`; binding uses C-contiguous `array_t` |
| NumPy view returned over `std::vector` | corrupted/dangling data after return | create an owning array or call `.copy()` |
| Validation only in `fit` | predict/score accepts incompatible feature counts | validate X and `X.shape[1]` in all read methods |
| `_ensure_fitted` omitted | silently invalid predictions from empty C++ state | decorate predict, score, learned properties, and `state_dict` |
| Partial state persistence | `load_state_dict` cannot reproduce the model | save learned parameters, feature count, and configuration |
| Public symbol absent from `__all__` | documented import fails or is undiscoverable | re-export it from category `__init__.py` |
| Stub differs from wrapper | IDE/type checker exposes a false API | update the stub in the same change |
| Documentation describes C++ rather than Python | users need source code to use the API | document only the public signature and contract |

## 14. Final checklist

### Architecture and build

- [ ] An existing type was chosen: model, stateless function, or internal helper.
- [ ] C++ header and implementation are in mirrored category paths.
- [ ] A separate `py_<name>.cpp` and `bind_<name>` exist.
- [ ] The binding is registered in `module.cpp`.
- [ ] Both new `.cpp` files appear in `pybind11_add_module`.
- [ ] The header was not added to CMake as a compilation source.

### Public API

- [ ] Names match at C++, binding, and Python layers according to section 4.
- [ ] The wrapper hides `_core` and fully validates input before the native call.
- [ ] `X`, `y`, parameters, and feature compatibility are validated in every applicable public method.
- [ ] Results have documented shapes and dtypes.
- [ ] A model has fitted lifecycle, learned properties, `state_dict`, and `load_state_dict`.
- [ ] A function returns memory-owning results that do not depend on C++ temporaries.
- [ ] The symbol is in the category's `__init__.py` and `__all__`.

### Quality and delivery

- [ ] The domain `.pyi` was updated and its path from `_core.pyi` was checked.
- [ ] `tests/<category>/test_<name>.py` covers positive and negative cases.
- [ ] `docs/lib/<category>/<name>.md` follows the documentation standard.
- [ ] The README link was updated when appropriate.
- [ ] Editable rebuild/install and `python -m pytest` were run.
- [ ] The change does not include `build/`, cache, or machine-specific artifacts.

## 15. Decisions requiring a separate project convention

The existing code does not define the following areas. Do not make a silent choice for a substantial new module:

- a new top-level category and its public import;
- multi-class/multi-output shapes and label dtypes;
- sparse arrays, pandas, GPU support, parallelism, or zero-copy ownership;
- a base estimator/protocol, inheritance hierarchy, or shared serialization format/versioning;
- logging, warnings, exception taxonomy beyond current `TypeError`/`ValueError`/`RuntimeError`;
- formatting/linting CI and minimum coverage/performance thresholds.

For each such decision, first add an explicit API contract, tests, and an update to this guide. This promotes a current **[Inferred]** convention into a maintained project rule.