# MLite Project Structure Guide

## Architecture

**mlite** — This is an ML library with a hybrid architecture:
- **C++ core** — Fast computations using pybind11 (`_core`)
- **Python wrappers** — typing, validation, user-friendly API

## How to Add a New Model—A Complete Example with Multiple Templates

Below are detailed templates and code examples for each step.

### 1. Type stub for C++ (`python/mlite/_typing/new_model.pyi`)

A .pyi file (stub file) describes the interface of a C++ class that is exported via pybind11.
A .pyi file does not contain method implementations and does not participate in the execution of the program. 
It is intended solely to improve the experience of developing and documenting the C++ module interface for Python tools.

```python
# python/mlite/_typing/new_model.pyi
import numpy as np
from numpy.typing import NDArray

class NewModel:
    coef_: NDArray[np.float64]
    intercept_: float
    n_features_in_: int

    def __init__(self) -> None: ...

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        learning_rate: float,
        epochs: int,
    ) -> None: ...

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def score(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> float: ...
    def load_state(self, coef: NDArray[np.float64], intercept: float, n_features_in: int) -> None: ...

    # You'll need getters in C++/pybind11, and they'll appear in the pyi file
```

---

### 2. C++ header (core/include/mlite/linear_models/new_model.hpp)

The header file contains the declaration of the model class and its public interface.

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

            double predict_sample(const MatrixView& X, std::size_t row) const;

        public:
            NewModel();

            void fit(const MatrixView& X, const VectorView& y, double learning_rate, std::size_t epochs);

            std::vector<double> predict(const MatrixView& X) const;

            double score(const MatrixView& X, const VectorView& y) const;

            const std::vector<double>& get_coef() const;
            double get_intercept() const;
            std::size_t get_n_features_in() const;

            void load_state(const std::vector<double>& coef, double intercept, std::size_t n_features_in);
    };
}
```

---

### 3. C++ implementation (core/src/linear_models/new_model.cpp)

An example of implementing basic methods (gradient descent; the template is similar to LinearRegression in the project).

```cpp
#include "mlite/linear_models/new_model.hpp"
#include <algorithm>

namespace mlite {
    NewModel::NewModel()
        : coef_(), intercept_(0.0), n_features_in_(0) {}

    double NewModel::predict_sample(const MatrixView& X, std::size_t row) const {
        const double* x_row = X.data() + (row * n_features_in_);
        double prediction = intercept_;
        for (std::size_t j = 0; j < n_features_in_; ++j) {
            prediction += coef_[j] * x_row[j];
        }
        return prediction;
    }

    void NewModel::fit(const MatrixView& X, const VectorView& y, double learning_rate, std::size_t epochs) {
        const std::size_t samples = X.rows();
        const std::size_t features = X.cols();
        coef_.assign(features, 0.0);
        intercept_ = 0.0;
        n_features_in_ = features;

        std::vector<double> gradients(features);
        const double inv_samples = 2.0 / static_cast<double>(samples);

        for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
            std::fill(gradients.begin(), gradients.end(), 0.0);
            double bias_grad = 0.0;
            for (std::size_t i = 0; i < samples; ++i) {
                const double prediction = predict_sample(X, i);
                const double err = prediction - y(i);
                const double* x_row = X.data() + i * features;
                for (std::size_t j = 0; j < features; ++j) {
                    gradients[j] += err * x_row[j];
                }
                bias_grad += err;
            }
            for (std::size_t j = 0; j < features; ++j) {
                coef_[j] -= learning_rate * gradients[j] * inv_samples;
            }
            intercept_ -= learning_rate * bias_grad * inv_samples;
        }
    }

    std::vector<double> NewModel::predict(const MatrixView& X) const {
        const std::size_t samples = X.rows();
        std::vector<double> out(samples);
        for (std::size_t i = 0; i < samples; ++i) {
            out[i] = predict_sample(X, i);
        }
        return out;
    }

    double NewModel::score(const MatrixView& X, const VectorView& y) const {
        const std::size_t n = y.size();
        double mean = 0.0;
        for (std::size_t i = 0; i < n; ++i) mean += y(i);
        mean /= static_cast<double>(n);
        double ss_tot = 0.0;
        double ss_res = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double pred = predict_sample(X, i);
            double diff = y(i) - pred;
            double tot = y(i) - mean;
            ss_res += diff * diff;
            ss_tot += tot * tot;
        }
        if (ss_tot == 0.0) return 0.0;
        return 1.0 - (ss_res / ss_tot);
    }

    const std::vector<double>& NewModel::get_coef() const { return coef_; }

    double NewModel::get_intercept() const { return intercept_; }

    std::size_t NewModel::get_n_features_in() const { return n_features_in_; }

    void NewModel::load_state(const std::vector<double>& coef, double intercept, std::size_t n_features_in) {
        coef_ = coef;
        intercept_ = intercept;
        n_features_in_ = n_features_in;
    }
}
```

---

### 4. Pybind11 binding (core/bindings/linear_models/py_new_model.cpp)

The binding file is responsible for exporting a C++ class to Python using pybind11.
It is used to:
- register the C++ class in the Python module _core;
- export the class’s constructors, methods, and attributes;
- convert data between Python and C++;
- work with numpy.ndarray arrays;
- converting NumPy buffers to the library’s internal types (MatrixView, VectorView);

It is the binding file that establishes the connection between Python and the C++ core. 
Without it, the class would exist only in C++ and could not be imported from Python.

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "mlite/linear_models/new_model.hpp"

namespace py = pybind11;

void bind_new_model(py::module_& m) {
    py::class_<mlite::NewModel>(m, "NewModel")
        .def(py::init<>())

        .def(
            "fit",
            [](mlite::NewModel& self,
               py::array_t<double, py::array::c_style | py::array::forcecast> X,
               py::array_t<double, py::array::c_style | py::array::forcecast> y,
               double learning_rate,
               std::size_t epochs) {

                auto x_buf = X.request();
                auto y_buf = y.request();

                mlite::MatrixView x_view(
                    static_cast<const double*>(x_buf.ptr),
                    static_cast<std::size_t>(x_buf.shape[0]),
                    static_cast<std::size_t>(x_buf.shape[1])
                );

                mlite::VectorView y_view(
                    static_cast<const double*>(y_buf.ptr),
                    static_cast<std::size_t>(y_buf.shape[0])
                );

                self.fit(x_view, y_view, learning_rate, epochs);
            },
            py::arg("X"), py::arg("y"), py::arg("learning_rate"), py::arg("epochs")
        )

        .def(
            "predict",
            [](const mlite::NewModel& self, py::array_t<double> X) {
                auto buf = X.request();
                mlite::MatrixView x_view(
                    static_cast<double*>(buf.ptr),
                    static_cast<std::size_t>(buf.shape[0]),
                    static_cast<std::size_t>(buf.shape[1])
                );
                std::vector<double> out = self.predict(x_view);
                py::array_t<double> result(out.size());
                auto r = result.mutable_unchecked<1>();
                for (std::size_t i = 0; i < out.size(); ++i) r(i) = out[i];
                return result;
            }
        )

        .def_property_readonly("coef_", [](const mlite::NewModel& self) { return self.get_coef(); })
        .def_property_readonly("intercept_", &mlite::NewModel::get_intercept)
        .def_property_readonly("n_features_in_", &mlite::NewModel::get_n_features_in);
}
```

And register `bind_new_model` in `core/bindings/module.cpp`.

---

### 5. Module Registration (core/bindings/module.cpp)

```cpp
#include <pybind11/pybind11.h>
namespace py = pybind11;

void bind_new_model(py::module_& m);

PYBIND11_MODULE(_core, m) {
    m.doc() = "mlite core";
    bind_new_model(m);
}
```

---

### 6. Python wrapper (python/mlite/linear_models/new_model.py)

The Python wrapper provides a high-level API on top of the C++ implementation of the model, exported via pybind11.

The wrapper is responsible for:
- a user-friendly interface;
- input data validation;
- NumPy type conversion;
- checking the model’s state;
- serialization and restoration of the model’s state;
- hiding the low-level C++ API from the user.

```python
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .._core import NewModel as _NewModel  # type: ignore
from ..utils._decorators import _ensure_fitted
from ..utils._constants import DEFAULT_LEARNING_RATE, DEFAULT_EPOCHS
from ..utils._validators import _validate_X, _validate_y, _validate_learning_rate, _validate_epochs

class NewModel:
    """
    NewModel example wrapper.

    Parameters
    ----------
    learning_rate : float
        Learning rate for gradient updates.
    epochs : int
        Number of training iterations.
    """

    def __init__(self, learning_rate: float = DEFAULT_LEARNING_RATE, epochs: int = DEFAULT_EPOCHS) -> None:
        self._model = _NewModel()
        self.learning_rate = _validate_learning_rate(learning_rate)
        self.epochs = _validate_epochs(epochs)
        self._fitted = False

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        X = _validate_X(X)
        y = _validate_y(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")

        self._model.fit(X, y, self.learning_rate, self.epochs)
        self._fitted = True

    @_ensure_fitted
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        X = _validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has different number of features")
        return self._model.predict(X)

    @_ensure_fitted
    def score(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> float:
        X = _validate_X(X)
        y = _validate_y(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        return self._model.score(X, y)

    def state_dict(self) -> dict:
        return {
            "coef": np.asarray(self._model.coef_, dtype=np.float64),
            "intercept": float(self._model.intercept_),
            "n_features_in": int(self._model.n_features_in_),
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
        }

    def load_state_dict(self, state: dict) -> None:
        self._model.load_state(
            np.asarray(state["coef"], dtype=np.float64),
            float(state["intercept"]),
            int(state["n_features_in"]),
        )
        self.learning_rate = float(state["learning_rate"])
        self.epochs = int(state["epochs"])
        self._fitted = True

    @property
    @_ensure_fitted
    def coef_(self) -> NDArray[np.float64]:
        return np.asarray(self._model.coef_, dtype=np.float64)

    @property
    @_ensure_fitted
    def intercept_(self) -> float:
        return float(self._model.intercept_)

    @property
    @_ensure_fitted
    def n_features_in_(self) -> int:
        return int(self._model.n_features_in_)
```

---

### 7. Export Module (python/mlite/linear_models/__init__.py)

```python
from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression
from .new_model import NewModel

__all__ = ["LinearRegression", "LogisticRegression", "NewModel"]
```

---

### 8. Tests (tests/linear_models/test_new_model.py)

It is recommended to create a corresponding test file for each source code file.

Example:

```
python/mlite/linear_models/linear_regression.py
→ tests/linear_models/test_linear_regression.py
```

Test examples: functional and negative test cases.

```python
import numpy as np
import pytest
from mlite.linear_models import NewModel


def test_fit_predict_basic():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 6.0])
    model = NewModel(learning_rate=0.01, epochs=100)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (3,)


def test_predict_before_fit_raises():
    model = NewModel()
    with pytest.raises(RuntimeError):
        model.predict(np.array([[1.0]]))


def test_feature_mismatch_raises():
    X = np.array([[1.0], [2.0]])
    y = np.array([1.0, 2.0])
    model = NewModel()
    model.fit(X, y)
    with pytest.raises(ValueError):
        model.predict(np.array([[1.0, 2.0]]))
```

---

### 9. CMake and assembly (tips)

`mlite` uses:
- `scikit-build-core` to integrate Python and CMake;
- `pybind11` to create the `_core` Python extension module;
- `CMake` to build the library's C++ core.

The build configuration is located at:

```
CMakeLists.txt
pyproject.toml
```

#### Local Development

We recommend using editable mode for development:

```bash
python -m pip install -e .[dev]
```

#### Registering a New Model in the Build

Suppose the following files have been created:

```
core/include/mlite/linear_models/new_model.hpp
core/src/linear_models/new_model.cpp
core/bindings/linear_models/py_new_model.cpp
```

The header file (`.hpp`) is not added to `CMakeLists.txt`.
Only new `.cpp` files need to be registered.

In that case, you need to add them to the call:

```c
pybind11_add_module(
    _core
    ...

    core/bindings/linear_models/py_new_model.cpp
    core/src/linear_models/new_model.cpp
)
```