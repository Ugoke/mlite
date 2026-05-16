#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "mlite/linear_regression.hpp"

namespace py = pybind11;

void bind_linear(py::module_& m) {
    py::class_<mlite::LinearRegression>(m, "LinearRegression")
        .def(py::init<>())

        .def(
            "fit",
            &mlite::LinearRegression::fit,
            py::arg("X"),
            py::arg("y"),
            py::arg("learning_rate"),
            py::arg("epochs")
        )
        
        .def(
            "predict",
            &mlite::LinearRegression::predict,
            py::arg("X")
        );
}