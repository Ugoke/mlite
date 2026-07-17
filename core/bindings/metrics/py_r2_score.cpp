#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "mlite/metrics/r2_score.hpp"

namespace py = pybind11;

void bind_r2_score(py::module_& m) {
    m.def("r2_score", [](
            py::array_t<double,
            py::array::c_style | py::array::forcecast> y_true,
            py::array_t<double,
            py::array::c_style | py::array::forcecast> y_pred
        ) {
            auto y_true_buf = y_true.request();
            auto y_pred_buf = y_pred.request();

            mlite::VectorView y_true_view(
                static_cast<const double*>(y_true_buf.ptr),
                static_cast<std::size_t>(y_true_buf.shape[0])
            );

            mlite::VectorView y_pred_view(
                static_cast<const double*>(y_pred_buf.ptr),
                static_cast<std::size_t>(y_pred_buf.shape[0])
            );

            return mlite::r2_score(
                y_true_view,
                y_pred_view
            );
        },
        py::arg("y_true"),
        py::arg("y_pred")
    );
}