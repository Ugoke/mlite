#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "mlite/metrics/mean_absolute_error.hpp"

namespace py = pybind11;

void bind_mean_absolute_error(py::module_& m) {
    m.def("mean_absolute_error", [](
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

            return mlite::mean_absolute_error(y_true_view, y_pred_view);
        },
        py::arg("y_true"),
        py::arg("y_pred")
    );
}
