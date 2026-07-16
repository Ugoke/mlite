#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "mlite/model_selection/stratified_shuffle_split.hpp"

namespace py = pybind11;

void bind_stratified_shuffle_split(py::module_& m) {
    m.def("stratified_shuffle_split", [](
            py::array_t<double,
            py::array::c_style | py::array::forcecast> X,
            py::array_t<double,
            py::array::c_style | py::array::forcecast> y,
            double test_size,
            std::uint32_t random_state
        ) {
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

            auto split = mlite::stratified_shuffle_split(
                x_view,
                y_view,
                test_size,
                random_state
            );

            py::array_t<double> X_train(
                {split.train_rows, split.n_features},
                split.X_train.data()
            );

            py::array_t<double> X_test(
                {split.test_rows, split.n_features},
                split.X_test.data()
            );

            py::array_t<double> y_train(
                split.train_rows,
                split.y_train.data()
            );

            py::array_t<double> y_test(
                split.test_rows,
                split.y_test.data()
            );

            return py::make_tuple(
                X_train.attr("copy")(),
                X_test.attr("copy")(),
                y_train.attr("copy")(),
                y_test.attr("copy")()
            );
        },
        py::arg("X"),
        py::arg("y"),
        py::arg("test_size") = 0.2,
        py::arg("random_state") = 42
    );
}