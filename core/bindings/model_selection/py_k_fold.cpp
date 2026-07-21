#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "mlite/model_selection/k_fold.hpp"

namespace py = pybind11;

void bind_k_fold(py::module_& m) {
    m.def("k_fold", [](
            py::array_t<double,
            py::array::c_style | py::array::forcecast> X,
            py::array_t<double,
            py::array::c_style | py::array::forcecast> y,
            std::size_t n_splits,
            bool shuffle,
            std::uint32_t random_state
        ) {
            auto x_buf = X.request();
            auto y_buf = y.request();

            mlite::MatrixView x_view(static_cast<const double*>(x_buf.ptr), static_cast<std::size_t>(x_buf.shape[0]), static_cast<std::size_t>(x_buf.shape[1]));
            mlite::VectorView y_view(static_cast<const double*>(y_buf.ptr), static_cast<std::size_t>(y_buf.shape[0]));

            auto folds = mlite::k_fold(x_view, y_view, n_splits, shuffle, random_state);

            py::list result;

            for (auto& fold : folds) {
                py::array_t<double> X_train(
                    {
                        static_cast<py::ssize_t>(fold.train_rows),
                        static_cast<py::ssize_t>(fold.n_features)
                    },
                    fold.X_train.data()
                );

                py::array_t<double> X_test(
                    {
                        static_cast<py::ssize_t>(fold.test_rows),
                        static_cast<py::ssize_t>(fold.n_features)
                    },
                    fold.X_test.data()
                );

                py::array_t<double> y_train(
                    static_cast<py::ssize_t>(fold.train_rows),
                    fold.y_train.data()
                );

                py::array_t<double> y_test(
                    static_cast<py::ssize_t>(fold.test_rows),
                    fold.y_test.data()
                );

                result.append(
                    py::make_tuple(
                        X_train.attr("copy")(),
                        X_test.attr("copy")(),
                        y_train.attr("copy")(),
                        y_test.attr("copy")()
                    )
                );
            }

            return result;
        },
        py::arg("X"),
        py::arg("y"),
        py::arg("n_splits"),
        py::arg("shuffle"),
        py::arg("random_state")
    );
}