#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "mlite/linear_models/logistic_regression.hpp"


namespace py = pybind11;

void bind_logistic_regression(py::module_& m) {
    py::class_<mlite::LogisticRegression>(m, "LogisticRegression")
        .def(py::init<>())

        .def(
            "fit",
            [](
                mlite::LogisticRegression& self,

                py::array_t<double, py::array::c_style | py::array::forcecast> X,
                py::array_t<double, py::array::c_style | py::array::forcecast> y,

                double learning_rate,
                std::size_t epochs
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

                self.fit(
                    x_view,
                    y_view,
                    learning_rate,
                    epochs
                );
            },

            py::arg("X"),
            py::arg("y"),
            py::arg("learning_rate"),
            py::arg("epochs")
        )

        .def(
            "predict_proba",
            [](
                const mlite::LogisticRegression& self,
                
                py::array_t<double, py::array::c_style | py::array::forcecast> X
            ) {
                auto x_buf = X.request();

                mlite::MatrixView x_view(
                    static_cast<const double*>(x_buf.ptr),
                    static_cast<std::size_t>(x_buf.shape[0]),
                    static_cast<std::size_t>(x_buf.shape[1])
                );

                return self.predict_proba(x_view);
            },

            py::arg("X")
        )

        .def(
            "predict",
            [](
                const mlite::LogisticRegression& self,

                py::array_t<double, py::array::c_style | py::array::forcecast> X,

                double threshold
            ) {
                auto x_buf = X.request();

                mlite::MatrixView x_view(
                    static_cast<const double*>(x_buf.ptr),
                    static_cast<std::size_t>(x_buf.shape[0]),
                    static_cast<std::size_t>(x_buf.shape[1])
                );

                return self.predict(
                    x_view,
                    threshold
                );
            },

            py::arg("X"),
            py::arg("threshold")
        )

        .def(
            "score",
            [](
                const mlite::LogisticRegression& self,

                py::array_t<double, py::array::c_style | py::array::forcecast> X,
                py::array_t<double, py::array::c_style | py::array::forcecast> y,

                double threshold
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

                return self.score(
                    x_view,
                    y_view,
                    threshold
                );
            },

            py::arg("X"),
            py::arg("y"),
            py::arg("threshold")
        )

        .def(
            "load_state",
            &mlite::LogisticRegression::load_state,
            py::arg("coef"),
            py::arg("intercept"),
            py::arg("n_features_in")
        )

        .def_property_readonly("coef_", &mlite::LogisticRegression::get_coef)

        .def_property_readonly("intercept_", &mlite::LogisticRegression::get_intercept)

        .def_property_readonly("n_features_in_", &mlite::LogisticRegression::get_n_features_in);
}