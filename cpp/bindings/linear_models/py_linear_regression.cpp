#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "mlite/linear_models/linear_regression.hpp"


namespace py = pybind11;

void bind_linear_regression(py::module_& m) {
    py::class_<mlite::LinearRegression>(m, "LinearRegression")
        .def(py::init<>())

        .def(
            "fit",
            [](mlite::LinearRegression& self,
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
            py::arg("X"),
            py::arg("y"),
            py::arg("learning_rate"),
            py::arg("epochs")
        )

        .def(
            "predict",
            [](const mlite::LinearRegression& self,
                py::array_t<double> X) {
                auto buf = X.request();
            
                mlite::MatrixView x_view(
                    static_cast<double*>(buf.ptr),
                    buf.shape[0],
                    buf.shape[1]
                );
            
                std::vector<double> predictions = self.predict(x_view);
            
                py::array_t<double> result(predictions.size());
            
                auto out = result.mutable_unchecked<1>();
            
                for (std::size_t i = 0; i < predictions.size(); ++i) {
                    out(i) = predictions[i];
                }
            
                return result;
            }
        )

        .def(
            "score",
            [](const mlite::LinearRegression& self,
               py::array_t<double, py::array::c_style | py::array::forcecast> X,
               py::array_t<double, py::array::c_style | py::array::forcecast> y) {

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

                return self.score(x_view, y_view);
            },
            py::arg("X"),
            py::arg("y")
        )

        .def(
            "load_state",
            &mlite::LinearRegression::load_state,
            py::arg("coef"),
            py::arg("intercept"),
            py::arg("n_features_in")
        )

        .def_property_readonly("coef_", [](const mlite::LinearRegression& self) {
            return self.get_coef();
        })

        .def_property_readonly("intercept_", &mlite::LinearRegression::get_intercept)

        .def_property_readonly("n_features_in_", &mlite::LinearRegression::get_n_features_in);
}