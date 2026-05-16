#include "mlite/linear_regression.hpp"

#include <cmath>

namespace py = pybind11;

namespace mlite {
    void LinearRegression::fit(py::array_t<double> X, py::array_t<double> y, double lr, int epochs) {
        auto x_buf = X.unchecked<2>();
        auto y_buf = y.unchecked<1>();

        size_t n = X.shape(0);

        weight = 0.0;
        bias = 0.0;

        for (int epoch = 0; epoch < epochs; epoch++) {
            double dw = 0.0;
            double db = 0.0;

            for (size_t i = 0; i < n; i++) {
                double x = x_buf(i, 0);
                double pred = weight * x + bias;
                double error = pred - y_buf(i);

                dw += error * x;
                db += error;
            }

            dw /= static_cast<double>(n);
            db /= static_cast<double>(n);

            weight -= lr * dw;
            bias -= lr * db;
        }
    }

    py::array_t<double> LinearRegression::predict(py::array_t<double> X) {
        auto x_buf = X.unchecked<2>();
        size_t n = X.shape(0);
        py::array_t<double> result(n);
        auto r = result.mutable_unchecked<1>();

        for (size_t i = 0; i < n; i++) {
            double x = x_buf(i, 0);
            r(i) = weight * x + bias;
        }

        return result;
    }
}