#pragma once

#include <pybind11/numpy.h>

namespace mlite {
    class LinearRegression {
        public:
            LinearRegression() = default;
        
            void fit(
                pybind11::array_t<double> X,
                pybind11::array_t<double> y,
                double lr,
                int epochs
            );
        
            pybind11::array_t<double> predict(
                pybind11::array_t<double> X
            );
        
        private:
            double weight = 0.0;
            double bias = 0.0;
    };
}