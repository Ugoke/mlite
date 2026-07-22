#include "mlite/metrics/mean_squared_error.hpp"


namespace mlite {
    double mean_squared_error(const VectorView& y_true, const VectorView& y_pred) {
        const std::size_t samples = y_true.size();
        double squared_error_sum = 0.0;

        for (std::size_t i = 0; i < samples; ++i) {
            const double error = y_true(i) - y_pred(i);
            squared_error_sum += error * error;
        }

        return squared_error_sum / static_cast<double>(samples);
    }
}
