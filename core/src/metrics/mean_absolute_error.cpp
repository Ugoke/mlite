#include <cmath>

#include "mlite/metrics/mean_absolute_error.hpp"


namespace mlite {
    double mean_absolute_error(const VectorView& y_true, const VectorView& y_pred) {
        const std::size_t samples = y_true.size();
        double absolute_error_sum = 0.0;

        for (std::size_t i = 0; i < samples; ++i) {
            absolute_error_sum += std::abs(y_true(i) - y_pred(i));
        }

        return absolute_error_sum / static_cast<double>(samples);
    }
}
