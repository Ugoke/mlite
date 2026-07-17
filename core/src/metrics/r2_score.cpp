#include "mlite/metrics/r2_score.hpp"


namespace mlite {
    double r2_score(const VectorView& y_true, const VectorView& y_pred) {
        const std::size_t samples = y_true.size();

        double mean = 0.0;
        for (std::size_t i = 0; i < samples; ++i) {
            mean += y_true(i);
        }
        mean /= static_cast<double>(samples);

        double ss_total = 0.0;
        double ss_residual = 0.0;

        for (std::size_t i = 0; i < samples; ++i) {
            const double total_diff = y_true(i) - mean;
            const double residual_diff = y_true(i) - y_pred(i);

            ss_total += total_diff * total_diff;
            ss_residual += residual_diff * residual_diff;
        }

        if (ss_total == 0.0) {
            return 0.0;
        }

        return 1.0 - (ss_residual / ss_total);
    }
}