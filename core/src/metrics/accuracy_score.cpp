#include "mlite/metrics/accuracy_score.hpp"


namespace mlite {
    double accuracy_score(const VectorView& y_true, const VectorView& y_pred) {
        const std::size_t samples = y_true.size();

        if (samples == 0) {
            return 0.0;
        }

        std::size_t correct = 0;

        for (std::size_t i = 0; i < samples; ++i) {
            if (static_cast<int>(y_true(i)) == static_cast<int>(y_pred(i))) {
                ++correct;
            }
        }

        return static_cast<double>(correct) / static_cast<double>(samples);
    }
}