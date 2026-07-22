#include "mlite/metrics/recall_score.hpp"


namespace mlite {
    double recall_score(const VectorView& y_true, const VectorView& y_pred) {
        std::size_t true_positives = 0;
        std::size_t false_negatives = 0;

        for (std::size_t i = 0; i < y_true.size(); ++i) {
            if (y_true(i) == 1.0) {
                if (y_pred(i) == 1.0) {
                    ++true_positives;
                } else {
                    ++false_negatives;
                }
            }
        }

        const std::size_t actual_positives = true_positives + false_negatives;
        if (actual_positives == 0) {
            return 0.0;
        }

        return static_cast<double>(true_positives) / static_cast<double>(actual_positives);
    }
}
