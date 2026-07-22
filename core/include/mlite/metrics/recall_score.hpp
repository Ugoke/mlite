#pragma once

#include "mlite/core/vector_view.hpp"


namespace mlite {
    double recall_score(
        const VectorView& y_true,
        const VectorView& y_pred
    );
}
