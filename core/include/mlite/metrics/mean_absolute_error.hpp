#pragma once

#include "mlite/core/vector_view.hpp"


namespace mlite {
    double mean_absolute_error(
        const VectorView& y_true,
        const VectorView& y_pred
    );
}
