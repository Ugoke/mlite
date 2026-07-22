#pragma once

#include "mlite/core/vector_view.hpp"


namespace mlite {
    double root_mean_squared_error(
        const VectorView& y_true,
        const VectorView& y_pred
    );
}
