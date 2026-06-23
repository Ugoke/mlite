#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "mlite/core/matrix_view.hpp"
#include "mlite/core/vector_view.hpp"


namespace mlite {
    struct TrainTestSplit {
        std::vector<double> X_train;
        std::vector<double> X_test;

        std::vector<double> y_train;
        std::vector<double> y_test;

        std::size_t train_rows;
        std::size_t test_rows;
        std::size_t n_features;
    };

    TrainTestSplit train_test_split(
        const MatrixView& X,
        const VectorView& y,
        double test_size,
        std::uint32_t random_state
    );
}