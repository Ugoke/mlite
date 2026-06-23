#include "mlite/model_selection/train_test_split.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <random>


namespace mlite {
    TrainTestSplit train_test_split(const MatrixView& X, const VectorView& y, double test_size, std::uint32_t random_state) {
        const std::size_t n_samples = X.rows();
        const std::size_t n_features = X.cols();
        const std::size_t test_rows = static_cast<std::size_t>(n_samples * test_size);
        const std::size_t train_rows = n_samples - test_rows;

        std::vector<std::size_t> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::mt19937 gen(random_state);
        std::shuffle(indices.begin(), indices.end(), gen);

        TrainTestSplit result;

        result.train_rows = train_rows;
        result.test_rows = test_rows;
        result.n_features = n_features;

        result.X_train.resize(train_rows * n_features);
        result.X_test.resize(test_rows * n_features);
        result.y_train.resize(train_rows);
        result.y_test.resize(test_rows);

        for (std::size_t i = 0; i < train_rows; ++i) {
            const auto idx = indices[i];

            std::memcpy(
                result.X_train.data() + i * n_features,
                X.data() + idx * n_features,
                n_features * sizeof(double)
            );

            result.y_train[i] = y(idx);
        }

        for (std::size_t i = 0; i < test_rows; ++i) {
            const auto idx = indices[train_rows + i];

            std::memcpy(
                result.X_test.data() + i * n_features,
                X.data() + idx * n_features,
                n_features * sizeof(double)
            );

            result.y_test[i] = y(idx);
        }

        return result;
    }
}