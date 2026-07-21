#include "mlite/model_selection/k_fold.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <random>


namespace mlite {
    std::vector<Fold> k_fold(const MatrixView& X, const VectorView& y, std::size_t n_splits, bool shuffle, std::uint32_t random_state) {
        const std::size_t n_samples = X.rows();
        const std::size_t n_features = X.cols();

        std::vector<std::size_t> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);

        if (shuffle) {
            std::mt19937 gen(random_state);
            std::shuffle(indices.begin(), indices.end(), gen);
        }

        const std::size_t fold_size = n_samples / n_splits;
        const std::size_t remainder = n_samples % n_splits;

        std::vector<Fold> folds;
        folds.reserve(n_splits);

        std::size_t start = 0;

        for (std::size_t fold = 0; fold < n_splits; ++fold) {
            const std::size_t current_test_size = fold_size + (fold < remainder ? 1 : 0);
            const std::size_t end = start + current_test_size;
            const std::size_t train_size = n_samples - current_test_size;

            Fold result;

            result.train_rows = train_size;
            result.test_rows = current_test_size;
            result.n_features = n_features;

            result.X_train.resize(train_size * n_features);
            result.X_test.resize(current_test_size * n_features);

            result.y_train.resize(train_size);
            result.y_test.resize(current_test_size);

            std::size_t train_row = 0;
            std::size_t test_row = 0;

            for (std::size_t i = 0; i < n_samples; ++i) {
                const auto idx = indices[i];

                if (i >= start && i < end) {
                    std::memcpy(result.X_test.data() + test_row * n_features, X.data() + idx * n_features, n_features * sizeof(double));

                    result.y_test[test_row++] = y(idx);
                } else {
                    std::memcpy(result.X_train.data() + train_row * n_features, X.data() + idx * n_features, n_features * sizeof(double));

                    result.y_train[train_row++] = y(idx);
                }
            }

            folds.push_back(std::move(result));
            start = end;
        }

        return folds;
    }
}