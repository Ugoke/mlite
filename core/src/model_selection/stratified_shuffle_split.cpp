#include "mlite/model_selection/stratified_shuffle_split.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <unordered_map>


namespace mlite {
    StratifiedShuffleSplit stratified_shuffle_split(const MatrixView& X, const VectorView& y, double test_size, std::uint32_t random_state) {
        const std::size_t n_samples = X.rows();
        const std::size_t n_features = X.cols();

        std::unordered_map<double, std::vector<std::size_t>> class_indices;

        for (std::size_t i = 0; i < n_samples; ++i) {
            class_indices[y(i)].push_back(i);
        }

        std::mt19937 rng(random_state);

        std::vector<std::size_t> train_indices;
        std::vector<std::size_t> test_indices;

        train_indices.reserve(n_samples);
        test_indices.reserve(n_samples);

        for (auto& entry : class_indices) {
            auto& indices = entry.second;

            std::shuffle(indices.begin(), indices.end(), rng);

            const std::size_t class_size = indices.size();
            const std::size_t n_test = static_cast<std::size_t>(std::round(class_size * test_size));

            if (n_test == 0 || n_test >= class_size) {
                throw std::invalid_argument(
                    "Unable to perform stratified split: "
                    "each class must have at least one sample "
                    "in both the train and test sets."
                );
            }

            test_indices.insert(test_indices.end(), indices.begin(), indices.begin() + n_test);
            train_indices.insert(train_indices.end(), indices.begin() + n_test, indices.end());
        }

        std::shuffle(train_indices.begin(), train_indices.end(), rng);
        std::shuffle(test_indices.begin(), test_indices.end(), rng);

        StratifiedShuffleSplit result;

        result.train_rows = train_indices.size();
        result.test_rows = test_indices.size();
        result.n_features = n_features;

        result.X_train.reserve(result.train_rows * n_features);
        result.X_test.reserve(result.test_rows * n_features);
        result.y_train.reserve(result.train_rows);
        result.y_test.reserve(result.test_rows);

        for (std::size_t idx : train_indices) {
            for (std::size_t j = 0; j < n_features; ++j) {
                result.X_train.push_back(X(idx, j));
            }

            result.y_train.push_back(y(idx));
        }

        for (std::size_t idx : test_indices) {
            for (std::size_t j = 0; j < n_features; ++j) {
                result.X_test.push_back(X(idx, j));
            }

            result.y_test.push_back(y(idx));
        }

        return result;
    }
}