#include "mlite/preprocessing/ordinal_encoder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>


namespace {
    bool find_category(
        const std::vector<double>& categories,
        double value,
        std::size_t& index
    ) {
        const auto position = std::lower_bound(
            categories.begin(),
            categories.end(),
            value
        );

        if (position == categories.end() || *position != value) {
            return false;
        }

        index = static_cast<std::size_t>(position - categories.begin());
        return true;
    }
}


namespace mlite {
    OrdinalEncoder::OrdinalEncoder()
        : categories_(),
          n_features_in_(0) {}

    void OrdinalEncoder::fit(const MatrixView& X) {
        if (X.rows() == 0 || X.cols() == 0) {
            throw std::invalid_argument("X must contain samples and features");
        }

        categories_.assign(X.cols(), {});

        for (std::size_t feature = 0; feature < X.cols(); ++feature) {
            auto& categories = categories_[feature];
            categories.reserve(X.rows());

            for (std::size_t sample = 0; sample < X.rows(); ++sample) {
                categories.push_back(X(sample, feature));
            }

            std::sort(categories.begin(), categories.end());
            categories.erase(
                std::unique(categories.begin(), categories.end()),
                categories.end()
            );
        }

        n_features_in_ = categories_.size();
    }

    std::vector<double> OrdinalEncoder::transform(
        const MatrixView& X,
        bool use_encoded_unknown,
        double unknown_value
    ) const {
        if (X.cols() != n_features_in_) {
            throw std::invalid_argument("X has different number of features");
        }

        std::vector<double> result(X.rows() * n_features_in_);

        for (std::size_t sample = 0; sample < X.rows(); ++sample) {
            for (std::size_t feature = 0; feature < n_features_in_; ++feature) {
                std::size_t category = 0;
                if (!find_category(categories_[feature], X(sample, feature), category)) {
                    if (use_encoded_unknown) {
                        result[sample * n_features_in_ + feature] = unknown_value;
                        continue;
                    }
                    throw std::invalid_argument("X contains an unknown category");
                }

                result[sample * n_features_in_ + feature] =
                    static_cast<double>(category);
            }
        }

        return result;
    }

    std::vector<double> OrdinalEncoder::inverse_transform(
        const MatrixView& X,
        bool use_encoded_unknown,
        double unknown_value
    ) const {
        if (X.cols() != n_features_in_) {
            throw std::invalid_argument("X has different number of features");
        }

        std::vector<double> result(
            X.rows() * n_features_in_,
            std::numeric_limits<double>::quiet_NaN()
        );

        for (std::size_t sample = 0; sample < X.rows(); ++sample) {
            for (std::size_t feature = 0; feature < n_features_in_; ++feature) {
                const double value = X(sample, feature);

                if (use_encoded_unknown && value == unknown_value) {
                    continue;
                }

                if (!std::isfinite(value) || value < 0.0 || std::floor(value) != value) {
                    throw std::invalid_argument("X contains an invalid ordinal code");
                }

                const std::size_t category = static_cast<std::size_t>(value);
                if (category >= categories_[feature].size()) {
                    throw std::invalid_argument("X contains an invalid ordinal code");
                }

                result[sample * n_features_in_ + feature] = categories_[feature][category];
            }
        }

        return result;
    }

    void OrdinalEncoder::load_state(
        const std::vector<std::vector<double>>& categories
    ) {
        categories_ = categories;

        for (auto& feature_categories : categories_) {
            if (feature_categories.empty()) {
                throw std::invalid_argument("Each feature must contain at least one category");
            }

            std::sort(feature_categories.begin(), feature_categories.end());
            feature_categories.erase(
                std::unique(feature_categories.begin(), feature_categories.end()),
                feature_categories.end()
            );
        }

        n_features_in_ = categories_.size();
    }

    std::vector<std::size_t> OrdinalEncoder::get_category_counts() const {
        std::vector<std::size_t> counts;
        counts.reserve(categories_.size());

        for (const auto& categories : categories_) {
            counts.push_back(categories.size());
        }

        return counts;
    }

    std::size_t OrdinalEncoder::get_n_features_in() const {
        return n_features_in_;
    }
}
