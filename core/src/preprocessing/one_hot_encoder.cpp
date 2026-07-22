#include "mlite/preprocessing/one_hot_encoder.hpp"

#include <algorithm>
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
    OneHotEncoder::OneHotEncoder()
        : categories_(),
          offsets_(),
          n_features_in_(0),
          n_features_out_(0) {}

    void OneHotEncoder::rebuild_offsets() {
        n_features_in_ = categories_.size();
        n_features_out_ = 0;
        offsets_.resize(n_features_in_);

        for (std::size_t feature = 0; feature < n_features_in_; ++feature) {
            if (categories_[feature].empty()) {
                throw std::invalid_argument("Each feature must contain at least one category");
            }

            offsets_[feature] = n_features_out_;
            n_features_out_ += categories_[feature].size();
        }
    }

    void OneHotEncoder::fit(const MatrixView& X) {
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
            categories.erase(std::unique(categories.begin(), categories.end()), categories.end());
        }

        rebuild_offsets();
    }

    std::vector<double> OneHotEncoder::transform(const MatrixView& X, bool ignore_unknown) const {
        if (X.cols() != n_features_in_) {
            throw std::invalid_argument("X has different number of features");
        }

        std::vector<double> result(X.rows() * n_features_out_, 0.0);

        for (std::size_t sample = 0; sample < X.rows(); ++sample) {
            for (std::size_t feature = 0; feature < n_features_in_; ++feature) {
                std::size_t category = 0;
                if (!find_category(categories_[feature], X(sample, feature), category)) {
                    if (ignore_unknown) {
                        continue;
                    }
                    throw std::invalid_argument("X contains an unknown category");
                }

                result[sample * n_features_out_ + offsets_[feature] + category] = 1.0;
            }
        }

        return result;
    }

    std::vector<double> OneHotEncoder::inverse_transform(const MatrixView& X) const {
        if (X.cols() != n_features_out_) {
            throw std::invalid_argument("X has different number of encoded features");
        }

        std::vector<double> result(X.rows() * n_features_in_, std::numeric_limits<double>::quiet_NaN());

        for (std::size_t sample = 0; sample < X.rows(); ++sample) {
            for (std::size_t feature = 0; feature < n_features_in_; ++feature) {
                const std::size_t start = offsets_[feature];
                const std::size_t count = categories_[feature].size();
                double best_value = 0.0;
                std::size_t best_category = 0;
                bool found = false;

                for (std::size_t category = 0; category < count; ++category) {
                    const double value = X(sample, start + category);
                    if (!found || value > best_value) {
                        best_value = value;
                        best_category = category;
                        found = true;
                    }
                }

                if (found && best_value > 0.0) {
                    result[sample * n_features_in_ + feature] = categories_[feature][best_category];
                }
            }
        }

        return result;
    }

    void OneHotEncoder::load_state(
        const std::vector<std::vector<double>>& categories
    ) {
        categories_ = categories;

        for (auto& feature_categories : categories_) {
            std::sort(feature_categories.begin(), feature_categories.end());
            feature_categories.erase(
                std::unique(feature_categories.begin(), feature_categories.end()),
                feature_categories.end()
            );
        }

        rebuild_offsets();
    }

    std::vector<std::size_t> OneHotEncoder::get_category_counts() const {
        std::vector<std::size_t> counts;
        counts.reserve(categories_.size());

        for (const auto& categories : categories_) {
            counts.push_back(categories.size());
        }

        return counts;
    }

    std::size_t OneHotEncoder::get_n_features_in() const {
        return n_features_in_;
    }

    std::size_t OneHotEncoder::get_n_features_out() const {
        return n_features_out_;
    }
}
