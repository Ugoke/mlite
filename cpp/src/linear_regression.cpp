#include "mlite/linear_regression.hpp"
#include <algorithm>

namespace mlite {
    LinearRegression::LinearRegression()
        : coef_(),
          intercept_(0.0),
          n_features_in_(0),
          fitted_(false) {}

    double LinearRegression::predict_sample(const MatrixView& X, std::size_t row) const {
        const double* x_row = X.data() + (row * n_features_in_);
        double prediction = intercept_;

        for (std::size_t j = 0; j < n_features_in_; ++j) {
            prediction += coef_[j] * x_row[j];
        }

        return prediction;
    }

    void LinearRegression::fit(const MatrixView& X, const VectorView& y, double learning_rate, std::size_t epochs) {
        const std::size_t samples = X.rows();

        if (samples == 0) {
            throw std::invalid_argument("Empty dataset");
        }

        const std::size_t features = X.cols();
        coef_.assign(features, 0.0);
        intercept_ = 0.0;
        n_features_in_ = features;

        std::vector<double> gradients(features, 0.0);
        const double inv_samples =2.0 / static_cast<double>(samples);

        for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
            std::fill(gradients.begin(), gradients.end(), 0.0);

            double bias_gradient = 0.0;

            for (std::size_t i = 0; i < samples; ++i) {
                const double* x_row = X.data() + (i * features);
                const double prediction = predict_sample(X, i);
                const double error = prediction - y(i);

                for (std::size_t j = 0; j < features; ++j) {
                    gradients[j] += error * x_row[j];
                }

                bias_gradient += error;
            }

            for (std::size_t j = 0; j < features; ++j) {
                coef_[j] -= learning_rate * gradients[j] * inv_samples;
            }

            intercept_ -= learning_rate * bias_gradient * inv_samples;
        }
        fitted_ = true;
    }

    std::vector<double> LinearRegression::predict(const MatrixView& X) const {
        const std::size_t samples = X.rows();

        std::vector<double> result(samples);

        for (std::size_t i = 0; i < samples; ++i) {
            result[i] = predict_sample(X, i);
        }

        return result;
    }

    double LinearRegression::score(const MatrixView& X, const VectorView& y) const {
        const std::size_t samples = y.size();

        double mean = 0.0;

        for (std::size_t i = 0; i < samples; ++i) {
            mean += y(i);
        }

        mean /= static_cast<double>(samples);

        double ss_total = 0.0;
        double ss_residual = 0.0;

        for (std::size_t i = 0; i < samples; ++i) {
            const double prediction = predict_sample(X, i);
            const double total_diff = y(i) - mean;
            const double residual_diff = y(i) - prediction;

            ss_total += total_diff * total_diff;
            ss_residual += residual_diff * residual_diff;
        }

        if (ss_total == 0.0) {
            return 0.0;
        }

        return 1.0 - (ss_residual / ss_total);
    }

    const std::vector<double>&
    LinearRegression::get_coef() const {
        return coef_;
    }

    double LinearRegression::get_intercept() const {
        return intercept_;
    }

    std::size_t
    LinearRegression::get_n_features_in() const {
        return n_features_in_;
    }
}