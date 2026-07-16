#include "mlite/linear_models/logistic_regression.hpp"

#include <algorithm>
#include <cmath>
#include <vector>


namespace mlite {
    LogisticRegression::LogisticRegression()
        : coef_(),
          intercept_(0.0),
          n_features_in_(0) {}

    static inline double sigmoid_stable(double z) noexcept {
        if (z >= 0.0) {
            const double e = std::exp(-z);
            return 1.0 / (1.0 + e);
        } else {
            const double e = std::exp(z);
            return e / (1.0 + e);
        }
    }

    double LogisticRegression::sigmoid(double z) const {
        return sigmoid_stable(z);
    }

    double LogisticRegression::predict_probability_sample(const MatrixView& X, std::size_t row) const {
        const std::size_t features = n_features_in_;
        const double* x_row = X.data() + row * features;

        double linear = intercept_;
        const double* c = coef_.data();

        for (std::size_t j = 0; j < features; ++j) {
            linear += c[j] * x_row[j];
        }

        return sigmoid_stable(linear);
    }

    void LogisticRegression::fit(const MatrixView& X, const VectorView& y, double learning_rate, std::size_t epochs) {
        const std::size_t samples = X.rows();
        const std::size_t features = X.cols();

        coef_.assign(features, 0.0);
        intercept_ = 0.0;
        n_features_in_ = features;

        std::vector<double> gradients(features, 0.0);
        const double inv_samples = 1.0 / static_cast<double>(samples);
        const double* x_data = X.data();
        const double* c = coef_.data();

        for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
            std::fill(gradients.begin(), gradients.end(), 0.0);
            double bias_gradient = 0.0;

            for (std::size_t i = 0; i < samples; ++i) {
                const double* x_row = x_data + i * features;

                double linear = intercept_;
                for (std::size_t j = 0; j < features; ++j) {
                    linear += c[j] * x_row[j];
                }

                const double prediction = sigmoid_stable(linear);
                const double error = prediction - y(i);

                double* g = gradients.data();
                for (std::size_t j = 0; j < features; ++j) {
                    g[j] += error * x_row[j];
                }

                bias_gradient += error;
            }

            for (std::size_t j = 0; j < features; ++j) {
                coef_[j] -= learning_rate * gradients[j] * inv_samples;
            }
            intercept_ -= learning_rate * bias_gradient * inv_samples;
        }
    }

    std::vector<double> LogisticRegression::predict_proba(const MatrixView& X) const {
        const std::size_t samples = X.rows();
        const std::size_t features = n_features_in_;
        const double* x_data = X.data();

        std::vector<double> probabilities(samples);

        for (std::size_t i = 0; i < samples; ++i) {
            const double* x_row = x_data + i * features;
            double linear = intercept_;

            for (std::size_t j = 0; j < features; ++j) {
                linear += coef_[j] * x_row[j];
            }

            probabilities[i] = sigmoid_stable(linear);
        }

        return probabilities;
    }

    std::vector<int> LogisticRegression::predict(const MatrixView& X, double threshold) const {
        const std::size_t samples = X.rows();
        const std::size_t features = n_features_in_;
        const double* x_data = X.data();

        std::vector<int> predictions(samples);

        for (std::size_t i = 0; i < samples; ++i) {
            const double* x_row = x_data + i * features;
            double linear = intercept_;

            for (std::size_t j = 0; j < features; ++j) {
                linear += coef_[j] * x_row[j];
            }

            predictions[i] = (sigmoid_stable(linear) >= threshold) ? 1 : 0;
        }

        return predictions;
    }

    double LogisticRegression::score(const MatrixView& X, const VectorView& y, double threshold) const {
        const std::size_t samples = X.rows();
        const std::size_t features = n_features_in_;
        const double* x_data = X.data();
        std::size_t correct = 0;

        for (std::size_t i = 0; i < samples; ++i) {
            const double* x_row = x_data + i * features;
            double linear = intercept_;

            for (std::size_t j = 0; j < features; ++j) {
                linear += coef_[j] * x_row[j];
            }

            const int predicted = (sigmoid_stable(linear) >= threshold) ? 1 : 0;
            if (predicted == static_cast<int>(y(i))) {
                ++correct;
            }
        }

        return static_cast<double>(correct) / static_cast<double>(samples);
    }

    const std::vector<double>& LogisticRegression::get_coef() const {
        return coef_;
    }

    double LogisticRegression::get_intercept() const {
        return intercept_;
    }

    std::size_t LogisticRegression::get_n_features_in() const {
        return n_features_in_;
    }

    void LogisticRegression::load_state(const std::vector<double>& coef, double intercept, std::size_t n_features_in) {
        coef_ = coef;
        intercept_ = intercept;
        n_features_in_ = n_features_in;
    }
}