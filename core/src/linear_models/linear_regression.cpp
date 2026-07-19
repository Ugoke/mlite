#include "mlite/linear_models/linear_regression.hpp"
#include "mlite/metrics/r2_score.hpp"

#include <algorithm>


namespace mlite {
    LinearRegression::LinearRegression()
        : coef_(),
          intercept_(0.0),
          n_features_in_(0) {}

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
        const std::size_t features = X.cols();
        coef_.assign(features, 0.0);
        intercept_ = 0.0;
        n_features_in_ = features;

        std::vector<double> gradients(features, 0.0);
        const double inv_samples = 2.0 / static_cast<double>(samples);

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
        std::vector<double> predictions = predict(X);
    
        VectorView y_pred(predictions.data(), predictions.size());
    
        return mlite::r2_score(y, y_pred);
    }

    const std::vector<double>& LinearRegression::get_coef() const {
        return coef_;
    }

    double LinearRegression::get_intercept() const {
        return intercept_;
    }

    std::size_t LinearRegression::get_n_features_in() const {
        return n_features_in_;
    }

    void LinearRegression::load_state(const std::vector<double>& coef, double intercept, std::size_t n_features_in) {
        coef_ = coef;
        intercept_ = intercept;
        n_features_in_ = n_features_in;
    }
}