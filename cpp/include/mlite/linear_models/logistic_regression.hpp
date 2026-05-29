#pragma once

#include <vector>
#include <cstddef>

#include "mlite/core/matrix_view.hpp"
#include "mlite/core/vector_view.hpp"


namespace mlite {
    class LogisticRegression {
        private:
            std::vector<double> coef_;
            double intercept_;
            std::size_t n_features_in_;

            double sigmoid(double z) const;

            double predict_probability_sample(
                const MatrixView& X,
                std::size_t row
            ) const;

        public:
            LogisticRegression();

            void fit(
                const MatrixView& X,
                const VectorView& y,
                double learning_rate,
                std::size_t epochs
            );

            std::vector<double> predict_proba(
                const MatrixView& X
            ) const;

            std::vector<int> predict(
                const MatrixView& X,
                double threshold
            ) const;

            double score(
                const MatrixView& X,
                const VectorView& y,
                double threshold
            ) const;

            void load_state(
                const std::vector<double>& coef,
                double intercept,
                std::size_t n_features_in
            );

            const std::vector<double>& get_coef() const;
            double get_intercept() const;
            std::size_t get_n_features_in() const;
    };
}