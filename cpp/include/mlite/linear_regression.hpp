#pragma once

#include <cstddef>
#include <vector>

#include "matrix_view.hpp"
#include "vector_view.hpp"

namespace mlite {
    class LinearRegression {
        private:
            std::vector<double> coef_;
            double intercept_;
            std::size_t n_features_in_;
        
            double predict_sample(const MatrixView& X, std::size_t row) const;
        
        public:
            LinearRegression();
        
            void fit(
                const MatrixView& X,
                const VectorView& y,
                double learning_rate,
                std::size_t epochs
            );
        
            std::vector<double> predict(const MatrixView& X) const;
        
            double score(const MatrixView& X, const VectorView& y) const;
        
            const std::vector<double>& get_coef() const;
            double get_intercept() const;
            std::size_t get_n_features_in() const;

            void load_state(
                const std::vector<double>& coef,
                double intercept,
                std::size_t n_features_in
            );
    };
}