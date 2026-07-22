#pragma once

#include <cstddef>
#include <vector>

#include "mlite/core/matrix_view.hpp"


namespace mlite {
    class OrdinalEncoder {
        private:
            std::vector<std::vector<double>> categories_;
            std::size_t n_features_in_;

        public:
            OrdinalEncoder();

            void fit(const MatrixView& X);

            std::vector<double> transform(const MatrixView& X, bool use_encoded_unknown, double unknown_value) const;
            std::vector<double> inverse_transform(const MatrixView& X,bool use_encoded_unknown,double unknown_value) const;

            void load_state(const std::vector<std::vector<double>>& categories);

            std::vector<std::size_t> get_category_counts() const;
            std::size_t get_n_features_in() const;
    };
}
