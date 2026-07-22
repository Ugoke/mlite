#pragma once

#include <cstddef>
#include <vector>

#include "mlite/core/matrix_view.hpp"


namespace mlite {
    class OneHotEncoder {
        private:
            std::vector<std::vector<double>> categories_;
            std::vector<std::size_t> offsets_;
            std::size_t n_features_in_;
            std::size_t n_features_out_;

            void rebuild_offsets();

        public:
            OneHotEncoder();

            void fit(const MatrixView& X);

            std::vector<double> transform(const MatrixView& X, bool ignore_unknown) const;
            std::vector<double> inverse_transform(const MatrixView& X) const;

            void load_state(const std::vector<std::vector<double>>& categories);

            std::vector<std::size_t> get_category_counts() const;
            std::size_t get_n_features_in() const;
            std::size_t get_n_features_out() const;
    };
}
