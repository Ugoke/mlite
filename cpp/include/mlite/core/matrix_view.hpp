#pragma once

#include <cstddef>
#include <stdexcept>

namespace mlite {
    class MatrixView {
        private:
            const double* data_;
        
            std::size_t rows_;
            std::size_t cols_;
        
        public:
            explicit MatrixView(
                const double* data,
                std::size_t rows,
                std::size_t cols
            )
                : data_(data),
                  rows_(rows),
                  cols_(cols)
            {
                if (data == nullptr && rows * cols != 0) {
                    throw std::invalid_argument(
                        "MatrixView: data is nullptr"
                    );
                }
            }
        
            double operator()(
                std::size_t row,
                std::size_t col
            ) const {
                if (row >= rows_ || col >= cols_) {
                    throw std::out_of_range(
                        "MatrixView: index out of range"
                    );
                }
            
                return data_[row * cols_ + col];
            }
        
            std::size_t rows() const noexcept {
                return rows_;
            }
        
            std::size_t cols() const noexcept {
                return cols_;
            }
        
            const double* data() const noexcept {
                return data_;
            }
    };
}