#pragma once

#include <cstddef>
#include <stdexcept>

namespace mlite {
    class VectorView {
        private:
            const double* data_;
            std::size_t size_;

        public:
            explicit VectorView(
                const double* data,
                std::size_t size
            )
                : data_(data),
                  size_(size)
            {
                if (data == nullptr && size != 0) {
                    throw std::invalid_argument(
                        "VectorView: data is nullptr"
                    );
                }
            }

            double operator()(std::size_t i) const {
                if (i >= size_) {
                    throw std::out_of_range(
                        "VectorView: index out of range"
                    );
                }

                return data_[i];
            }

            std::size_t size() const noexcept {
                return size_;
            }

            const double* data() const noexcept {
                return data_;
            }
    }; 
}