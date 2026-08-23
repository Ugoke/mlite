#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "mlite/tensor/tensor.hpp"

namespace mlite::ops {
    class TensorIterator {
        public:
            explicit TensorIterator(std::vector<const Tensor*> inputs);

            const Shape& shape() const noexcept;
            std::int64_t numel() const noexcept;
            std::size_t input_count() const noexcept;
            const Tensor& input(std::size_t index) const;
            const Strides& broadcast_strides(std::size_t input_index) const;
            std::int64_t offset(std::size_t input_index, std::int64_t flat_index) const;

        private:
            // Tensor is a cheap shared-storage value. Owning copies here prevents
            // iterators from retaining dangling pointers to temporary inputs.
            std::vector<Tensor> inputs_;
            std::vector<Strides> broadcast_strides_;
            Shape shape_;
            std::int64_t numel_;
    };
}
