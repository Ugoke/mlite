#pragma once

#include <cstddef>
#include <cstdint>

#include "mlite/tensor/dtype.hpp"
#include "mlite/tensor/shape.hpp"
#include "mlite/tensor/strides.hpp"

namespace mlite {
    class TensorSpec {
        public:
            TensorSpec(
                DType dtype,
                Shape shape,
                Strides strides,
                std::int64_t storage_offset = 0
            );

            static TensorSpec contiguous(DType dtype, const Shape& shape);

            DType dtype() const noexcept;
            const Shape& shape() const noexcept;
            const Strides& strides() const noexcept;
            std::int64_t storage_offset() const noexcept;
            std::size_t rank() const noexcept;
            std::int64_t numel() const noexcept;
            std::size_t nbytes() const;
            bool is_contiguous() const;
            bool is_fortran_contiguous() const;

        private:
            DType dtype_;
            Shape shape_;
            Strides strides_;
            std::int64_t storage_offset_;
            std::int64_t numel_;
    };
}
