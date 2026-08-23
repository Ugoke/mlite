#pragma once

#include <cstddef>
#include <cstdint>

#include "mlite/tensor/storage.hpp"
#include "mlite/tensor/tensor_spec.hpp"

namespace mlite {
    class Tensor {
        public:
            Tensor(StoragePtr storage, TensorSpec spec);
            Tensor(ConstStoragePtr storage, TensorSpec spec);

            const ConstStoragePtr& storage() const noexcept;
            const TensorSpec& spec() const noexcept;
            DType dtype() const noexcept;
            const Device& device() const noexcept;
            const Shape& shape() const noexcept;
            const Strides& strides() const noexcept;
            std::int64_t storage_offset() const noexcept;
            std::size_t rank() const noexcept;
            std::int64_t numel() const noexcept;
            std::size_t nbytes() const;
            bool writable() const noexcept;
            bool is_contiguous() const;
            bool is_fortran_contiguous() const;

            const void* data() const;
            void* mutable_data();

        private:
            void validate_storage_bounds() const;
            void validate_data_alignment() const;

            ConstStoragePtr storage_;
            TensorSpec spec_;
    };
}
