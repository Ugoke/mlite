#include "mlite/tensor/tensor.hpp"

#include <cstdint>
#include <utility>

#include "mlite/foundation/checked_math.hpp"
#include "mlite/foundation/errors.hpp"

namespace mlite {
    Tensor::Tensor(StoragePtr storage, TensorSpec spec)
        : Tensor(ConstStoragePtr(std::move(storage)), std::move(spec)) {}

    TensorSpec::TensorSpec(
        DType dtype,
        Shape shape,
        Strides strides,
        std::int64_t storage_offset
    )
        : dtype_(dtype),
          shape_(std::move(shape)),
          strides_(std::move(strides)),
          storage_offset_(storage_offset),
          numel_(compute_numel(shape_)) {
        if (shape_.size() != strides_.size()) {
            throw ShapeError("Tensor shape and strides must have equal rank");
        }
        if (storage_offset_ < 0) {
            throw ShapeError("Tensor storage offset cannot be negative");
        }
        static_cast<void>(dtype_size(dtype_));
    }

    TensorSpec TensorSpec::contiguous(DType dtype, const Shape& shape) {
        return TensorSpec(dtype, shape, make_contiguous_strides(shape));
    }

    DType TensorSpec::dtype() const noexcept { return dtype_; }
    const Shape& TensorSpec::shape() const noexcept { return shape_; }
    const Strides& TensorSpec::strides() const noexcept { return strides_; }
    std::int64_t TensorSpec::storage_offset() const noexcept { return storage_offset_; }
    std::size_t TensorSpec::rank() const noexcept { return shape_.size(); }
    std::int64_t TensorSpec::numel() const noexcept { return numel_; }
    std::size_t TensorSpec::nbytes() const {
        return detail::checked_bytes(numel_, dtype_size(dtype_));
    }
    bool TensorSpec::is_contiguous() const {
        return mlite::is_contiguous(shape_, strides_);
    }
    bool TensorSpec::is_fortran_contiguous() const {
        return mlite::is_fortran_contiguous(shape_, strides_);
    }

    Tensor::Tensor(ConstStoragePtr storage, TensorSpec spec)
        : storage_(std::move(storage)), spec_(std::move(spec)) {
        if (!storage_) {
            throw TensorError("Tensor requires storage");
        }
        validate_storage_bounds();
        validate_data_alignment();
    }

    const ConstStoragePtr& Tensor::storage() const noexcept { return storage_; }
    const TensorSpec& Tensor::spec() const noexcept { return spec_; }
    DType Tensor::dtype() const noexcept { return spec_.dtype(); }
    const Device& Tensor::device() const noexcept { return storage_->device(); }
    const Shape& Tensor::shape() const noexcept { return spec_.shape(); }
    const Strides& Tensor::strides() const noexcept { return spec_.strides(); }
    std::int64_t Tensor::storage_offset() const noexcept { return spec_.storage_offset(); }
    std::size_t Tensor::rank() const noexcept { return spec_.rank(); }
    std::int64_t Tensor::numel() const noexcept { return spec_.numel(); }
    std::size_t Tensor::nbytes() const { return spec_.nbytes(); }
    bool Tensor::writable() const noexcept { return storage_->writable(); }
    bool Tensor::is_contiguous() const { return spec_.is_contiguous(); }
    bool Tensor::is_fortran_contiguous() const { return spec_.is_fortran_contiguous(); }

    const void* Tensor::data() const {
        if (storage_->data() == nullptr) {
            return nullptr;
        }
        const auto offset = detail::checked_bytes(storage_offset(), dtype_size(dtype()));
        return static_cast<const std::uint8_t*>(storage_->data()) + offset;
    }

    void* Tensor::mutable_data() {
        if (!writable()) {
            throw ReadOnlyError("Tensor is read-only");
        }
        return const_cast<void*>(data());
    }

    void Tensor::validate_storage_bounds() const {
        const auto bounds = compute_storage_bounds(shape(), strides(), storage_offset());
        const auto item_size = dtype_size(dtype());
        if (bounds.empty) {
            if (detail::checked_bytes(storage_offset(), item_size) > storage_->bytes()) {
                throw ShapeError("Empty tensor offset is outside storage");
            }
            return;
        }
        if (bounds.minimum < 0) {
            throw ShapeError("Tensor strides access memory before storage");
        }
        const auto required = detail::checked_bytes(
            detail::checked_add(bounds.maximum, 1),
            item_size
        );
        if (required > storage_->bytes()) {
            throw ShapeError("Tensor metadata accesses memory outside storage");
        }
    }

    void Tensor::validate_data_alignment() const {
        if (storage_->data() == nullptr) {
            return;
        }
        const auto alignment = dtype_alignment(dtype());
        const auto byte_offset = detail::checked_bytes(
            storage_offset(),
            dtype_size(dtype())
        );
        const auto base_remainder =
            reinterpret_cast<std::uintptr_t>(storage_->data()) % alignment;
        if ((base_remainder + byte_offset % alignment) % alignment != 0) {
            throw TensorError(
                "Tensor data pointer is not aligned for dtype " + dtype_name(dtype())
            );
        }
    }
}
