#include "mlite/ops/tensor_iterator.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>

#include "mlite/foundation/checked_math.hpp"
#include "mlite/foundation/errors.hpp"

namespace mlite::ops {
    TensorIterator::TensorIterator(std::vector<const Tensor*> inputs)
        : numel_(0) {
        if (inputs.empty()) {
            throw TensorError("TensorIterator requires at least one input");
        }
        inputs_.reserve(inputs.size());
        for (const auto* input : inputs) {
            if (input == nullptr) {
                throw TensorError("TensorIterator input cannot be null");
            }
            inputs_.push_back(*input);
        }

        std::size_t rank = 0;
        for (const auto& input : inputs_) {
            rank = std::max(rank, input.rank());
        }
        shape_.assign(rank, 1);

        for (const auto& input : inputs_) {
            const auto shift = rank - input.rank();
            for (std::size_t dimension = 0; dimension < input.rank(); ++dimension) {
                const auto output_dimension = shift + dimension;
                const auto input_size = input.shape()[dimension];
                auto& output_size = shape_[output_dimension];
                if (output_size != 1 && input_size != 1 && output_size != input_size) {
                    throw ShapeError("Tensor shapes are not broadcast-compatible");
                }
                if (output_size == 1) {
                    output_size = input_size;
                }
            }
        }
        numel_ = compute_numel(shape_);

        broadcast_strides_.reserve(inputs_.size());
        for (const auto& input : inputs_) {
            Strides strides(rank, 0);
            const auto shift = rank - input.rank();
            for (std::size_t dimension = 0; dimension < input.rank(); ++dimension) {
                const auto output_dimension = shift + dimension;
                if (input.shape()[dimension] != 1) {
                    strides[output_dimension] = input.strides()[dimension];
                }
            }
            broadcast_strides_.push_back(std::move(strides));
        }
    }

    const Shape& TensorIterator::shape() const noexcept { return shape_; }
    std::int64_t TensorIterator::numel() const noexcept { return numel_; }
    std::size_t TensorIterator::input_count() const noexcept { return inputs_.size(); }

    const Tensor& TensorIterator::input(std::size_t index) const {
        if (index >= inputs_.size()) {
            throw TensorError("TensorIterator input index is out of range");
        }
        return inputs_[index];
    }

    const Strides& TensorIterator::broadcast_strides(std::size_t input_index) const {
        if (input_index >= broadcast_strides_.size()) {
            throw TensorError("TensorIterator input index is out of range");
        }
        return broadcast_strides_[input_index];
    }

    std::int64_t TensorIterator::offset(
        std::size_t input_index,
        std::int64_t flat_index
    ) const {
        if (input_index >= inputs_.size() || flat_index < 0 || flat_index >= numel_) {
            throw TensorError("TensorIterator offset index is out of range");
        }

        auto remaining = flat_index;
        auto result = inputs_[input_index].storage_offset();
        for (std::size_t index = shape_.size(); index > 0; --index) {
            const auto dimension = index - 1;
            const auto coordinate = remaining % shape_[dimension];
            remaining /= shape_[dimension];
            result = mlite::detail::checked_add(
                result,
                mlite::detail::checked_multiply(
                    coordinate,
                    broadcast_strides_[input_index][dimension]
                )
            );
        }
        return result;
    }
}
