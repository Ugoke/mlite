#include "mlite/ops/arithmetic.hpp"

#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "mlite/execution/parallel.hpp"
#include "mlite/foundation/checked_math.hpp"
#include "mlite/foundation/errors.hpp"
#include "mlite/ops/creation.hpp"
#include "mlite/ops/dispatch.hpp"
#include "mlite/ops/tensor_iterator.hpp"

namespace {
    enum class BinaryOperation {
        Add,
        Subtract,
        Multiply,
        Divide
    };

    template <typename T>
    T checked_integral_binary(T left, T right, BinaryOperation operation) {
        constexpr auto minimum = std::numeric_limits<T>::lowest();
        constexpr auto maximum = std::numeric_limits<T>::max();

        switch (operation) {
            case BinaryOperation::Add:
                if ((right > 0 && left > maximum - right) ||
                    (right < 0 && left < minimum - right)) {
                    throw mlite::TensorError("Integer addition overflow");
                }
                return static_cast<T>(left + right);
            case BinaryOperation::Subtract:
                if ((right > 0 && left < minimum + right) ||
                    (right < 0 && left > maximum + right)) {
                    throw mlite::TensorError("Integer subtraction overflow");
                }
                return static_cast<T>(left - right);
            case BinaryOperation::Multiply:
                if (left == 0 || right == 0) {
                    return 0;
                }
                if ((left == -1 && right == minimum) ||
                    (right == -1 && left == minimum)) {
                    throw mlite::TensorError("Integer multiplication overflow");
                }
                if (left > 0) {
                    if ((right > 0 && left > maximum / right) ||
                        (right < 0 && right < minimum / left)) {
                        throw mlite::TensorError("Integer multiplication overflow");
                    }
                } else if ((right > 0 && left < minimum / right) ||
                           (right < 0 && left < maximum / right)) {
                    throw mlite::TensorError("Integer multiplication overflow");
                }
                return static_cast<T>(left * right);
            case BinaryOperation::Divide:
                if (right == 0) {
                    throw mlite::TensorError("Integer division by zero");
                }
                if (left == minimum && right == -1) {
                    throw mlite::TensorError("Integer division overflow");
                }
                return static_cast<T>(left / right);
        }
        throw mlite::TensorError("Unsupported binary operation");
    }

    template <typename T>
    T apply_binary(T left, T right, BinaryOperation operation) {
        if constexpr (std::is_integral<T>::value) {
            return checked_integral_binary(left, right, operation);
        }
        switch (operation) {
            case BinaryOperation::Add: return left + right;
            case BinaryOperation::Subtract: return left - right;
            case BinaryOperation::Multiply: return left * right;
            case BinaryOperation::Divide: return left / right;
        }
        throw mlite::TensorError("Unsupported binary operation");
    }

    mlite::Tensor binary_operation(
        const mlite::Tensor& left,
        const mlite::Tensor& right,
        BinaryOperation operation,
        const mlite::ExecutionContext& context
    ) {
        if (left.device() != right.device()) {
            throw mlite::DeviceError("Binary operands must be on the same device");
        }
        if (left.device().type() != mlite::DeviceType::CPU || left.device().index() != 0) {
            throw mlite::DeviceError("Arithmetic operations only support CPU tensors");
        }
        if (left.dtype() != right.dtype()) {
            throw mlite::DTypeError("Binary operands must have the same dtype");
        }
        if (left.dtype() == mlite::DType::Bool) {
            throw mlite::DTypeError("Arithmetic operations do not support bool tensors");
        }

        if (
            left.shape() == right.shape() &&
            left.is_contiguous() &&
            right.is_contiguous()
        ) {
            auto result = mlite::ops::empty(
                left.shape(),
                left.dtype(),
                context,
                left.device()
            );
            mlite::ops::detail::dispatch_numeric_dtype(left.dtype(), [&](auto type) {
                using T = typename decltype(type)::type;
                // Tensor::data(), unlike Storage::data(), includes storage_offset.
                const auto* left_data = static_cast<const T*>(left.data());
                const auto* right_data = static_cast<const T*>(right.data());
                auto* output_data = static_cast<T*>(result.mutable_data());
                mlite::parallel_for(context, result.numel(), [&](auto begin, auto end) {
                    for (auto index = begin; index < end; ++index) {
                        output_data[index] = apply_binary(
                            left_data[index],
                            right_data[index],
                            operation
                        );
                    }
                });
            });
            return result;
        }

        mlite::ops::TensorIterator iterator({&left, &right});
        auto result = mlite::ops::empty(
            iterator.shape(),
            left.dtype(),
            context,
            left.device()
        );
        if (iterator.numel() == 0) {
            return result;
        }

        mlite::ops::detail::dispatch_numeric_dtype(left.dtype(), [&](auto type) {
            using T = typename decltype(type)::type;
            const auto* left_data = static_cast<const T*>(left.storage()->data());
            const auto* right_data = static_cast<const T*>(right.storage()->data());
            auto* output_data = static_cast<T*>(result.mutable_data());
            const auto& shape = iterator.shape();
            const auto& left_strides = iterator.broadcast_strides(0);
            const auto& right_strides = iterator.broadcast_strides(1);

            mlite::Strides left_resets(shape.size(), 0);
            mlite::Strides right_resets(shape.size(), 0);
            for (std::size_t dimension = 0; dimension < shape.size(); ++dimension) {
                left_resets[dimension] = mlite::detail::checked_multiply(
                    mlite::detail::checked_multiply(
                        shape[dimension] - 1,
                        left_strides[dimension]
                    ),
                    -1
                );
                right_resets[dimension] = mlite::detail::checked_multiply(
                    mlite::detail::checked_multiply(
                        shape[dimension] - 1,
                        right_strides[dimension]
                    ),
                    -1
                );
            }

            mlite::parallel_for(context, iterator.numel(), [&](auto begin, auto end) {
                auto left_offset = iterator.offset(0, begin);
                auto right_offset = iterator.offset(1, begin);
                std::vector<std::int64_t> coordinates(shape.size(), 0);
                auto remaining = begin;
                for (std::size_t index = shape.size(); index > 0; --index) {
                    const auto dimension = index - 1;
                    coordinates[dimension] = remaining % shape[dimension];
                    remaining /= shape[dimension];
                }

                for (auto index = begin; index < end; ++index) {
                    output_data[index] = apply_binary(
                        left_data[left_offset],
                        right_data[right_offset],
                        operation
                    );
                    if (index + 1 == end) {
                        continue;
                    }

                    for (std::size_t cursor = shape.size(); cursor > 0; --cursor) {
                        const auto dimension = cursor - 1;
                        if (coordinates[dimension] + 1 < shape[dimension]) {
                            ++coordinates[dimension];
                            left_offset = mlite::detail::checked_add(
                                left_offset,
                                left_strides[dimension]
                            );
                            right_offset = mlite::detail::checked_add(
                                right_offset,
                                right_strides[dimension]
                            );
                            break;
                        }
                        coordinates[dimension] = 0;
                        left_offset = mlite::detail::checked_add(
                            left_offset,
                            left_resets[dimension]
                        );
                        right_offset = mlite::detail::checked_add(
                            right_offset,
                            right_resets[dimension]
                        );
                    }
                }
            });
        });
        return result;
    }
}

namespace mlite::ops {
    Tensor add(const Tensor& left, const Tensor& right, const ExecutionContext& context) {
        return binary_operation(left, right, BinaryOperation::Add, context);
    }

    Tensor subtract(const Tensor& left, const Tensor& right, const ExecutionContext& context) {
        return binary_operation(left, right, BinaryOperation::Subtract, context);
    }

    Tensor multiply(const Tensor& left, const Tensor& right, const ExecutionContext& context) {
        return binary_operation(left, right, BinaryOperation::Multiply, context);
    }

    Tensor divide(const Tensor& left, const Tensor& right, const ExecutionContext& context) {
        return binary_operation(left, right, BinaryOperation::Divide, context);
    }
}
