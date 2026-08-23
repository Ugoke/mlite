#pragma once

#include <concepts>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "mlite/execution/execution_context.hpp"
#include "mlite/foundation/errors.hpp"
#include "mlite/tensor/tensor.hpp"

namespace mlite::ops {
    Tensor empty(
        const Shape& shape,
        DType dtype,
        const ExecutionContext& context,
        const Device& device = Device::cpu()
    );

    Tensor zeros(
        const Shape& shape,
        DType dtype,
        const ExecutionContext& context,
        const Device& device = Device::cpu()
    );

    Tensor ones(
        const Shape& shape,
        DType dtype,
        const ExecutionContext& context,
        const Device& device = Device::cpu()
    );

    Tensor full(
        const Shape& shape,
        bool value,
        DType dtype,
        const ExecutionContext& context,
        const Device& device = Device::cpu()
    );

    Tensor full(
        const Shape& shape,
        std::int64_t value,
        DType dtype,
        const ExecutionContext& context,
        const Device& device = Device::cpu()
    );

    Tensor full(
        const Shape& shape,
        double value,
        DType dtype,
        const ExecutionContext& context,
        const Device& device = Device::cpu()
    );

    template <std::integral Integer>
        requires (
            !std::same_as<std::remove_cv_t<Integer>, bool> &&
            !std::same_as<std::remove_cv_t<Integer>, std::int64_t>
        )
    Tensor full(
        const Shape& shape,
        Integer value,
        DType dtype,
        const ExecutionContext& context,
        const Device& device = Device::cpu()
    ) {
        constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
        if constexpr (std::is_unsigned_v<Integer>) {
            if (value > static_cast<std::make_unsigned_t<std::int64_t>>(maximum)) {
                throw DTypeError("Tensor fill value is outside int64 scalar range");
            }
        } else if constexpr (
            std::numeric_limits<Integer>::digits >
            std::numeric_limits<std::int64_t>::digits
        ) {
            if (value < static_cast<Integer>(std::numeric_limits<std::int64_t>::lowest()) ||
                value > static_cast<Integer>(maximum)) {
                throw DTypeError("Tensor fill value is outside int64 scalar range");
            }
        }
        return full(
            shape,
            static_cast<std::int64_t>(value),
            dtype,
            context,
            device
        );
    }

    Tensor clone(const Tensor& source, const ExecutionContext& context);
    Tensor contiguous(const Tensor& source, const ExecutionContext& context);
}
