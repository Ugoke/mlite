#pragma once

#include <cstdint>
#include <utility>

#include "mlite/foundation/errors.hpp"
#include "mlite/tensor/dtype.hpp"

namespace mlite::ops::detail {
    template <typename T>
    struct TypeTag {
        using type = T;
    };

    template <typename Function>
    decltype(auto) dispatch_dtype(DType dtype, Function&& function) {
        switch (dtype) {
            case DType::Bool:
                return std::forward<Function>(function)(TypeTag<bool>{});
            case DType::Int32:
                return std::forward<Function>(function)(TypeTag<std::int32_t>{});
            case DType::Int64:
                return std::forward<Function>(function)(TypeTag<std::int64_t>{});
            case DType::Float32:
                return std::forward<Function>(function)(TypeTag<float>{});
            case DType::Float64:
                return std::forward<Function>(function)(TypeTag<double>{});
        }
        throw DTypeError("Unsupported tensor dtype");
    }

    template <typename Function>
    decltype(auto) dispatch_numeric_dtype(DType dtype, Function&& function) {
        switch (dtype) {
            case DType::Bool:
                throw DTypeError("Operation requires a numeric tensor dtype");
            case DType::Int32:
                return std::forward<Function>(function)(TypeTag<std::int32_t>{});
            case DType::Int64:
                return std::forward<Function>(function)(TypeTag<std::int64_t>{});
            case DType::Float32:
                return std::forward<Function>(function)(TypeTag<float>{});
            case DType::Float64:
                return std::forward<Function>(function)(TypeTag<double>{});
        }
        throw DTypeError("Unsupported tensor dtype");
    }
}
