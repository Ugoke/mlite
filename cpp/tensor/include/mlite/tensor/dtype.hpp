#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace mlite {
    enum class DType : std::uint8_t {
        Bool = 0,
        Int32,
        Int64,
        Float32,
        Float64
    };

    std::size_t dtype_size(DType dtype);
    std::size_t dtype_alignment(DType dtype);
    std::string dtype_name(DType dtype);
    bool is_floating(DType dtype) noexcept;
    bool is_integral(DType dtype) noexcept;
}
