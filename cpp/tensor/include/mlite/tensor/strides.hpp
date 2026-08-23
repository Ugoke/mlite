#pragma once

#include <cstdint>
#include <vector>

#include "mlite/tensor/shape.hpp"

namespace mlite {
    using Strides = std::vector<std::int64_t>;

    struct StorageBounds {
        bool empty;
        std::int64_t minimum;
        std::int64_t maximum;
    };

    Strides make_contiguous_strides(const Shape& shape);
    Strides make_fortran_strides(const Shape& shape);
    bool is_contiguous(const Shape& shape, const Strides& strides);
    bool is_fortran_contiguous(const Shape& shape, const Strides& strides);
    StorageBounds compute_storage_bounds(
        const Shape& shape,
        const Strides& strides,
        std::int64_t storage_offset
    );
}
