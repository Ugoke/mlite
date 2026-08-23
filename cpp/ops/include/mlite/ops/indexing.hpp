#pragma once

#include <cstdint>

#include "mlite/tensor/tensor.hpp"

namespace mlite::ops {
    Tensor reshape(const Tensor& source, const Shape& requested_shape);
    Tensor transpose(const Tensor& source, std::int64_t dimension0, std::int64_t dimension1);
}
