#pragma once

#include <cstdint>
#include <vector>

namespace mlite {
    using Shape = std::vector<std::int64_t>;

    void validate_shape(const Shape& shape);
    std::int64_t compute_numel(const Shape& shape);
}
