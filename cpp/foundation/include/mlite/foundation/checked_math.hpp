#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include "mlite/foundation/errors.hpp"

namespace mlite::detail {
    inline std::int64_t checked_add(std::int64_t left, std::int64_t right) {
        constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
        constexpr auto minimum = std::numeric_limits<std::int64_t>::min();

        if ((right > 0 && left > maximum - right) ||
            (right < 0 && left < minimum - right)) {
            throw ShapeError("Integer overflow while computing tensor metadata");
        }
        return left + right;
    }

    inline std::int64_t checked_multiply(std::int64_t left, std::int64_t right) {
        constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
        constexpr auto minimum = std::numeric_limits<std::int64_t>::min();

        if (left == 0 || right == 0) {
            return 0;
        }
        if ((left == -1 && right == minimum) || (right == -1 && left == minimum)) {
            throw ShapeError("Integer overflow while computing tensor metadata");
        }
        if (left > 0) {
            if ((right > 0 && left > maximum / right) ||
                (right < 0 && right < minimum / left)) {
                throw ShapeError("Integer overflow while computing tensor metadata");
            }
        } else if ((right > 0 && left < minimum / right) ||
                   (right < 0 && left < maximum / right)) {
            throw ShapeError("Integer overflow while computing tensor metadata");
        }
        return left * right;
    }

    inline std::size_t checked_bytes(std::int64_t elements, std::size_t item_size) {
        if (elements < 0) {
            throw ShapeError("Tensor element count cannot be negative");
        }
        const auto count = static_cast<std::uint64_t>(elements);
        constexpr auto maximum = std::numeric_limits<std::size_t>::max();
        if (item_size != 0 && count > maximum / item_size) {
            throw ShapeError("Tensor byte size overflows size_t");
        }
        return static_cast<std::size_t>(count) * item_size;
    }
}
