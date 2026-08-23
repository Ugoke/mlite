#include "mlite/tensor/shape.hpp"

#include <algorithm>

#include "mlite/foundation/checked_math.hpp"
#include "mlite/foundation/errors.hpp"
#include "mlite/tensor/strides.hpp"

namespace mlite {
    void validate_shape(const Shape& shape) {
        for (const auto dimension : shape) {
            if (dimension < 0) {
                throw ShapeError("Tensor dimensions cannot be negative");
            }
        }
    }

    std::int64_t compute_numel(const Shape& shape) {
        validate_shape(shape);
        if (std::find(shape.begin(), shape.end(), 0) != shape.end()) {
            return 0;
        }
        std::int64_t result = 1;
        for (const auto dimension : shape) {
            result = detail::checked_multiply(result, dimension);
        }
        return result;
    }

    Strides make_contiguous_strides(const Shape& shape) {
        validate_shape(shape);
        Strides strides(shape.size(), 1);
        if (std::find(shape.begin(), shape.end(), 0) != shape.end()) {
            return strides;
        }
        std::int64_t running = 1;
        for (std::size_t index = shape.size(); index > 0; --index) {
            const auto dimension = index - 1;
            strides[dimension] = running;
            if (dimension != 0) {
                running = detail::checked_multiply(running, shape[dimension]);
            }
        }
        return strides;
    }

    Strides make_fortran_strides(const Shape& shape) {
        validate_shape(shape);
        Strides strides(shape.size(), 1);
        if (std::find(shape.begin(), shape.end(), 0) != shape.end()) {
            return strides;
        }
        std::int64_t running = 1;
        for (std::size_t dimension = 0; dimension < shape.size(); ++dimension) {
            strides[dimension] = running;
            if (dimension + 1 < shape.size()) {
                running = detail::checked_multiply(running, shape[dimension]);
            }
        }
        return strides;
    }

    bool is_contiguous(const Shape& shape, const Strides& strides) {
        if (shape.size() != strides.size()) {
            return false;
        }
        if (compute_numel(shape) == 0) {
            return true;
        }

        std::int64_t expected = 1;
        for (std::size_t index = shape.size(); index > 0; --index) {
            const auto dimension = index - 1;
            if (shape[dimension] == 1) {
                continue;
            }
            if (strides[dimension] != expected) {
                return false;
            }
            expected = detail::checked_multiply(expected, shape[dimension]);
        }
        return true;
    }

    bool is_fortran_contiguous(const Shape& shape, const Strides& strides) {
        if (shape.size() != strides.size()) {
            return false;
        }
        if (compute_numel(shape) == 0) {
            return true;
        }

        std::int64_t expected = 1;
        for (std::size_t dimension = 0; dimension < shape.size(); ++dimension) {
            if (shape[dimension] == 1) {
                continue;
            }
            if (strides[dimension] != expected) {
                return false;
            }
            expected = detail::checked_multiply(expected, shape[dimension]);
        }
        return true;
    }

    StorageBounds compute_storage_bounds(
        const Shape& shape,
        const Strides& strides,
        std::int64_t storage_offset
    ) {
        validate_shape(shape);
        if (shape.size() != strides.size()) {
            throw ShapeError("Tensor shape and strides must have equal rank");
        }
        if (storage_offset < 0) {
            throw ShapeError("Tensor storage offset cannot be negative");
        }
        if (compute_numel(shape) == 0) {
            return {true, storage_offset, storage_offset};
        }

        auto minimum = storage_offset;
        auto maximum = storage_offset;
        for (std::size_t dimension = 0; dimension < shape.size(); ++dimension) {
            const auto delta = detail::checked_multiply(
                shape[dimension] - 1,
                strides[dimension]
            );
            if (delta < 0) {
                minimum = detail::checked_add(minimum, delta);
            } else {
                maximum = detail::checked_add(maximum, delta);
            }
        }
        return {false, minimum, maximum};
    }
}
