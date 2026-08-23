#include "mlite/ops/indexing.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>

#include "mlite/foundation/checked_math.hpp"
#include "mlite/foundation/errors.hpp"

namespace {
    mlite::Shape resolve_shape(const mlite::Shape& requested, std::int64_t numel) {
        auto resolved = requested;
        std::int64_t inferred_dimension = -1;
        auto has_zero_dimension = false;

        for (std::size_t dimension = 0; dimension < resolved.size(); ++dimension) {
            const auto size = resolved[dimension];
            if (size == -1) {
                if (inferred_dimension != -1) {
                    throw mlite::ShapeError("Only one reshape dimension can be inferred");
                }
                inferred_dimension = static_cast<std::int64_t>(dimension);
            } else if (size < 0) {
                throw mlite::ShapeError("Reshape dimensions cannot be negative except -1");
            } else if (size == 0) {
                has_zero_dimension = true;
            }
        }

        std::int64_t known_product = has_zero_dimension ? 0 : 1;
        if (!has_zero_dimension) {
            for (const auto size : resolved) {
                if (size != -1) {
                    known_product = mlite::detail::checked_multiply(
                        known_product,
                        size
                    );
                }
            }
        }

        if (inferred_dimension != -1) {
            if (known_product == 0 || numel % known_product != 0) {
                throw mlite::ShapeError("Cannot infer reshape dimension");
            }
            resolved[static_cast<std::size_t>(inferred_dimension)] = numel / known_product;
        }
        if (mlite::compute_numel(resolved) != numel) {
            throw mlite::ShapeError("Reshape cannot change the element count");
        }
        return resolved;
    }

    std::size_t normalize_dimension(std::int64_t dimension, std::size_t rank) {
        if (dimension < 0) {
            dimension += static_cast<std::int64_t>(rank);
        }
        if (dimension < 0 || dimension >= static_cast<std::int64_t>(rank)) {
            throw mlite::ShapeError("Tensor dimension is out of range");
        }
        return static_cast<std::size_t>(dimension);
    }
}

namespace mlite::ops {
    Tensor reshape(const Tensor& source, const Shape& requested_shape) {
        if (!source.is_contiguous()) {
            throw ShapeError("Reshape requires a contiguous tensor");
        }
        const auto shape = resolve_shape(requested_shape, source.numel());
        return Tensor(
            source.storage(),
            TensorSpec(
                source.dtype(),
                shape,
                make_contiguous_strides(shape),
                source.storage_offset()
            )
        );
    }

    Tensor transpose(const Tensor& source, std::int64_t dimension0, std::int64_t dimension1) {
        const auto first = normalize_dimension(dimension0, source.rank());
        const auto second = normalize_dimension(dimension1, source.rank());
        auto shape = source.shape();
        auto strides = source.strides();
        std::swap(shape[first], shape[second]);
        std::swap(strides[first], strides[second]);
        return Tensor(
            source.storage(),
            TensorSpec(
                source.dtype(),
                std::move(shape),
                std::move(strides),
                source.storage_offset()
            )
        );
    }
}
