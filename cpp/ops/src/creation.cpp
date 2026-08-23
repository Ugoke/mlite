#include "mlite/ops/creation.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "mlite/execution/parallel.hpp"
#include "mlite/foundation/checked_math.hpp"
#include "mlite/foundation/errors.hpp"
#include "mlite/ops/dispatch.hpp"
#include "mlite/ops/tensor_iterator.hpp"

namespace {
    using ScalarValue = std::variant<bool, std::int64_t, double>;
    using ConvertedScalar = std::variant<
        bool,
        std::int32_t,
        std::int64_t,
        float,
        double
    >;

    void require_cpu(const mlite::Device& device, const char* operation) {
        if (device.type() != mlite::DeviceType::CPU || device.index() != 0) {
            throw mlite::DeviceError(
                std::string(operation) + " only supports CPU tensors"
            );
        }
    }

    template <typename T, typename Source>
    T checked_integer_cast(Source value) {
        if constexpr (std::is_floating_point_v<Source>) {
            if (!std::isfinite(value) || std::trunc(value) != value) {
                throw mlite::DTypeError(
                    "Integer tensor fill value must be a finite integer"
                );
            }
            static_assert(
                std::numeric_limits<T>::radix == 2,
                "Floating-point integer conversion requires a binary integer dtype"
            );
            const auto magnitude_limit = std::ldexp(
                Source{1},
                std::numeric_limits<T>::digits
            );
            const auto outside_range = std::is_signed_v<T>
                ? value < -magnitude_limit || value >= magnitude_limit
                : value < Source{0} || value >= magnitude_limit;
            if (outside_range) {
                throw mlite::DTypeError("Tensor fill value is outside dtype range");
            }
        } else if constexpr (!std::is_same_v<Source, bool>) {
            if (value < static_cast<Source>(std::numeric_limits<T>::lowest()) ||
                value > static_cast<Source>(std::numeric_limits<T>::max())) {
                throw mlite::DTypeError("Tensor fill value is outside dtype range");
            }
        }
        return static_cast<T>(value);
    }

    template <typename T, typename Source>
    T convert_fill_value(Source value) {
        if constexpr (std::is_same_v<T, bool>) {
            return value != static_cast<Source>(0);
        } else if constexpr (std::is_integral_v<T>) {
            return checked_integer_cast<T>(value);
        } else {
            return static_cast<T>(value);
        }
    }

    ConvertedScalar convert_fill_value(mlite::DType dtype, const ScalarValue& value) {
        return mlite::ops::detail::dispatch_dtype(dtype, [&](auto type) -> ConvertedScalar {
            using T = typename decltype(type)::type;
            return std::visit([](auto source) -> ConvertedScalar {
                return ConvertedScalar(
                    std::in_place_type<T>,
                    convert_fill_value<T>(source)
                );
            }, value);
        });
    }

    mlite::Tensor full_impl(
        const mlite::Shape& shape,
        ScalarValue value,
        mlite::DType dtype,
        const mlite::ExecutionContext& context,
        const mlite::Device& device
    ) {
        require_cpu(device, "full");

        // Complete all potentially-throwing scalar validation before allocating.
        const auto converted = convert_fill_value(dtype, value);
        auto result = mlite::ops::empty(shape, dtype, context, device);
        std::visit([&](auto fill) {
            using T = decltype(fill);
            auto* data = static_cast<T*>(result.mutable_data());
            mlite::parallel_for(context, result.numel(), [data, fill](auto begin, auto end) {
                for (auto index = begin; index < end; ++index) {
                    data[index] = fill;
                }
            });
        }, converted);
        return result;
    }
}

namespace mlite::ops {
    Tensor empty(
        const Shape& shape,
        DType dtype,
        const ExecutionContext& context,
        const Device& device
    ) {
        auto spec = TensorSpec::contiguous(dtype, shape);
        auto storage = context.allocator().allocate(spec.nbytes(), device);
        if (!storage) {
            throw TensorError("Allocator returned null storage");
        }
        if (storage->device() != device) {
            throw DeviceError("Allocator returned storage on the wrong device");
        }
        if (!storage->writable()) {
            throw ReadOnlyError("Allocator returned read-only storage");
        }
        return Tensor(std::move(storage), std::move(spec));
    }

    Tensor zeros(
        const Shape& shape,
        DType dtype,
        const ExecutionContext& context,
        const Device& device
    ) {
        require_cpu(device, "zeros");
        auto result = empty(shape, dtype, context, device);
        if (result.nbytes() != 0) {
            std::memset(result.mutable_data(), 0, result.nbytes());
        }
        return result;
    }

    Tensor ones(
        const Shape& shape,
        DType dtype,
        const ExecutionContext& context,
        const Device& device
    ) {
        return full(shape, std::int64_t{1}, dtype, context, device);
    }

    Tensor full(
        const Shape& shape,
        bool value,
        DType dtype,
        const ExecutionContext& context,
        const Device& device
    ) {
        return full_impl(shape, ScalarValue{value}, dtype, context, device);
    }

    Tensor full(
        const Shape& shape,
        std::int64_t value,
        DType dtype,
        const ExecutionContext& context,
        const Device& device
    ) {
        return full_impl(shape, ScalarValue{value}, dtype, context, device);
    }

    Tensor full(
        const Shape& shape,
        double value,
        DType dtype,
        const ExecutionContext& context,
        const Device& device
    ) {
        return full_impl(shape, ScalarValue{value}, dtype, context, device);
    }

    Tensor clone(const Tensor& source, const ExecutionContext& context) {
        require_cpu(source.device(), "clone");
        auto result = empty(source.shape(), source.dtype(), context, source.device());
        if (source.numel() == 0) {
            return result;
        }
        if (source.is_contiguous()) {
            std::memcpy(result.mutable_data(), source.data(), source.nbytes());
            return result;
        }

        TensorIterator iterator({&source});
        const auto item_size = dtype_size(source.dtype());
        const auto* source_base = static_cast<const std::uint8_t*>(source.storage()->data());
        auto* destination = static_cast<std::uint8_t*>(result.mutable_data());
        mlite::parallel_for(context, source.numel(), [&](auto begin, auto end) {
            for (auto index = begin; index < end; ++index) {
                const auto source_offset = mlite::detail::checked_bytes(
                    iterator.offset(0, index),
                    item_size
                );
                const auto destination_offset = mlite::detail::checked_bytes(index, item_size);
                std::memcpy(
                    destination + destination_offset,
                    source_base + source_offset,
                    item_size
                );
            }
        });
        return result;
    }

    Tensor contiguous(const Tensor& source, const ExecutionContext& context) {
        return source.is_contiguous() ? source : clone(source, context);
    }
}
