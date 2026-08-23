#include "mlite/tensor/dtype.hpp"

#include "mlite/foundation/errors.hpp"

namespace mlite {
    std::size_t dtype_size(DType dtype) {
        switch (dtype) {
            case DType::Bool: return sizeof(bool);
            case DType::Int32: return sizeof(std::int32_t);
            case DType::Int64: return sizeof(std::int64_t);
            case DType::Float32: return sizeof(float);
            case DType::Float64: return sizeof(double);
        }
        throw DTypeError("Unsupported tensor dtype");
    }

    std::size_t dtype_alignment(DType dtype) {
        switch (dtype) {
            case DType::Bool: return alignof(bool);
            case DType::Int32: return alignof(std::int32_t);
            case DType::Int64: return alignof(std::int64_t);
            case DType::Float32: return alignof(float);
            case DType::Float64: return alignof(double);
        }
        throw DTypeError("Unsupported tensor dtype");
    }

    std::string dtype_name(DType dtype) {
        switch (dtype) {
            case DType::Bool: return "bool";
            case DType::Int32: return "int32";
            case DType::Int64: return "int64";
            case DType::Float32: return "float32";
            case DType::Float64: return "float64";
        }
        throw DTypeError("Unsupported tensor dtype");
    }

    bool is_floating(DType dtype) noexcept {
        return dtype == DType::Float32 || dtype == DType::Float64;
    }

    bool is_integral(DType dtype) noexcept {
        return dtype == DType::Int32 || dtype == DType::Int64;
    }
}
