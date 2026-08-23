#include "mlite/bindings/numpy.hpp"

#include <Python.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>

#include "mlite/foundation/checked_math.hpp"
#include "mlite/foundation/errors.hpp"
#include "mlite/tensor/storage.hpp"

namespace py = pybind11;

namespace {
    mlite::DType tensor_dtype(const py::dtype& dtype) {
        if (!dtype.attr("isnative").cast<bool>()) {
            throw mlite::DTypeError("Only native-endian NumPy dtypes are supported");
        }

        const auto kind = dtype.kind();
        const auto item_size = dtype.itemsize();
        if (kind == 'b' && item_size == static_cast<py::ssize_t>(sizeof(bool))) {
            return mlite::DType::Bool;
        }
        if (kind == 'i' && item_size == static_cast<py::ssize_t>(sizeof(std::int32_t))) {
            return mlite::DType::Int32;
        }
        if (kind == 'i' && item_size == static_cast<py::ssize_t>(sizeof(std::int64_t))) {
            return mlite::DType::Int64;
        }
        if (kind == 'f' && item_size == static_cast<py::ssize_t>(sizeof(float))) {
            return mlite::DType::Float32;
        }
        if (kind == 'f' && item_size == static_cast<py::ssize_t>(sizeof(double))) {
            return mlite::DType::Float64;
        }
        throw mlite::DTypeError("Unsupported NumPy dtype: " + py::str(dtype).cast<std::string>());
    }

    std::shared_ptr<void> numpy_owner(const py::array& array) {
        auto* object = new py::object(array);
        return std::shared_ptr<void>(object, [](void* pointer) {
            if (!Py_IsInitialized()) {
                return;
            }
            py::gil_scoped_acquire acquire;
            delete static_cast<py::object*>(pointer);
        });
    }

    std::string buffer_format(mlite::DType dtype) {
        switch (dtype) {
            case mlite::DType::Bool: return py::format_descriptor<bool>::format();
            case mlite::DType::Int32: return py::format_descriptor<std::int32_t>::format();
            case mlite::DType::Int64: return py::format_descriptor<std::int64_t>::format();
            case mlite::DType::Float32: return py::format_descriptor<float>::format();
            case mlite::DType::Float64: return py::format_descriptor<double>::format();
        }
        throw mlite::DTypeError("Unsupported tensor dtype");
    }
}

namespace mlite::bindings {
    Tensor tensor_from_numpy(py::array array, CopyPolicy copy_policy) {
        const auto dtype = tensor_dtype(array.dtype());
        const auto flags = array.flags();
        const auto contiguous =
            (flags & py::array::c_style) != 0 || (flags & py::array::f_style) != 0;
        const auto address = reinterpret_cast<std::uintptr_t>(array.data());
        const auto aligned = address % dtype_alignment(dtype) == 0;
        const auto zero_copy_compatible = contiguous && aligned;

        if (copy_policy == CopyPolicy::Never && !zero_copy_compatible) {
            throw TensorError("copy=False requires a contiguous, aligned NumPy array");
        }
        if (copy_policy == CopyPolicy::Always || !zero_copy_compatible) {
            array = py::array::ensure(array.attr("copy")(py::str("C")));
            if (!array) {
                throw TensorError("Could not copy NumPy array");
            }
        }

        Shape shape;
        Strides strides;
        shape.reserve(static_cast<std::size_t>(array.ndim()));
        strides.reserve(static_cast<std::size_t>(array.ndim()));
        const auto item_size = array.itemsize();

        for (py::ssize_t dimension = 0; dimension < array.ndim(); ++dimension) {
            if (array.strides(dimension) % item_size != 0) {
                throw TensorError("NumPy byte strides must be divisible by dtype size");
            }
            shape.push_back(static_cast<std::int64_t>(array.shape(dimension)));
            strides.push_back(static_cast<std::int64_t>(array.strides(dimension) / item_size));
        }

        auto storage = std::make_shared<Storage>(
            const_cast<void*>(array.data()),
            static_cast<std::size_t>(array.nbytes()),
            Device::cpu(),
            array.writeable(),
            numpy_owner(array)
        );
        return Tensor(
            std::move(storage),
            TensorSpec(dtype, std::move(shape), std::move(strides))
        );
    }

    py::dtype numpy_dtype(DType dtype) {
        switch (dtype) {
            case DType::Bool: return py::dtype::of<bool>();
            case DType::Int32: return py::dtype::of<std::int32_t>();
            case DType::Int64: return py::dtype::of<std::int64_t>();
            case DType::Float32: return py::dtype::of<float>();
            case DType::Float64: return py::dtype::of<double>();
        }
        throw DTypeError("Unsupported tensor dtype");
    }

    py::array tensor_to_numpy(const Tensor& tensor, bool copy) {
        if (tensor.device().type() != DeviceType::CPU) {
            throw DeviceError("NumPy interop only supports CPU tensors");
        }

        std::vector<py::ssize_t> shape;
        std::vector<py::ssize_t> strides;
        shape.reserve(tensor.rank());
        strides.reserve(tensor.rank());
        const auto item_size = static_cast<std::int64_t>(dtype_size(tensor.dtype()));

        for (std::size_t dimension = 0; dimension < tensor.rank(); ++dimension) {
            shape.push_back(static_cast<py::ssize_t>(tensor.shape()[dimension]));
            strides.push_back(static_cast<py::ssize_t>(detail::checked_multiply(
                tensor.strides()[dimension],
                item_size
            )));
        }

        auto* owner = new Tensor(tensor);
        py::capsule capsule(owner, [](void* pointer) {
            delete static_cast<Tensor*>(pointer);
        });
        py::array result(numpy_dtype(tensor.dtype()), shape, strides, tensor.data(), capsule);
        if (!tensor.writable()) {
            result.attr("setflags")(false);
        }
        if (copy) {
            return py::array::ensure(result.attr("copy")(py::str("K")));
        }
        return result;
    }

    py::buffer_info tensor_buffer(const Tensor& tensor) {
        if (tensor.device().type() != DeviceType::CPU) {
            throw DeviceError("The Python buffer protocol only supports CPU tensors");
        }

        std::vector<py::ssize_t> shape;
        std::vector<py::ssize_t> strides;
        shape.reserve(tensor.rank());
        strides.reserve(tensor.rank());
        const auto item_size = static_cast<py::ssize_t>(dtype_size(tensor.dtype()));

        for (std::size_t dimension = 0; dimension < tensor.rank(); ++dimension) {
            shape.push_back(static_cast<py::ssize_t>(tensor.shape()[dimension]));
            strides.push_back(static_cast<py::ssize_t>(detail::checked_multiply(
                tensor.strides()[dimension],
                item_size
            )));
        }
        return py::buffer_info(
            const_cast<void*>(tensor.data()),
            item_size,
            buffer_format(tensor.dtype()),
            static_cast<py::ssize_t>(tensor.rank()),
            std::move(shape),
            std::move(strides),
            !tensor.writable()
        );
    }
}
