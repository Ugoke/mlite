#pragma once

#include <pybind11/numpy.h>

#include "mlite/tensor/tensor.hpp"

namespace mlite::bindings {
    enum class CopyPolicy {
        IfNeeded,
        Never,
        Always
    };

    Tensor tensor_from_numpy(pybind11::array array, CopyPolicy copy_policy);
    pybind11::array tensor_to_numpy(const Tensor& tensor, bool copy);
    pybind11::dtype numpy_dtype(DType dtype);
    pybind11::buffer_info tensor_buffer(const Tensor& tensor);
}
