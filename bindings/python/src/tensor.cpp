#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mlite/backends/cpu/cpu_allocator.hpp"
#include "mlite/backends/cpu/thread_pool_executor.hpp"
#include "mlite/bindings/numpy.hpp"
#include "mlite/execution/execution_context.hpp"
#include "mlite/foundation/errors.hpp"
#include "mlite/ops/arithmetic.hpp"
#include "mlite/ops/creation.hpp"
#include "mlite/ops/indexing.hpp"
#include "mlite/tensor/dtype.hpp"
#include "mlite/tensor/tensor.hpp"

namespace py = pybind11;

namespace {
    class CpuRuntime {
        public:
            mlite::ThreadPoolExecutor& executor() {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!executor_) {
                    executor_ = std::make_unique<mlite::ThreadPoolExecutor>(requested_threads_);
                }
                return *executor_;
            }

            void set_thread_count(std::size_t thread_count) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (executor_) {
                    throw mlite::ExecutionError(
                        "set_num_threads() must be called before the first operation "
                        "that initializes the CPU executor"
                    );
                }
                requested_threads_ = thread_count;
            }

            std::size_t thread_count() {
                std::lock_guard<std::mutex> lock(mutex_);
                if (executor_) {
                    return executor_->thread_count();
                }
                if (requested_threads_ != 0) {
                    return requested_threads_;
                }
                return std::max<std::size_t>(
                    static_cast<std::size_t>(std::thread::hardware_concurrency()),
                    1
                );
            }

        private:
            std::mutex mutex_;
            std::size_t requested_threads_ = 0;
            std::unique_ptr<mlite::ThreadPoolExecutor> executor_;
    };

    CpuRuntime& cpu_runtime() {
        static CpuRuntime runtime;
        return runtime;
    }

    mlite::CpuAllocator& cpu_allocator() {
        static mlite::CpuAllocator allocator;
        return allocator;
    }

    mlite::ThreadPoolExecutor& cpu_executor() {
        return cpu_runtime().executor();
    }

    mlite::ExecutionContext execution_context() {
        return mlite::ExecutionContext(cpu_allocator(), cpu_executor());
    }

    mlite::ExecutionContext allocation_context() {
        // Creation operations such as empty() and zeros() only need an allocator.
        // A single-thread executor satisfies ExecutionContext without eagerly
        // starting the process-wide worker pool.
        static mlite::ThreadPoolExecutor inline_executor(1);
        return mlite::ExecutionContext(cpu_allocator(), inline_executor);
    }

    mlite::bindings::CopyPolicy copy_policy(const py::object& copy) {
        if (copy.is_none()) {
            return mlite::bindings::CopyPolicy::IfNeeded;
        }
        if (!py::isinstance<py::bool_>(copy)) {
            throw py::type_error("copy must be True, False, or None");
        }
        return copy.cast<bool>()
            ? mlite::bindings::CopyPolicy::Always
            : mlite::bindings::CopyPolicy::Never;
    }

    mlite::Tensor tensor_from_object(
        const py::object& data,
        const py::object& dtype,
        const py::object& copy
    ) {
        const auto policy = copy_policy(copy);
        if (py::isinstance<mlite::Tensor>(data)) {
            auto source = data.cast<mlite::Tensor>();
            if (!dtype.is_none() && dtype.cast<mlite::DType>() != source.dtype()) {
                if (policy == mlite::bindings::CopyPolicy::Never) {
                    throw mlite::DTypeError(
                        "copy=False does not allow Tensor dtype conversion"
                    );
                }
                auto converted = py::array::ensure(
                    py::module_::import("numpy").attr("asarray")(
                        mlite::bindings::tensor_to_numpy(source, false),
                        py::arg("dtype") = mlite::bindings::numpy_dtype(
                            dtype.cast<mlite::DType>()
                        ),
                        py::arg("order") = "C"
                    )
                );
                if (!converted) {
                    throw mlite::TensorError("Could not convert Tensor dtype");
                }
                // A differing dtype guarantees independent NumPy storage.
                // Adopt it directly instead of copying twice.
                return mlite::bindings::tensor_from_numpy(
                    std::move(converted),
                    mlite::bindings::CopyPolicy::Never
                );
            }
            if (policy == mlite::bindings::CopyPolicy::Always) {
                py::gil_scoped_release release;
                return mlite::ops::clone(source, execution_context());
            }
            return source;
        }

        py::array array;
        if (policy == mlite::bindings::CopyPolicy::Never) {
            if (!py::isinstance<py::array>(data)) {
                throw mlite::TensorError(
                    "copy=False requires data to be a NumPy ndarray; converting this object "
                    "would require a copy"
                );
            }
            array = py::reinterpret_borrow<py::array>(data);
            if (!dtype.is_none()) {
                const auto requested_dtype = dtype.cast<mlite::DType>();
                if (!array.dtype().equal(mlite::bindings::numpy_dtype(requested_dtype))) {
                    throw mlite::DTypeError("copy=False does not allow dtype conversion");
                }
            }
        } else if (policy == mlite::bindings::CopyPolicy::Always) {
            auto numpy = py::module_::import("numpy");
            if (dtype.is_none()) {
                array = py::array::ensure(numpy.attr("array")(
                    data,
                    py::arg("copy") = true,
                    py::arg("order") = "C"
                ));
            } else {
                array = py::array::ensure(numpy.attr("array")(
                    data,
                    py::arg("dtype") = mlite::bindings::numpy_dtype(
                        dtype.cast<mlite::DType>()
                    ),
                    py::arg("copy") = true,
                    py::arg("order") = "C"
                ));
            }
            if (!array) {
                throw py::type_error("data cannot be converted to a NumPy array");
            }
            // numpy.array(copy=True, order="C") already made the required
            // independent, compatible allocation. Do not copy it a second time.
            return mlite::bindings::tensor_from_numpy(
                std::move(array),
                mlite::bindings::CopyPolicy::Never
            );
        } else if (dtype.is_none()) {
            array = py::array::ensure(data);
        } else {
            array = py::array::ensure(
                py::module_::import("numpy").attr("asarray")(
                    data,
                    py::arg("dtype") = mlite::bindings::numpy_dtype(dtype.cast<mlite::DType>())
                )
            );
        }
        if (!array) {
            throw py::type_error("data cannot be converted to a NumPy array");
        }
        return mlite::bindings::tensor_from_numpy(std::move(array), policy);
    }

    template <typename Operation>
    mlite::Tensor run_binary(
        const mlite::Tensor& left,
        const mlite::Tensor& right,
        Operation operation
    ) {
        py::gil_scoped_release release;
        return operation(left, right, execution_context());
    }

    template <typename Value>
    mlite::Tensor run_full(
        const mlite::Shape& shape,
        Value value,
        mlite::DType dtype
    ) {
        py::gil_scoped_release release;
        return mlite::ops::full(shape, value, dtype, execution_context());
    }

    mlite::Tensor full_from_python(
        const mlite::Shape& shape,
        const py::object& value,
        mlite::DType dtype
    ) {
        if (py::isinstance<py::bool_>(value)) {
            return run_full(shape, value.cast<bool>(), dtype);
        }
        if (PyIndex_Check(value.ptr())) {
            auto integer_object = py::reinterpret_steal<py::object>(
                PyNumber_Index(value.ptr())
            );
            if (!integer_object) {
                throw py::error_already_set();
            }
            const auto integer = PyLong_AsLongLong(integer_object.ptr());
            if (integer == -1 && PyErr_Occurred()) {
                PyErr_Clear();
                throw mlite::DTypeError(
                    "Integer fill value is outside the int64 range"
                );
            }
            return run_full(shape, static_cast<std::int64_t>(integer), dtype);
        }
        if (PyFloat_Check(value.ptr())) {
            return run_full(shape, PyFloat_AsDouble(value.ptr()), dtype);
        }
        const auto numpy_boolean = py::module_::import("numpy").attr("bool_");
        if (py::isinstance(value, numpy_boolean)) {
            const auto truth = PyObject_IsTrue(value.ptr());
            if (truth < 0) {
                throw py::error_already_set();
            }
            return run_full(shape, truth != 0, dtype);
        }
        const auto real_number = py::module_::import("numbers").attr("Real");
        if (py::isinstance(value, real_number)) {
            const auto floating = PyFloat_AsDouble(value.ptr());
            if (floating == -1.0 && PyErr_Occurred()) {
                throw py::error_already_set();
            }
            return run_full(shape, floating, dtype);
        }
        throw py::type_error("fill_value must be a real scalar");
    }

    py::tuple tuple_from_values(const std::vector<std::int64_t>& values) {
        py::tuple result(values.size());
        for (std::size_t index = 0; index < values.size(); ++index) {
            result[index] = values[index];
        }
        return result;
    }

    std::string tensor_repr(const mlite::Tensor& tensor) {
        std::ostringstream stream;
        stream << "Tensor(shape=(";
        for (std::size_t dimension = 0; dimension < tensor.rank(); ++dimension) {
            if (dimension != 0) {
                stream << ", ";
            }
            stream << tensor.shape()[dimension];
        }
        if (tensor.rank() == 1) {
            stream << ',';
        }
        stream << "), dtype=" << mlite::dtype_name(tensor.dtype())
               << ", device='" << tensor.device().name() << "')";
        return stream.str();
    }
}

void bind_tensor(py::module_& module) {
    py::enum_<mlite::DType>(module, "DType")
        .value("bool_", mlite::DType::Bool)
        .value("int32", mlite::DType::Int32)
        .value("int64", mlite::DType::Int64)
        .value("float32", mlite::DType::Float32)
        .value("float64", mlite::DType::Float64)
        .export_values();

    py::class_<mlite::Tensor>(
        module,
        "Tensor",
        "A dense multidimensional array backed by CPU storage.",
        py::buffer_protocol()
    )
        .def(
            py::init([](const py::object& data, const py::object& copy) {
                return tensor_from_object(data, py::none(), copy);
            }),
            py::arg("data"),
            py::arg("copy") = py::none(),
            "Construct a tensor from a Tensor, NumPy ndarray, or array-like object."
        )
        .def_property_readonly("shape", [](const mlite::Tensor& tensor) {
            return tuple_from_values(tensor.shape());
        })
        .def_property_readonly("strides", [](const mlite::Tensor& tensor) {
            return tuple_from_values(tensor.strides());
        })
        .def_property_readonly("ndim", &mlite::Tensor::rank)
        .def_property_readonly("size", &mlite::Tensor::numel)
        .def_property_readonly("nbytes", &mlite::Tensor::nbytes)
        .def_property_readonly("dtype", &mlite::Tensor::dtype)
        .def_property_readonly("device", [](const mlite::Tensor& tensor) {
            return tensor.device().name();
        })
        .def_property_readonly("writable", &mlite::Tensor::writable)
        .def_property_readonly("is_contiguous", &mlite::Tensor::is_contiguous)
        .def_property_readonly("is_fortran_contiguous", &mlite::Tensor::is_fortran_contiguous)
        .def(
            "numpy",
            &mlite::bindings::tensor_to_numpy,
            py::arg("copy") = false,
            "Return a NumPy view, or an independent array when copy=True."
        )
        .def("clone", [](const mlite::Tensor& tensor) {
            py::gil_scoped_release release;
            return mlite::ops::clone(tensor, execution_context());
        }, "Return an independent C-contiguous copy of the tensor.")
        .def("contiguous", [](const mlite::Tensor& tensor) {
            py::gil_scoped_release release;
            return mlite::ops::contiguous(tensor, execution_context());
        }, "Return a C-contiguous tensor, sharing storage when already contiguous.")
        .def(
            "reshape",
            &mlite::ops::reshape,
            py::arg("shape"),
            "Return a zero-copy view with a new shape."
        )
        .def(
            "transpose",
            &mlite::ops::transpose,
            py::arg("dimension0") = 0,
            py::arg("dimension1") = 1,
            "Return a view with two dimensions exchanged."
        )
        .def_property_readonly("T", [](const mlite::Tensor& tensor) {
            if (tensor.rank() != 2) {
                throw mlite::ShapeError("T is only defined for rank-2 tensors");
            }
            return mlite::ops::transpose(tensor, 0, 1);
        })
        .def("__add__", [](const mlite::Tensor& left, const mlite::Tensor& right) {
            return run_binary(left, right, mlite::ops::add);
        }, py::is_operator())
        .def("__sub__", [](const mlite::Tensor& left, const mlite::Tensor& right) {
            return run_binary(left, right, mlite::ops::subtract);
        }, py::is_operator())
        .def("__mul__", [](const mlite::Tensor& left, const mlite::Tensor& right) {
            return run_binary(left, right, mlite::ops::multiply);
        }, py::is_operator())
        .def("__truediv__", [](const mlite::Tensor& left, const mlite::Tensor& right) {
            return run_binary(left, right, mlite::ops::divide);
        }, py::is_operator())
        .def("__array__", [](const mlite::Tensor& tensor, const py::object& dtype, const py::object& copy) {
            const auto policy = copy_policy(copy);
            py::dtype requested_dtype;
            auto requires_dtype_conversion = false;
            if (!dtype.is_none()) {
                requested_dtype = py::dtype::from_args(dtype);
                requires_dtype_conversion = !requested_dtype.equal(
                    mlite::bindings::numpy_dtype(tensor.dtype())
                );
                if (
                    policy == mlite::bindings::CopyPolicy::Never
                    && requires_dtype_conversion
                ) {
                    throw py::value_error(
                        "copy=False cannot satisfy the requested dtype without a copy"
                    );
                }
            }

            const auto must_copy =
                policy == mlite::bindings::CopyPolicy::Always
                && !requires_dtype_conversion;
            auto result = mlite::bindings::tensor_to_numpy(tensor, must_copy);
            if (requires_dtype_conversion) {
                result = py::array::ensure(
                    py::module_::import("numpy").attr("asarray")(
                        result,
                        py::arg("dtype") = requested_dtype
                    )
                );
            }
            return result;
        },
            py::arg("dtype") = py::none(),
            py::arg("copy") = py::none(),
            "Return a NumPy representation honoring the requested dtype and copy policy."
        )
        .def_buffer(&mlite::bindings::tensor_buffer)
        .def("__repr__", &tensor_repr);

    module.def(
        "tensor",
        &tensor_from_object,
        py::arg("data"),
        py::arg("dtype") = py::none(),
        py::arg("copy") = py::none(),
        "Create a tensor from a Tensor, NumPy ndarray, or array-like object."
    );
    module.def("empty", [](const mlite::Shape& shape, mlite::DType dtype) {
        py::gil_scoped_release release;
        return mlite::ops::empty(shape, dtype, allocation_context());
    },
        py::arg("shape"),
        py::arg("dtype") = mlite::DType::Float32,
        "Return an uninitialized C-contiguous CPU tensor."
    );
    module.def("zeros", [](const mlite::Shape& shape, mlite::DType dtype) {
        py::gil_scoped_release release;
        return mlite::ops::zeros(shape, dtype, allocation_context());
    },
        py::arg("shape"),
        py::arg("dtype") = mlite::DType::Float32,
        "Return a zero-filled C-contiguous CPU tensor."
    );
    module.def("ones", [](const mlite::Shape& shape, mlite::DType dtype) {
        py::gil_scoped_release release;
        return mlite::ops::ones(shape, dtype, execution_context());
    },
        py::arg("shape"),
        py::arg("dtype") = mlite::DType::Float32,
        "Return a one-filled C-contiguous CPU tensor."
    );
    module.def(
        "full",
        &full_from_python,
        py::arg("shape"),
        py::arg("fill_value"),
        py::arg("dtype") = mlite::DType::Float32,
        "Return a tensor filled with a scalar value."
    );
    module.def("add", [](const mlite::Tensor& left, const mlite::Tensor& right) {
        return run_binary(left, right, mlite::ops::add);
    },
        py::arg("left"),
        py::arg("right"),
        "Return the elementwise sum of two tensors."
    );
    module.def("subtract", [](const mlite::Tensor& left, const mlite::Tensor& right) {
        return run_binary(left, right, mlite::ops::subtract);
    },
        py::arg("left"),
        py::arg("right"),
        "Return the elementwise difference of two tensors."
    );
    module.def("multiply", [](const mlite::Tensor& left, const mlite::Tensor& right) {
        return run_binary(left, right, mlite::ops::multiply);
    },
        py::arg("left"),
        py::arg("right"),
        "Return the elementwise product of two tensors."
    );
    module.def("divide", [](const mlite::Tensor& left, const mlite::Tensor& right) {
        return run_binary(left, right, mlite::ops::divide);
    },
        py::arg("left"),
        py::arg("right"),
        "Return the elementwise quotient of two tensors."
    );
    module.def(
        "set_num_threads",
        [](std::int64_t thread_count) {
            if (thread_count <= 0) {
                throw py::value_error("thread_count must be a positive integer");
            }
            cpu_runtime().set_thread_count(static_cast<std::size_t>(thread_count));
        },
        py::arg("thread_count"),
        "Set the CPU worker count before the global executor is initialized."
    );
    module.def(
        "get_num_threads",
        [] { return cpu_runtime().thread_count(); },
        "Return the configured CPU execution width without starting worker threads."
    );
}
