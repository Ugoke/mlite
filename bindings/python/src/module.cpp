#include <string>

#include <pybind11/pybind11.h>

#include "mlite/foundation/errors.hpp"
#include "mlite/foundation/version.hpp"

namespace py = pybind11;

void bind_tensor(py::module_& module);

PYBIND11_MODULE(_core, module) {
    module.doc() = mlite::description.data();
    module.attr("__package_name__") = std::string(mlite::package_name);
    module.attr("__version__") = std::string(mlite::version);
    module.attr("__description__") = std::string(mlite::description);

    auto error = py::register_exception<mlite::Error>(
        module,
        "Error",
        PyExc_RuntimeError
    );
    auto tensor_error = py::register_exception<mlite::TensorError>(
        module,
        "TensorError",
        error.ptr()
    );
    py::register_exception<mlite::ShapeError>(module, "ShapeError", tensor_error.ptr());
    py::register_exception<mlite::DTypeError>(module, "DTypeError", tensor_error.ptr());
    py::register_exception<mlite::DeviceError>(module, "DeviceError", tensor_error.ptr());
    py::register_exception<mlite::ReadOnlyError>(module, "ReadOnlyError", tensor_error.ptr());
    py::register_exception<mlite::ExecutionError>(module, "ExecutionError", error.ptr());
    bind_tensor(module);
}
