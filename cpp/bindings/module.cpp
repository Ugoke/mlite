#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_linear(py::module_& m);

PYBIND11_MODULE(_core, m) {
    m.doc() = "mlite core";

    bind_linear(m);
}