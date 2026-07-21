#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_linear_regression(py::module_& m);
void bind_logistic_regression(py::module_& m);

void bind_train_test_split(py::module_& m);
void bind_stratified_shuffle_split(py::module_& m);
void bind_k_fold(py::module_& m);

void bind_r2_score(py::module_& m);
void bind_accuracy_score(py::module_& m);

PYBIND11_MODULE(_core, m) {
    m.doc() = "mlite core";

    bind_linear_regression(m);
    bind_logistic_regression(m);
    
    bind_train_test_split(m);
    bind_stratified_shuffle_split(m);
    bind_k_fold(m);

    bind_r2_score(m);
    bind_accuracy_score(m);
}