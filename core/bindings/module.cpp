#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_linear_regression(py::module_& m);
void bind_logistic_regression(py::module_& m);

void bind_train_test_split(py::module_& m);
void bind_stratified_shuffle_split(py::module_& m);
void bind_k_fold(py::module_& m);

void bind_r2_score(py::module_& m);
void bind_accuracy_score(py::module_& m);
void bind_mean_squared_error(py::module_& m);
void bind_root_mean_squared_error(py::module_& m);
void bind_mean_absolute_error(py::module_& m);
void bind_recall_score(py::module_& m);

void bind_one_hot_encoder(py::module_& m);
void bind_ordinal_encoder(py::module_& m);

PYBIND11_MODULE(_core, m) {
    m.doc() = "mlite core";

    bind_linear_regression(m);
    bind_logistic_regression(m);
    
    bind_train_test_split(m);
    bind_stratified_shuffle_split(m);
    bind_k_fold(m);

    bind_r2_score(m);
    bind_accuracy_score(m);
    bind_mean_squared_error(m);
    bind_root_mean_squared_error(m);
    bind_mean_absolute_error(m);
    bind_recall_score(m);

    bind_one_hot_encoder(m);
    bind_ordinal_encoder(m);
}