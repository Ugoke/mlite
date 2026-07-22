#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mlite/preprocessing/ordinal_encoder.hpp"


namespace py = pybind11;

namespace {
    mlite::MatrixView ordinal_matrix_view_from(const py::buffer_info& buffer) {
        return mlite::MatrixView(
            static_cast<const double*>(buffer.ptr),
            static_cast<std::size_t>(buffer.shape[0]),
            static_cast<std::size_t>(buffer.shape[1])
        );
    }

    py::array_t<double> ordinal_matrix_from_vector(
        const std::vector<double>& values,
        std::size_t rows,
        std::size_t columns
    ) {
        py::array_t<double> result({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(columns)});
        auto output = result.mutable_unchecked<2>();

        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                output(
                    static_cast<py::ssize_t>(row),
                    static_cast<py::ssize_t>(column)
                ) = values[row * columns + column];
            }
        }

        return result;
    }
}


void bind_ordinal_encoder(py::module_& m) {
    py::class_<mlite::OrdinalEncoder>(m, "OrdinalEncoder")
        .def(py::init<>())

        .def(
            "fit",
            [](mlite::OrdinalEncoder& self,
               py::array_t<double,
               py::array::c_style | py::array::forcecast> X) {
                const auto buffer = X.request();
                self.fit(ordinal_matrix_view_from(buffer));
            },
            py::arg("X")
        )

        .def(
            "transform",
            [](const mlite::OrdinalEncoder& self,
               py::array_t<double,
               py::array::c_style | py::array::forcecast> X,
               bool use_encoded_unknown,
               double unknown_value) {
                const auto buffer = X.request();
                const auto transformed = self.transform(
                    ordinal_matrix_view_from(buffer),
                    use_encoded_unknown,
                    unknown_value
                );

                return ordinal_matrix_from_vector(
                    transformed,
                    static_cast<std::size_t>(buffer.shape[0]),
                    self.get_n_features_in()
                );
            },
            py::arg("X"),
            py::arg("use_encoded_unknown"),
            py::arg("unknown_value")
        )

        .def(
            "inverse_transform",
            [](const mlite::OrdinalEncoder& self,
               py::array_t<double,
               py::array::c_style | py::array::forcecast> X,
               bool use_encoded_unknown,
               double unknown_value) {
                const auto buffer = X.request();
                const auto transformed = self.inverse_transform(
                    ordinal_matrix_view_from(buffer),
                    use_encoded_unknown,
                    unknown_value
                );

                return ordinal_matrix_from_vector(
                    transformed,
                    static_cast<std::size_t>(buffer.shape[0]),
                    self.get_n_features_in()
                );
            },
            py::arg("X"),
            py::arg("use_encoded_unknown"),
            py::arg("unknown_value")
        )

        .def(
            "load_state",
            &mlite::OrdinalEncoder::load_state,
            py::arg("categories")
        )

        .def_property_readonly(
            "category_counts_",
            &mlite::OrdinalEncoder::get_category_counts
        )

        .def_property_readonly(
            "n_features_in_",
            &mlite::OrdinalEncoder::get_n_features_in
        );
}
