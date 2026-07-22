#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mlite/preprocessing/one_hot_encoder.hpp"


namespace py = pybind11;

namespace {
    mlite::MatrixView matrix_view_from(const py::buffer_info& buffer) {
        return mlite::MatrixView(
            static_cast<const double*>(buffer.ptr),
            static_cast<std::size_t>(buffer.shape[0]),
            static_cast<std::size_t>(buffer.shape[1])
        );
    }

    py::array_t<double> matrix_from_vector(
        const std::vector<double>& values,
        std::size_t rows,
        std::size_t columns
    ) {
        py::array_t<double> result({
            static_cast<py::ssize_t>(rows),
            static_cast<py::ssize_t>(columns)
        });
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


void bind_one_hot_encoder(py::module_& m) {
    py::class_<mlite::OneHotEncoder>(m, "OneHotEncoder")
        .def(py::init<>())

        .def(
            "fit",
            [](mlite::OneHotEncoder& self,
               py::array_t<double,
               py::array::c_style | py::array::forcecast> X) {
                const auto buffer = X.request();
                self.fit(matrix_view_from(buffer));
            },
            py::arg("X")
        )

        .def(
            "transform",
            [](const mlite::OneHotEncoder& self,
               py::array_t<double,
               py::array::c_style | py::array::forcecast> X,
               bool ignore_unknown) {
                const auto buffer = X.request();
                const auto transformed = self.transform(
                    matrix_view_from(buffer),
                    ignore_unknown
                );

                return matrix_from_vector(
                    transformed,
                    static_cast<std::size_t>(buffer.shape[0]),
                    self.get_n_features_out()
                );
            },
            py::arg("X"),
            py::arg("ignore_unknown")
        )

        .def(
            "inverse_transform",
            [](const mlite::OneHotEncoder& self,
               py::array_t<double,
               py::array::c_style | py::array::forcecast> X) {
                const auto buffer = X.request();
                const auto transformed = self.inverse_transform(
                    matrix_view_from(buffer)
                );

                return matrix_from_vector(
                    transformed,
                    static_cast<std::size_t>(buffer.shape[0]),
                    self.get_n_features_in()
                );
            },
            py::arg("X")
        )

        .def(
            "load_state",
            &mlite::OneHotEncoder::load_state,
            py::arg("categories")
        )

        .def_property_readonly(
            "category_counts_",
            &mlite::OneHotEncoder::get_category_counts
        )

        .def_property_readonly(
            "n_features_in_",
            &mlite::OneHotEncoder::get_n_features_in
        )

        .def_property_readonly(
            "n_features_out_",
            &mlite::OneHotEncoder::get_n_features_out
        );
}
