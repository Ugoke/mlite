#pragma once

#include <stdexcept>

namespace mlite {
    class Error : public std::runtime_error {
        public:
            using std::runtime_error::runtime_error;
    };

    class TensorError : public Error {
        public:
            using Error::Error;
    };

    class ShapeError : public TensorError {
        public:
            using TensorError::TensorError;
    };

    class DTypeError : public TensorError {
        public:
            using TensorError::TensorError;
    };

    class DeviceError : public TensorError {
        public:
            using TensorError::TensorError;
    };

    class ReadOnlyError : public TensorError {
        public:
            using TensorError::TensorError;
    };

    class ExecutionError : public Error {
        public:
            using Error::Error;
    };

}
