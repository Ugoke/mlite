#pragma once

#include "mlite/execution/execution_context.hpp"
#include "mlite/tensor/tensor.hpp"

namespace mlite::ops {
    Tensor add(const Tensor& left, const Tensor& right, const ExecutionContext& context);
    Tensor subtract(const Tensor& left, const Tensor& right, const ExecutionContext& context);
    Tensor multiply(const Tensor& left, const Tensor& right, const ExecutionContext& context);
    Tensor divide(const Tensor& left, const Tensor& right, const ExecutionContext& context);
}
