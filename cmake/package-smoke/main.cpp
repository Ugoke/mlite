#include <cstdint>

#include "mlite/backends/cpu/cpu_allocator.hpp"
#include "mlite/backends/cpu/thread_pool_executor.hpp"
#include "mlite/execution/execution_context.hpp"
#include "mlite/foundation/version.hpp"
#include "mlite/ops/creation.hpp"
#include "mlite/tensor/dtype.hpp"
#include "mlite/tensor/shape.hpp"

int main() {
    mlite::CpuAllocator allocator;
    mlite::ThreadPoolExecutor executor(1);
    const mlite::ExecutionContext context(allocator, executor);
    const auto tensor = mlite::ops::ones(
        mlite::Shape{2, 3},
        mlite::DType::Float32,
        context
    );

    return tensor.numel() == 6 && !mlite::version.empty() ? 0 : 1;
}
