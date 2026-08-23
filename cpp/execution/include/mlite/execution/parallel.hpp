#pragma once

#include <cstdint>

#include "mlite/execution/execution_context.hpp"
#include "mlite/foundation/errors.hpp"

namespace mlite {
    inline void parallel_for(
        const ExecutionContext& context,
        std::int64_t count,
        const RangeTask& task,
        std::int64_t grain_size = 16384
    ) {
        if (count < 0) {
            throw ExecutionError("Parallel range cannot be negative");
        }
        if (grain_size <= 0) {
            throw ExecutionError("Parallel grain size must be positive");
        }
        context.cancellation().throw_if_cancelled();
        if (count == 0) {
            return;
        }
        context.executor().parallel_for(
            0,
            count,
            grain_size,
            task,
            context.cancellation()
        );
    }
}
