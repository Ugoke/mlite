#pragma once

#include <utility>

#include "mlite/execution/allocator.hpp"
#include "mlite/execution/cancellation.hpp"
#include "mlite/execution/executor.hpp"

namespace mlite {
    class ExecutionContext {
        public:
            ExecutionContext(
                Allocator& allocator,
                Executor& executor,
                CancellationToken cancellation = CancellationToken()
            )
                : allocator_(&allocator),
                  executor_(&executor),
                  cancellation_(std::move(cancellation)) {}

            Allocator& allocator() const noexcept { return *allocator_; }
            Executor& executor() const noexcept { return *executor_; }
            const CancellationToken& cancellation() const noexcept { return cancellation_; }

        private:
            Allocator* allocator_;
            Executor* executor_;
            CancellationToken cancellation_;
    };
}
