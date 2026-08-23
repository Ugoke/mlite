#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

#include "mlite/execution/cancellation.hpp"

namespace mlite {
    using RangeTask = std::function<void(std::int64_t begin, std::int64_t end)>;

    class Executor {
        public:
            virtual ~Executor() = default;

            virtual std::size_t thread_count() const noexcept = 0;
            virtual void parallel_for(
                std::int64_t begin,
                std::int64_t end,
                std::int64_t grain_size,
                const RangeTask& task,
                const CancellationToken& cancellation = CancellationToken()
            ) = 0;
    };
}
