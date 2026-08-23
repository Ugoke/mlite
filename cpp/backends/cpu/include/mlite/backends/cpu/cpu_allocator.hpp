#pragma once

#include "mlite/execution/allocator.hpp"

namespace mlite {
    class CpuAllocator final : public Allocator {
        public:
            StoragePtr allocate(std::size_t bytes, const Device& device) override;
    };
}
