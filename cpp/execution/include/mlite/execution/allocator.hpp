#pragma once

#include <cstddef>

#include "mlite/tensor/device.hpp"
#include "mlite/tensor/storage.hpp"

namespace mlite {
    class Allocator {
        public:
            virtual ~Allocator() = default;
            virtual StoragePtr allocate(std::size_t bytes, const Device& device) = 0;
    };
}
