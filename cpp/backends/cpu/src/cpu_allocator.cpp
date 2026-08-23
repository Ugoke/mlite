#include "mlite/backends/cpu/cpu_allocator.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <new>
#include <utility>

#include "mlite/foundation/errors.hpp"

namespace {
    constexpr std::size_t cpu_alignment = 64;
}

namespace mlite {
    StoragePtr CpuAllocator::allocate(std::size_t bytes, const Device& device) {
        if (device.type() != DeviceType::CPU || device.index() != 0) {
            throw DeviceError("CpuAllocator only supports device cpu:0");
        }

        const auto allocation_size = std::max<std::size_t>(bytes, 1);
        auto* data = ::operator new(allocation_size, std::align_val_t{cpu_alignment});
        auto owner = std::shared_ptr<void>(data, [](void* pointer) {
            ::operator delete(pointer, std::align_val_t{cpu_alignment});
        });
        return std::make_shared<Storage>(data, bytes, device, true, std::move(owner));
    }
}
