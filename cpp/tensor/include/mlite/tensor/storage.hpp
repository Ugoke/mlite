#pragma once

#include <cstddef>
#include <memory>

#include "mlite/tensor/device.hpp"

namespace mlite {
    class Storage {
        public:
            Storage(
                void* data,
                std::size_t bytes,
                Device device,
                bool writable,
                std::shared_ptr<void> owner
            );

            void* data();
            const void* data() const noexcept;
            std::size_t bytes() const noexcept;
            const Device& device() const noexcept;
            bool writable() const noexcept;

        private:
            void* data_;
            std::size_t bytes_;
            Device device_;
            bool writable_;
            std::shared_ptr<void> owner_;
    };

    using StoragePtr = std::shared_ptr<Storage>;
    using ConstStoragePtr = std::shared_ptr<const Storage>;
}
