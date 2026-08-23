#pragma once

#include <cstdint>
#include <string>

namespace mlite {
    enum class DeviceType : std::uint8_t {
        CPU = 0,
        CUDA = 1
    };

    class Device {
        public:
            explicit Device(DeviceType type = DeviceType::CPU, std::int32_t index = 0);

            static Device cpu();
            DeviceType type() const noexcept;
            std::int32_t index() const noexcept;
            std::string name() const;

            bool operator==(const Device& other) const noexcept;
            bool operator!=(const Device& other) const noexcept;

        private:
            DeviceType type_;
            std::int32_t index_;
    };
}
