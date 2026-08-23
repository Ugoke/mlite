#include "mlite/tensor/storage.hpp"

#include <string>
#include <utility>

#include "mlite/foundation/errors.hpp"

namespace mlite {
    Device::Device(DeviceType type, std::int32_t index) : type_(type), index_(index) {
        if (index_ < 0) {
            throw DeviceError("Device index cannot be negative");
        }
        if (type_ == DeviceType::CPU && index_ != 0) {
            throw DeviceError("CPU device index must be zero");
        }
    }

    Device Device::cpu() { return Device(DeviceType::CPU, 0); }
    DeviceType Device::type() const noexcept { return type_; }
    std::int32_t Device::index() const noexcept { return index_; }

    std::string Device::name() const {
        const std::string prefix = type_ == DeviceType::CPU ? "cpu" : "cuda";
        return index_ == 0 ? prefix : prefix + ":" + std::to_string(index_);
    }

    bool Device::operator==(const Device& other) const noexcept {
        return type_ == other.type_ && index_ == other.index_;
    }

    bool Device::operator!=(const Device& other) const noexcept {
        return !(*this == other);
    }

    Storage::Storage(
        void* data,
        std::size_t bytes,
        Device device,
        bool writable,
        std::shared_ptr<void> owner
    )
        : data_(data),
          bytes_(bytes),
          device_(device),
          writable_(writable),
          owner_(std::move(owner)) {
        if (data_ == nullptr && bytes_ != 0) {
            throw TensorError("Non-empty storage requires a valid data pointer");
        }
        if (bytes_ != 0 && !owner_) {
            throw TensorError("Non-empty storage requires an owner");
        }
    }

    void* Storage::data() {
        if (!writable_) {
            throw ReadOnlyError("Storage is read-only");
        }
        return data_;
    }
    const void* Storage::data() const noexcept { return data_; }
    std::size_t Storage::bytes() const noexcept { return bytes_; }
    const Device& Storage::device() const noexcept { return device_; }
    bool Storage::writable() const noexcept { return writable_; }
}
