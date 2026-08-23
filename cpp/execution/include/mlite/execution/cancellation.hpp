#pragma once

#include <atomic>
#include <memory>
#include <utility>

#include "mlite/foundation/errors.hpp"

namespace mlite {
    class CancellationToken {
        public:
            CancellationToken() : cancelled_(std::make_shared<std::atomic_bool>(false)) {}

            bool is_cancelled() const noexcept {
                return cancelled_->load(std::memory_order_relaxed);
            }

            void throw_if_cancelled() const {
                if (is_cancelled()) {
                    throw ExecutionError("Operation was cancelled");
                }
            }

        private:
            explicit CancellationToken(std::shared_ptr<std::atomic_bool> cancelled)
                : cancelled_(std::move(cancelled)) {}

            std::shared_ptr<std::atomic_bool> cancelled_;

            friend class CancellationSource;
    };

    class CancellationSource {
        public:
            CancellationSource() : cancelled_(std::make_shared<std::atomic_bool>(false)) {}

            CancellationToken token() const { return CancellationToken(cancelled_); }
            void cancel() noexcept { cancelled_->store(true, std::memory_order_relaxed); }

        private:
            std::shared_ptr<std::atomic_bool> cancelled_;
    };
}
