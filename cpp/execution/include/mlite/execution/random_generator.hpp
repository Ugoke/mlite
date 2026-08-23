#pragma once

#include <cstdint>
#include <mutex>
#include <random>

namespace mlite {
    class RandomGenerator {
        public:
            explicit RandomGenerator(std::uint64_t seed = 0) : engine_(seed) {}

            std::uint64_t next_seed() {
                std::lock_guard<std::mutex> lock(mutex_);
                return engine_();
            }

        private:
            std::mutex mutex_;
            std::mt19937_64 engine_;
    };
}
