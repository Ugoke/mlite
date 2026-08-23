#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "mlite/execution/executor.hpp"

namespace mlite {
    class ThreadPoolExecutor final : public Executor {
        public:
            explicit ThreadPoolExecutor(std::size_t thread_count = 0);
            ~ThreadPoolExecutor() override;

            ThreadPoolExecutor(const ThreadPoolExecutor&) = delete;
            ThreadPoolExecutor& operator=(const ThreadPoolExecutor&) = delete;

            std::size_t thread_count() const noexcept override;
            void parallel_for(
                std::int64_t begin,
                std::int64_t end,
                std::int64_t grain_size,
                const RangeTask& task,
                const CancellationToken& cancellation = CancellationToken()
            ) override;

        private:
            void enqueue(std::function<void()> task);
            void worker_loop();

            std::size_t thread_count_;
            std::vector<std::thread> workers_;
            std::queue<std::function<void()>> tasks_;
            std::mutex mutex_;
            std::condition_variable task_ready_;
            bool stopping_ = false;

            static thread_local ThreadPoolExecutor* current_executor_;
    };
}
