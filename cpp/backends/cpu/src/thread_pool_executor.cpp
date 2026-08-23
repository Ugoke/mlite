#include "mlite/backends/cpu/thread_pool_executor.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>

#include "mlite/foundation/errors.hpp"

namespace {
    std::uint64_t range_size(std::int64_t begin, std::int64_t end) {
        if (end < begin) {
            throw mlite::ExecutionError("Invalid parallel range");
        }
        if (begin < 0 && end >= 0) {
            const auto distance_to_zero =
                static_cast<std::uint64_t>(-(begin + 1)) + 1;
            return distance_to_zero + static_cast<std::uint64_t>(end);
        }
        return static_cast<std::uint64_t>(end - begin);
    }

    std::int64_t advance_range(std::int64_t position, std::uint64_t distance) {
        constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
        if (position < 0) {
            const auto distance_to_zero =
                static_cast<std::uint64_t>(-(position + 1)) + 1;
            if (distance < distance_to_zero) {
                return position + static_cast<std::int64_t>(distance);
            }
            if (distance == distance_to_zero) {
                return 0;
            }
            distance -= distance_to_zero;
            if (distance > static_cast<std::uint64_t>(maximum)) {
                throw mlite::ExecutionError("Parallel range endpoint overflows int64");
            }
            return static_cast<std::int64_t>(distance);
        }

        const auto available = static_cast<std::uint64_t>(maximum - position);
        if (distance > available) {
            throw mlite::ExecutionError("Parallel range endpoint overflows int64");
        }
        return position + static_cast<std::int64_t>(distance);
    }
}

namespace mlite {
    thread_local ThreadPoolExecutor* ThreadPoolExecutor::current_executor_ = nullptr;

    ThreadPoolExecutor::ThreadPoolExecutor(std::size_t requested_threads) {
        const auto hardware_threads = static_cast<std::size_t>(std::thread::hardware_concurrency());
        thread_count_ = requested_threads == 0
            ? std::max<std::size_t>(hardware_threads, 1)
            : requested_threads;
        if (thread_count_ == 0) {
            throw ExecutionError("Executor thread count must be positive");
        }

        workers_.reserve(thread_count_ > 1 ? thread_count_ - 1 : 0);
        try {
            for (std::size_t index = 1; index < thread_count_; ++index) {
                workers_.emplace_back([this] { worker_loop(); });
            }
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stopping_ = true;
            }
            task_ready_.notify_all();
            for (auto& worker : workers_) {
                worker.join();
            }
            throw;
        }
    }

    ThreadPoolExecutor::~ThreadPoolExecutor() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        task_ready_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    std::size_t ThreadPoolExecutor::thread_count() const noexcept {
        return thread_count_;
    }

    void ThreadPoolExecutor::parallel_for(
        std::int64_t begin,
        std::int64_t end,
        std::int64_t grain_size,
        const RangeTask& task,
        const CancellationToken& cancellation
    ) {
        if (grain_size <= 0) {
            throw ExecutionError("Invalid parallel range");
        }
        const auto count = range_size(begin, end);
        cancellation.throw_if_cancelled();
        if (count == 0) {
            return;
        }
        const auto unsigned_grain_size = static_cast<std::uint64_t>(grain_size);
        if (
            thread_count_ == 1 ||
            count <= unsigned_grain_size ||
            current_executor_ == this
        ) {
            task(begin, end);
            return;
        }

        const auto desired_chunks =
            count / unsigned_grain_size +
            (count % unsigned_grain_size != 0 ? 1 : 0);
        const auto chunk_count =
            desired_chunks < static_cast<std::uint64_t>(thread_count_)
            ? static_cast<std::size_t>(desired_chunks)
            : thread_count_;
        const auto unsigned_chunk_count = static_cast<std::uint64_t>(chunk_count);
        const auto base_chunk_size = count / unsigned_chunk_count;
        const auto chunks_with_extra_item = count % unsigned_chunk_count;

        struct State {
            std::mutex mutex;
            std::condition_variable finished;
            std::size_t pending = 0;
            std::exception_ptr exception;
        };
        auto state = std::make_shared<State>();
        auto run_range = [state, task, cancellation](std::int64_t first, std::int64_t last) {
            try {
                cancellation.throw_if_cancelled();
                task(first, last);
            } catch (...) {
                std::lock_guard<std::mutex> lock(state->mutex);
                if (!state->exception) {
                    state->exception = std::current_exception();
                }
            }
        };

        auto next = begin;
        for (std::size_t chunk = 0; chunk + 1 < chunk_count; ++chunk) {
            const auto first = next;
            const auto chunk_size = base_chunk_size +
                (static_cast<std::uint64_t>(chunk) < chunks_with_extra_item ? 1 : 0);
            const auto last = advance_range(first, chunk_size);
            next = last;
            {
                std::lock_guard<std::mutex> lock(state->mutex);
                ++state->pending;
            }
            try {
                enqueue([state, run_range, first, last] {
                    run_range(first, last);
                    {
                        std::lock_guard<std::mutex> lock(state->mutex);
                        --state->pending;
                    }
                    state->finished.notify_one();
                });
            } catch (...) {
                std::unique_lock<std::mutex> lock(state->mutex);
                --state->pending;
                state->finished.wait(lock, [&state] { return state->pending == 0; });
                throw;
            }
        }

        run_range(next, end);

        std::unique_lock<std::mutex> lock(state->mutex);
        state->finished.wait(lock, [&state] { return state->pending == 0; });
        if (state->exception) {
            std::rethrow_exception(state->exception);
        }
    }

    void ThreadPoolExecutor::enqueue(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) {
                throw ExecutionError("Cannot enqueue work on a stopped executor");
            }
            tasks_.push(std::move(task));
        }
        task_ready_.notify_one();
    }

    void ThreadPoolExecutor::worker_loop() {
        current_executor_ = this;
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                task_ready_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
                if (stopping_ && tasks_.empty()) {
                    return;
                }
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
    }
}
