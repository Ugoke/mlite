#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "mlite/backends/cpu/cpu_allocator.hpp"
#include "mlite/backends/cpu/thread_pool_executor.hpp"
#include "mlite/execution/execution_context.hpp"
#include "mlite/execution/cancellation.hpp"
#include "mlite/execution/parallel.hpp"
#include "mlite/execution/random_generator.hpp"
#include "mlite/foundation/errors.hpp"
#include "mlite/ops/arithmetic.hpp"
#include "mlite/ops/creation.hpp"
#include "mlite/ops/indexing.hpp"
#include "mlite/ops/tensor_iterator.hpp"
#include "mlite/tensor/shape.hpp"
#include "mlite/tensor/storage.hpp"
#include "mlite/tensor/strides.hpp"
#include "mlite/tensor/tensor.hpp"

namespace {
    void require(bool condition, const char* message) {
        if (!condition) {
            throw std::runtime_error(message);
        }
    }

    template <typename Exception, typename Function>
    void require_throws(Function&& function, const char* message) {
        try {
            std::forward<Function>(function)();
        } catch (const Exception&) {
            return;
        }
        throw std::runtime_error(message);
    }

    class CountingAllocator final : public mlite::Allocator {
        public:
            mlite::StoragePtr allocate(
                std::size_t bytes,
                const mlite::Device& device
            ) override {
                ++calls;
                return allocator.allocate(bytes, device);
            }

            std::size_t calls = 0;

        private:
            mlite::CpuAllocator allocator;
    };

    mlite::StoragePtr make_aligned_storage(
        std::size_t bytes,
        const mlite::Device& device,
        bool writable = true
    ) {
        constexpr std::size_t alignment = 64;
        auto* data = ::operator new(
            std::max<std::size_t>(bytes, 1),
            std::align_val_t{alignment}
        );
        auto owner = std::shared_ptr<void>(data, [](void* pointer) {
            ::operator delete(pointer, std::align_val_t{alignment});
        });
        return std::make_shared<mlite::Storage>(
            data,
            bytes,
            device,
            writable,
            std::move(owner)
        );
    }

    class ReadOnlyAllocator final : public mlite::Allocator {
        public:
            mlite::StoragePtr allocate(
                std::size_t bytes,
                const mlite::Device& device
            ) override {
                return make_aligned_storage(bytes, device, false);
            }
    };

    void test_fill_conversion_precedes_allocation() {
        CountingAllocator allocator;
        mlite::ThreadPoolExecutor executor(1);
        const mlite::ExecutionContext context(allocator, executor);

        const auto maximum = std::numeric_limits<std::int64_t>::max();
        const auto minimum = std::numeric_limits<std::int64_t>::lowest();
        const auto maximum_tensor = mlite::ops::full(
            {1}, maximum, mlite::DType::Int64, context
        );
        const auto minimum_tensor = mlite::ops::full(
            {1}, minimum, mlite::DType::Int64, context
        );
        require(
            *static_cast<const std::int64_t*>(maximum_tensor.data()) == maximum,
            "int64 maximum fill was not preserved"
        );
        require(
            *static_cast<const std::int64_t*>(minimum_tensor.data()) == minimum,
            "int64 minimum fill was not preserved"
        );

        const auto calls_before_invalid_fill = allocator.calls;
        require_throws<mlite::DTypeError>([&] {
            static_cast<void>(mlite::ops::full(
                {1}, maximum, mlite::DType::Int32, context
            ));
        }, "out-of-range int32 fill did not throw");
        require(
            allocator.calls == calls_before_invalid_fill,
            "invalid integer fill allocated storage"
        );

        require_throws<mlite::DTypeError>([&] {
            static_cast<void>(mlite::ops::full(
                {1}, 9223372036854775808.0, mlite::DType::Int64, context
            ));
        }, "out-of-range double-to-int64 fill did not throw");
        require(
            allocator.calls == calls_before_invalid_fill,
            "invalid floating fill allocated storage"
        );

        const auto int64_limit = std::ldexp(1.0, 63);
        const auto largest_representable = std::nextafter(int64_limit, 0.0);
        const auto largest_tensor = mlite::ops::full(
            {1}, largest_representable, mlite::DType::Int64, context
        );
        require(
            *static_cast<const std::int64_t*>(largest_tensor.data()) ==
                static_cast<std::int64_t>(largest_representable),
            "largest representable in-range double was rejected or changed"
        );
        const auto minimum_tensor_from_double = mlite::ops::full(
            {1}, -int64_limit, mlite::DType::Int64, context
        );
        require(
            *static_cast<const std::int64_t*>(minimum_tensor_from_double.data()) == minimum,
            "exact int64 minimum double fill was not preserved"
        );
        const auto calls_before_lower_out_of_range = allocator.calls;
        require_throws<mlite::DTypeError>([&] {
            static_cast<void>(mlite::ops::full(
                {1}, std::nextafter(-int64_limit, -std::numeric_limits<double>::infinity()),
                mlite::DType::Int64, context
            ));
        }, "double below int64 minimum did not throw");
        require(
            allocator.calls == calls_before_lower_out_of_range,
            "out-of-range negative floating fill allocated storage"
        );

        const auto calls_before_invalid_unsigned = allocator.calls;
        require_throws<mlite::DTypeError>([&] {
            static_cast<void>(mlite::ops::full(
                {1}, std::numeric_limits<std::uint64_t>::max(),
                mlite::DType::Int64, context
            ));
        }, "out-of-range unsigned scalar did not throw");
        require(
            allocator.calls == calls_before_invalid_unsigned,
            "invalid unsigned scalar allocated storage"
        );
    }

    void verify_partition(
        mlite::ThreadPoolExecutor& executor,
        std::int64_t begin,
        std::int64_t end
    ) {
        std::mutex mutex;
        std::vector<std::pair<std::int64_t, std::int64_t>> ranges;
        executor.parallel_for(begin, end, 1, [&](auto first, auto last) {
            std::lock_guard<std::mutex> lock(mutex);
            ranges.emplace_back(first, last);
        });
        std::sort(ranges.begin(), ranges.end());
        require(!ranges.empty(), "parallel range produced no chunks");
        require(ranges.front().first == begin, "parallel range skipped its beginning");
        require(ranges.back().second == end, "parallel range skipped its end");
        for (std::size_t index = 0; index < ranges.size(); ++index) {
            require(
                ranges[index].first < ranges[index].second,
                "parallel range produced an empty or reversed chunk"
            );
            if (index != 0) {
                require(
                    ranges[index - 1].second == ranges[index].first,
                    "parallel chunks overlap or have a gap"
                );
            }
        }
    }

    void test_thread_pool_partitions_without_overflow() {
        mlite::ThreadPoolExecutor executor(6);
        verify_partition(executor, 0, 9);
        verify_partition(
            executor,
            std::numeric_limits<std::int64_t>::lowest(),
            std::numeric_limits<std::int64_t>::max()
        );

        std::mutex mutex;
        std::set<std::thread::id> thread_ids;
        executor.parallel_for(0, 100'000, 1, [&](auto, auto) {
            std::lock_guard<std::mutex> lock(mutex);
            thread_ids.insert(std::this_thread::get_id());
        });
        require(thread_ids.size() > 1, "parallel work used only one thread");

        require_throws<mlite::ExecutionError>([&] {
            executor.parallel_for(0, 4, 1, [](auto first, auto) {
                if (first == 0) {
                    throw mlite::ExecutionError("test exception");
                }
            });
        }, "worker exception was not propagated");
    }

    void test_cross_executor_nesting_uses_independent_pool() {
        mlite::ThreadPoolExecutor outer_executor(2);
        mlite::ThreadPoolExecutor inner_executor(2);
        std::mutex mutex;
        std::set<std::thread::id> inner_thread_ids;

        outer_executor.parallel_for(0, 2, 1, [&](auto first, auto) {
            if (first != 0) {
                return;
            }
            inner_executor.parallel_for(0, 2, 1, [&](auto, auto) {
                std::lock_guard<std::mutex> lock(mutex);
                inner_thread_ids.insert(std::this_thread::get_id());
            });
        });

        require(
            inner_thread_ids.size() > 1,
            "nested work on an independent executor was forced inline"
        );
    }

    void test_parallel_for_helper_validates_empty_ranges() {
        mlite::CpuAllocator allocator;
        mlite::ThreadPoolExecutor executor(1);
        const mlite::ExecutionContext context(allocator, executor);

        require_throws<mlite::ExecutionError>([&] {
            mlite::parallel_for(context, 0, [](auto, auto) {}, 0);
        }, "parallel_for helper accepted zero grain size for an empty range");
        require_throws<mlite::ExecutionError>([&] {
            mlite::parallel_for(context, 1, [](auto, auto) {}, -1);
        }, "parallel_for helper accepted negative grain size");

        mlite::CancellationSource cancellation;
        cancellation.cancel();
        const mlite::ExecutionContext cancelled_context(
            allocator,
            executor,
            cancellation.token()
        );
        auto task_was_called = false;
        require_throws<mlite::ExecutionError>([&] {
            mlite::parallel_for(cancelled_context, 0, [&](auto, auto) {
                task_was_called = true;
            });
        }, "parallel_for helper ignored cancellation for an empty range");
        require(!task_was_called, "empty parallel range invoked its task");
    }

    void test_execution_support_types() {
        mlite::CpuAllocator allocator;
        for (const auto bytes : {std::size_t{1}, std::size_t{1024}}) {
            const auto storage = allocator.allocate(bytes, mlite::Device::cpu());
            require(
                reinterpret_cast<std::uintptr_t>(storage->data()) % 64 == 0,
                "CPU allocation is not cache-line aligned"
            );
        }

        mlite::CancellationSource cancellation;
        const auto token = cancellation.token();
        cancellation.cancel();
        require_throws<mlite::ExecutionError>([&] {
            token.throw_if_cancelled();
        }, "cancelled token did not throw");

        mlite::RandomGenerator first(42);
        mlite::RandomGenerator second(42);
        mlite::RandomGenerator different(43);
        for (std::size_t index = 0; index < 4; ++index) {
            require(
                first.next_seed() == second.next_seed(),
                "equal random seeds produced different sequences"
            );
        }
        mlite::RandomGenerator baseline(42);
        require(
            baseline.next_seed() != different.next_seed(),
            "different random seeds produced the same first value"
        );
    }

    void test_tensor_alignment_is_a_core_invariant() {
        auto* allocation = new std::uint8_t[32];
        auto owner = std::shared_ptr<void>(allocation, [](void* pointer) {
            delete[] static_cast<std::uint8_t*>(pointer);
        });
        auto storage = std::make_shared<mlite::Storage>(
            allocation + 1,
            16,
            mlite::Device::cpu(),
            true,
            std::move(owner)
        );
        require_throws<mlite::TensorError>([&] {
            static_cast<void>(mlite::Tensor(
                storage,
                mlite::TensorSpec::contiguous(mlite::DType::Int64, {1})
            ));
        }, "misaligned tensor data was accepted");
    }

    void test_read_only_storage_and_allocator_contracts() {
        auto storage = make_aligned_storage(
            sizeof(std::int64_t),
            mlite::Device::cpu(),
            false
        );
        require_throws<mlite::ReadOnlyError>([&] {
            static_cast<void>(storage->data());
        }, "read-only Storage exposed mutable data");

        mlite::Tensor tensor(
            storage,
            mlite::TensorSpec::contiguous(mlite::DType::Int64, {1})
        );
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(tensor.storage().get())>>);
        require(
            tensor.storage()->data() != nullptr,
            "read-only Tensor did not preserve const data access"
        );
        require_throws<mlite::ReadOnlyError>([&] {
            static_cast<void>(tensor.mutable_data());
        }, "read-only Tensor exposed mutable data");

        ReadOnlyAllocator allocator;
        mlite::ThreadPoolExecutor executor(1);
        const mlite::ExecutionContext context(allocator, executor);
        require_throws<mlite::ReadOnlyError>([&] {
            static_cast<void>(mlite::ops::empty(
                {1}, mlite::DType::Int64, context
            ));
        }, "empty accepted read-only allocator storage");
    }

    void test_host_kernels_reject_non_cpu_storage() {
        const mlite::Device cuda(mlite::DeviceType::CUDA, 0);
        auto storage = make_aligned_storage(sizeof(std::int64_t), cuda);
        const mlite::Tensor tensor(
            storage,
            mlite::TensorSpec::contiguous(mlite::DType::Int64, {1})
        );
        CountingAllocator allocator;
        mlite::ThreadPoolExecutor executor(1);
        const mlite::ExecutionContext context(allocator, executor);

        require_throws<mlite::DeviceError>([&] {
            static_cast<void>(mlite::ops::add(tensor, tensor, context));
        }, "arithmetic accepted non-CPU storage");
        require(allocator.calls == 0, "non-CPU arithmetic allocated output storage");

        require_throws<mlite::DeviceError>([&] {
            static_cast<void>(mlite::ops::zeros(
                {1}, mlite::DType::Int64, context, cuda
            ));
        }, "zeros accepted a non-CPU device");
        require(allocator.calls == 0, "non-CPU zeros called the allocator");
    }

    void test_empty_shape_math_is_order_independent() {
        const auto huge = std::numeric_limits<std::int64_t>::max();
        require(
            mlite::compute_numel({huge, huge, 0}) == 0,
            "empty shape numel overflowed before observing zero"
        );
        require(
            mlite::make_contiguous_strides({huge, huge, 0}) == mlite::Strides({1, 1, 1}),
            "empty C strides are not canonical"
        );
        require(
            mlite::make_fortran_strides({0, huge, huge}) == mlite::Strides({1, 1, 1}),
            "empty Fortran strides are not canonical"
        );

        mlite::CpuAllocator allocator;
        mlite::ThreadPoolExecutor executor(1);
        const mlite::ExecutionContext context(allocator, executor);
        const auto empty = mlite::ops::empty(
            {0}, mlite::DType::Float32, context
        );
        const auto reshaped = mlite::ops::reshape(empty, {huge, huge, 0});
        require(
            reshaped.numel() == 0 && reshaped.shape() == mlite::Shape({huge, huge, 0}),
            "empty reshape overflowed before observing its zero dimension"
        );
    }

    void test_iterator_owns_inputs_and_fast_path_honors_offsets() {
        mlite::CpuAllocator allocator;
        std::optional<mlite::ops::TensorIterator> iterator;
        {
            auto storage = allocator.allocate(sizeof(std::int64_t), mlite::Device::cpu());
            *static_cast<std::int64_t*>(storage->data()) = 7;
            const mlite::Tensor temporary(
                std::move(storage),
                mlite::TensorSpec::contiguous(mlite::DType::Int64, {1})
            );
            iterator.emplace(std::vector<const mlite::Tensor*>{&temporary});
        }
        require(
            *static_cast<const std::int64_t*>(iterator->input(0).data()) == 7,
            "TensorIterator retained a dangling input pointer"
        );

        auto storage = allocator.allocate(4 * sizeof(std::int64_t), mlite::Device::cpu());
        auto* values = static_cast<std::int64_t*>(storage->data());
        values[0] = -1;
        values[1] = 2;
        values[2] = 3;
        values[3] = -1;
        const mlite::Tensor view(
            std::move(storage),
            mlite::TensorSpec(mlite::DType::Int64, {2}, {1}, 1)
        );
        mlite::ThreadPoolExecutor executor(1);
        const mlite::ExecutionContext context(allocator, executor);
        const auto result = mlite::ops::add(view, view, context);
        const auto* result_values = static_cast<const std::int64_t*>(result.data());
        require(
            result_values[0] == 4 && result_values[1] == 6,
            "contiguous arithmetic ignored storage_offset"
        );
    }
}

int main() {
    test_fill_conversion_precedes_allocation();
    test_thread_pool_partitions_without_overflow();
    test_cross_executor_nesting_uses_independent_pool();
    test_parallel_for_helper_validates_empty_ranges();
    test_execution_support_types();
    test_tensor_alignment_is_a_core_invariant();
    test_read_only_storage_and_allocator_contracts();
    test_host_kernels_reject_non_cpu_storage();
    test_empty_shape_math_is_order_independent();
    test_iterator_owns_inputs_and_fast_path_honors_offsets();
    return 0;
}
