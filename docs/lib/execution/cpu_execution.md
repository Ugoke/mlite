# CPU Execution

## 1. Description

The execution layer separates operations from memory allocation and scheduling. `ExecutionContext`
provides an `Allocator`, `Executor`, and `CancellationToken`; the CPU backend implements aligned
allocation and a persistent thread pool.

## 2. Mathematics / Logic

`parallel_for(begin, end, grain_size, task)` divides a range into at most the executor's configured
thread count. One chunk runs on the caller while remaining chunks are queued to worker threads.
Small ranges and nested worker calls run inline.

Worker exceptions are captured, synchronized, and rethrown on the caller. Cancellation is checked
before scheduling and before executing each range.

CPU allocation uses 64-byte alignment:

```text
address mod 64 = 0
```

## 3. Creation

Python uses one process-wide allocator and executor automatically:

```python
import mlite

mlite.set_num_threads(4)  # Must precede the first executor-backed operation.
print(mlite.get_num_threads())
x = mlite.ones((1_000_000,))
```

Python tensor operations use a process-wide executor. Its size can be selected once, before lazy
executor initialization. Executor construction and multiple independent contexts are exposed
through the C++ API.

## 4. Core Methods

| Component | Purpose |
|---|---|
| `Allocator` | Backend-independent storage allocation interface. |
| `Executor` | Backend-independent range execution interface. |
| `ExecutionContext` | Bundles allocation, execution, and cancellation dependencies. |
| `parallel_for` | Validates and delegates a logical range. |
| `CancellationSource/Token` | Shares cancellation state safely. |
| `CpuAllocator` | Produces aligned writable CPU Storage. |
| `ThreadPoolExecutor` | Reuses worker threads and propagates errors. |
| `RandomGenerator` | Produces synchronized deterministic seeds. |
| `set_num_threads` | Selects Python's global executor width before initialization. |
| `get_num_threads` | Returns the width without starting worker threads. |

## 5. Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| thread count | positive size | hardware concurrency | Total logical execution width including caller. |
| begin/end | signed 64-bit integers | — | Half-open range `[begin, end)`. |
| grain size | positive signed integer | operation-dependent | Minimum work per chunk. |
| cancellation | token | uncancelled token | Shared cancellation state. |

`set_num_threads()` accepts a positive integer. Calling it after an executor-backed operation
raises `ExecutionError`. Calling `get_num_threads()` is observational and does not prevent later
configuration. The Python `empty()` and `zeros()` factories use an allocation-only context and do
not initialize the process-wide worker pool.

## 6. Output

Parallel work returns when every scheduled range completes. Operations return their Tensor result
after the executor finishes. Invalid ranges or cancellation raise `ExecutionError`; worker
exceptions preserve their original C++ type.

## 7. Examples

### Minimal example

```python
import mlite

assert mlite.get_num_threads() >= 1
```

### Extended example

```python
import mlite

left = mlite.ones((2_000_000,))
right = mlite.full((2_000_000,), 3)
result = left + right

assert result.numpy()[0] == 4
```

## 8. Attributes

`ThreadPoolExecutor` stores worker threads, a synchronized task queue, stop state, and configured
thread count. Python exposes one-time configuration through `set_num_threads()` and observation
through `get_num_threads()`.

## 9. Notes

- The thread pool is reused instead of creating threads per tensor operation.
- Calls from worker threads execute nested ranges inline to prevent pool deadlock.
- CPU-bound binding calls release the Python GIL.
- Python owns one process-wide executor. Configure it before its first lazy initialization; it
  cannot be resized while operations may be using it.
- CPU is the only operational backend in the 0.2 series. Non-CPU operation requests are rejected.
