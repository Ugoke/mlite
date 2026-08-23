import subprocess
import sys

import mlite


def test_executor_reports_valid_thread_count():
    assert mlite.get_num_threads() >= 1


def test_thread_count_can_be_configured_before_executor_initialization():
    script = """
import mlite

for invalid in (0, -1):
    try:
        mlite.set_num_threads(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid thread count was accepted")

assert mlite.get_num_threads() >= 1
mlite.set_num_threads(2)
assert mlite.get_num_threads() == 2
mlite.ones((1,))

try:
    mlite.set_num_threads(1)
except mlite.ExecutionError:
    pass
else:
    raise AssertionError("executor was reconfigured after an operation initialized it")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_allocation_only_factories_do_not_initialize_the_worker_pool():
    script = """
import mlite

mlite.empty((0,))
mlite.zeros((2, 2))
mlite.set_num_threads(2)
assert mlite.get_num_threads() == 2
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
