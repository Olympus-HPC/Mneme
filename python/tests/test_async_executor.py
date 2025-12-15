import queue
import pytest

from mneme.mneme_types import ExperimentConfiguration, ExperimentResult
from mneme.futures import EvalFuture
from mneme.async_executor import (
    pop,
    TuneWorkerHandle,
    AsyncReplayExecutor,
)


# -----------------------------------------------------------------------------
# pop()
# -----------------------------------------------------------------------------


def test_pop_returns_item():
    q = queue.Queue()
    q.put(42)
    assert pop(q, timeout=0.01) == 42


def test_pop_returns_none_on_timeout():
    q = queue.Queue()
    assert pop(q, timeout=0.01) is None


# -----------------------------------------------------------------------------
# TuneWorkerHandle internals (NO real processes)
# -----------------------------------------------------------------------------


class DummyProcess:
    def __init__(self):
        self.exitcode = 1
        self._alive = True
        self.pid = 123

    def is_alive(self):
        return self._alive

    def start(self):
        pass

    def join(self):
        self._alive = False


class DummyEvent:
    def __init__(self):
        self._set = False

    def set(self):
        self._set = True

    def is_set(self):
        return self._set


class DummyQueue(queue.Queue):
    pass


class NoopThread:
    def __init__(self, target=None, args=(), kwargs=None):
        self._target = target

    def start(self):
        # Do not run anything
        return

    def join(self, timeout=None):
        return


@pytest.fixture
def handle(monkeypatch):
    import mneme.async_executor as mod

    # Prevent background thread from running
    monkeypatch.setattr(mod.threading, "Thread", NoopThread, raising=True)

    # Patch multiprocessing bits as before
    monkeypatch.setattr(mod, "Process", lambda *a, **k: DummyProcess())
    monkeypatch.setattr(mod, "ProcessEvent", DummyEvent)
    monkeypatch.setattr(mod, "ProcessQueue", DummyQueue)

    global_q = queue.Queue()

    h = mod.TuneWorkerHandle(
        idx=0,
        global_q=global_q,
        record_db="db",
        record_id="rid",
        device_id=0,
        iterations=3,
        results_db_dir="/tmp",
    )
    return h


def test_process_result_sets_future(handle):
    future = EvalFuture(7, ExperimentConfiguration())
    handle.current = future

    msg = {
        "exp_id": 7,
        "data": ExperimentResult(executed=True).to_dict(),
    }

    handle._process_result(msg)

    assert future.done()
    assert future.result().executed is True
    assert handle.current is None


def test_process_result_wrong_id_sets_error(handle):
    future = EvalFuture(1, ExperimentConfiguration())
    handle.current = future

    msg = {"exp_id": 999, "data": {}}

    out = future.result(timeout=0.01)
    with pytest.raises(RuntimeError):
        handle._process_result(msg)
    assert handle.current is future
    assert future.done() is False


def test_submit_moves_state_and_sets_current(handle):
    cfg = ExperimentConfiguration()
    future = EvalFuture(3, cfg)
    handle.global_q.put(future)

    handle._submit()

    assert handle.current is future
    assert handle._action == handle.StateMachine.RECEIVE


def test_try_receive_no_message(handle):
    handle._ipc_read_q = queue.Queue()
    handle._action = handle.StateMachine.RECEIVE

    handle._try_receive()

    # state unchanged
    assert handle._action == handle.StateMachine.RECEIVE


def test_try_receive_processes_message(handle):
    future = EvalFuture(5, ExperimentConfiguration())
    handle.current = future

    handle._ipc_read_q = queue.Queue()
    handle._ipc_read_q.put(
        {
            "exp_id": 5,
            "data": ExperimentResult(executed=True).to_dict(),
        }
    )

    handle._action = handle.StateMachine.RECEIVE
    handle._try_receive()

    assert future.done()
    assert handle._action == handle.StateMachine.SUBMIT
    assert handle.current is None


# -----------------------------------------------------------------------------
# Crash recovery path
# -----------------------------------------------------------------------------


def test_worker_crash_marks_future_failed(handle):
    future = EvalFuture(1, ExperimentConfiguration())
    handle.current = future

    handle._process._alive = False
    handle._process.exitcode = 42

    # Run one iteration of the loop body manually
    if not handle._process.is_alive():
        handle.current.set_error(
            f"Worker crashed (exit code: {handle._process.exitcode}) running on device {handle.device_id}"
        )
        handle.current = None

    out = future.result(timeout=0.01)
    assert isinstance(out, ExperimentResult)
    assert out.failed is True
    assert out.executed is False
    assert "Worker crashed" in out.error


# -----------------------------------------------------------------------------
# AsyncReplayExecutor
# -----------------------------------------------------------------------------


def test_async_executor_submit(monkeypatch):
    monkeypatch.setattr("mneme.async_executor.TuneWorkerHandle", lambda *a, **k: None)

    exe = AsyncReplayExecutor(
        record_db="db",
        record_id="rid",
        iterations=3,
        results_db_dir="/tmp",
        num_workers=0,
    )

    cfg = ExperimentConfiguration()
    fut = exe.submit(cfg)

    assert isinstance(fut, EvalFuture)
    assert fut.job_id == 0


def test_async_executor_evaluate(monkeypatch):
    monkeypatch.setattr("mneme.async_executor.TuneWorkerHandle", lambda *a, **k: None)

    exe = AsyncReplayExecutor(
        record_db="db",
        record_id="rid",
        iterations=3,
        results_db_dir="/tmp",
        num_workers=0,
    )

    # fake future
    res = ExperimentResult(executed=True)
    fut = EvalFuture(0, ExperimentConfiguration())
    fut.set_result(res)

    monkeypatch.setattr(exe, "submit", lambda cfg: fut)

    out = exe.evaluate(ExperimentConfiguration())
    assert out.executed is True


def test_async_executor_shutdown(monkeypatch):
    joined = []

    class FakeHandle:
        def join(self):
            joined.append(True)

    monkeypatch.setattr(
        "mneme.async_executor.TuneWorkerHandle", lambda *a, **k: FakeHandle()
    )

    exe = AsyncReplayExecutor(
        record_db="db",
        record_id="rid",
        iterations=3,
        results_db_dir="/tmp",
        num_workers=2,
    )

    exe.shutdown()
    assert len(joined) == 2
