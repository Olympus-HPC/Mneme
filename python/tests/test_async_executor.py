import threading
import time
from multiprocessing import Event
from queue import Queue as ThreadQueue

from mneme.async_executor import AsyncReplayExecutor, TuneWorkerHandle
from mneme.futures import EvalFuture
from mneme.replay_executor import TuneWorker


def fake_worker_run(
    write_q,
    read_q,
    record_db,
    record_id,
    device_id,
    iterations,
    results_db_dir,
    state_event,
):
    """
    Simulates a minimal TuneWorker.run loop.
    """
    # Signal we are alive
    state_event.set()

    while True:
        msg = write_q.get()

        if msg["payload"] == "terminate":
            break

        if msg["payload"] == "process":
            exp_id = msg["exp_id"]

            # Simulate compute delay
            time.sleep(0.01)

            read_q.put(
                {
                    "payload": "result",
                    "exp_id": exp_id,
                    "data": {"exec_time": 123, "params": msg["data"]},
                    "llvm_ir": "fake_ir.bc",
                }
            )


def test_tune_worker_handle_basic(monkeypatch):
    # Patch TuneWorker.run
    monkeypatch.setattr(
        TuneWorker,
        "run",
        staticmethod(fake_worker_run),
    )

    global_q = ThreadQueue()

    # Instantiate one worker handle
    h = TuneWorkerHandle(
        idx=0,
        global_q=global_q,
        record_db="fake.json",
        record_id="fake",
        device_id=0,
        iterations=1,
        results_db_dir="/tmp",
    )

    # Create a job
    future = EvalFuture(job_id=1, params={"x": 5})

    # Enqueue the future
    global_q.put(future)

    # Wait for worker to complete result
    result = future.result(timeout=2)

    assert result["payload"] == "result"
    assert result["exp_id"] == 1
    assert result["data"]["params"] == {"x": 5}

    # Shutdown
    h._shutdown_event.set()
    h.join()


def test_async_executor_multiple_jobs(monkeypatch):
    # Patch TuneWorker.run
    monkeypatch.setattr(
        TuneWorker,
        "run",
        staticmethod(fake_worker_run),
    )
    executor = AsyncReplayExecutor(
        record_db="fake.json",
        record_id="fake",
        iterations=1,
        results_db_dir="/tmp",
        num_workers=2,
    )

    futures = [executor.submit({"v": i}) for i in range(5)]

    # Collect results
    results = [f.result(timeout=2) for f in futures]

    assert all(r["payload"] == "result" for r in results)
    assert sorted([r["exp_id"] for r in results]) == list(range(5))

    # Shut down
    executor.terminate()
