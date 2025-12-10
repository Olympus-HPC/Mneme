import queue
import threading
import time
from enum import IntEnum
from multiprocessing import Event as ProcessEvent
from multiprocessing import Process
from multiprocessing import Queue as ProcessQueue
from queue import Queue as ThreadQueue
from threading import Event as ThreadEvent
from typing import Any, Dict

from mneme.futures import EvalFuture
from mneme.logging import logger
from mneme.replay_executor import TuneWorker


def pop(q, timeout):
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        return None


class TuneWorkerHandle:
    class StateMachine(IntEnum):
        SUBMIT = 1
        RECEIVE = 2

    def __init__(
        self,
        idx,
        global_q: ThreadQueue,
        record_db: str,
        record_id: str,
        device_id: int,
        iterations: int,
        results_db_dir: str,
    ):
        self.idx = idx
        self.global_q = global_q

        self._ipc_write_q = None
        self._ipc_read_q = None

        self._shutdown_event = ThreadEvent()
        self._action = self.StateMachine.SUBMIT

        self.record_db = record_db
        self.record_id = record_id
        self.device_id = device_id
        self.iterations = iterations
        self.results_db_dir = results_db_dir

        self._state = None  # ProcessEvent
        self._process = None  # Process
        self.current = None  # EvalFuture

        self._spawn_process()

        self._monitor_thread = threading.Thread(target=self._shadow_process_loop)
        self._monitor_thread.start()

    def _spawn_process(self):
        self._state = ProcessEvent()
        self._ipc_write_q = ProcessQueue()
        self._ipc_read_q = ProcessQueue()

        self._process = Process(
            target=TuneWorker.run,
            args=(
                self._ipc_write_q,
                self._ipc_read_q,
                self.record_db,
                self.record_id,
                self.device_id,
                self.iterations,
                self.results_db_dir,
                self._state,
            ),
            daemon=False,
        )
        self._process.start()
        self._action = self.StateMachine.SUBMIT

    # ------------------------------------------------------------
    # Result handling
    # ------------------------------------------------------------
    def _process_result(self, msg):
        future = self.current
        if future is None:
            return

        if future.job_id != msg["exp_id"]:
            raise RuntimeError(
                f"Worker {self.idx} received result for unexpected job "
                f"{msg['exp_id']} vs {future.job_id}"
            )

        future.set_result(msg)
        self.current = None

    def _try_receive(self):
        msg = pop(self._ipc_read_q, timeout=0.01)
        if msg is None:
            return

        self._process_result(msg)
        self._action = self.StateMachine.SUBMIT

    # ------------------------------------------------------------
    # Job submission
    # ------------------------------------------------------------
    def _submit(self):
        future: EvalFuture = pop(self.global_q, timeout=0.01)
        if future is None:
            return

        self.current = future
        msg = {
            "payload": "process",
            "data": future.params,
            "exp_id": future.job_id,
        }

        self._ipc_write_q.put(msg)
        self._action = self.StateMachine.RECEIVE

    # ------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------
    def _shadow_process_loop(self):
        while not self._shutdown_event.is_set():
            if not self._process.is_alive():
                # Crash recovery
                if self.current is not None:
                    self.current.set_error(
                        f"Worker crashed (exit {self._process.exitcode})"
                    )
                    self.current = None
                    _ = pop(self._ipc_read_q, timeout=0)

                self._spawn_process()
                continue

            # Wait for worker to initialize
            if not self._state.is_set():
                time.sleep(0.01)
                continue

            if self._action == self.StateMachine.SUBMIT:
                self._submit()
            else:
                self._try_receive()

        # Shutdown
        if self._process.is_alive():
            self._ipc_write_q.put({"payload": "terminate"})
            while self._process.is_alive():
                self._try_receive()
            self._process.join()

    def join(self):
        self._shutdown_event.set()
        self._monitor_thread.join()


# ============================================================================
# AsyncReplayExecutor
# ============================================================================
class AsyncReplayExecutor:
    def __init__(
        self,
        record_db: str,
        record_id: str,
        iterations: int,
        results_db_dir: str,
        num_workers: int,
    ):
        self.global_q = ThreadQueue()
        self._futures: Dict[int, EvalFuture] = {}
        self._next_id = 0
        self._lock = threading.Lock()

        self.workers = [
            TuneWorkerHandle(
                i,
                self.global_q,
                record_db,
                record_id,
                i,
                iterations,
                results_db_dir,
            )
            for i in range(num_workers)
        ]

    # ------------------------------------------------------------------
    # Submit new job (non-blocking)
    # ------------------------------------------------------------------
    def submit(self, params: Dict[str, Any]) -> EvalFuture:
        with self._lock:
            job_id = self._next_id
            self._next_id += 1

        future = EvalFuture(job_id, params)
        self._futures[job_id] = future
        self.global_q.put(future)
        return future

    def terminate(self):
        for w in self.workers:
            w.join()
