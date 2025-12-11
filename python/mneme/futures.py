# python/mneme/futures.py
import threading
from typing import Any, Optional, Dict


class EvalFuture:
    """
    Mneme Future: lightweight identity + synchronization.
    """

    def __init__(self, job_id: int, params: Dict[str, Any]):
        self.job_id = job_id
        self.params = params  # small dict of input params

        self._cond = threading.Condition()
        self._done = False
        self._result = None
        self._error = None

    def set_result(self, result: Dict[str, Any]):
        with self._cond:
            self._done = True
            self._result = result
            self._cond.notify_all()

    def set_error(self, error: str):
        with self._cond:
            self._done = True
            self._error = error
            self._cond.notify_all()

    def done(self):
        with self._cond:
            return self._done

    def result(self, timeout=None) -> Optional[Dict[str, Any]]:
        with self._cond:
            if not self._done:
                self._cond.wait(timeout=timeout)

            if self._error:
                raise RuntimeError(self._error)

            return self._result
