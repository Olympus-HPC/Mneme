# python/mneme/futures.py
import threading
from typing import Any, Optional, Dict
from mneme.mneme_types import ExperimentConfiguration, ExperimentResult


class EvalFuture:
    """
    Mneme Future: lightweight identity + synchronization.
    """

    def __init__(self, job_id: int, config: ExperimentConfiguration):
        self.job_id = job_id
        self.config = config  # small dict of input params

        self._cond = threading.Condition()
        self._done = False
        self._result: Optional[ExperimentResult] = None
        self._error: Optional[str] = None

    def set_result(self, result: ExperimentResult):
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

    def result(self, timeout=None) -> Optional[ExperimentResult]:
        with self._cond:
            if not self._done:
                self._cond.wait(timeout=timeout)

            if self._error:
                if self._result is None:
                    return ExperimentResult(
                        error=self._error, failed=True, executed=False
                    )
                else:
                    self._result.error = self._error

                raise RuntimeError(self._error)

            return self._result
