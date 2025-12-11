from typing import Dict, Any
import sys

from mneme.tuning.tuner_node import TunerNode
from mneme.tuning.search_space import SearchSpace
from mneme.async_executor import AsyncReplayExecutor


class LeafTunerNode(TunerNode):
    """
    Leaf tuner that evaluates configurations using AsyncReplayExecutor.
    """

    def __init__(self, space: SearchSpace, sampler, executor: AsyncReplayExecutor):
        super().__init__(space, sampler)
        self.executor = executor

    def ask(self, trial=None) -> Dict[str, Any]:
        return self.sampler.sample(self.space, trial)

    def tell(self, params: Dict[str, Any], feedback: float):
        logger.debug(f"[LeafTunerNode] {params} → {feedback}")
        pass

    def evaluate(self, fixed_params: Dict[str, Any]) -> float:
        """
        Synchronous evaluation. Returns the exec_time as feedback.
        """
        result = self.executor.evaluate(fixed_params)
        exec_time = sys.maxsize
        if result is not None:
            exec_time = result["data"]["exec_time"]

        self.tell(fixed_params, exec_time)
        return exec_time
