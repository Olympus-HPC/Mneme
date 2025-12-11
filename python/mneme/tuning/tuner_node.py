from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from .search_space import SearchSpace


class TunerNode(ABC):
    """
    A node in a hierarchical tuning system.

    Responsibilities:
      • ask()  -> produce a candidate configuration
      • tell(params, feedback) -> receive results from evaluation
      • evaluate(fixed_params) -> evaluate a fully known configuration

    Concrete tuners must implement ask() and tell().
    Higher-level tuners (in PR7+) may override evaluate() to run
    sub-tuners, reduce results, or perform nested search.
    """

    def __init__(self, space: SearchSpace, sampler):
        self.space = space
        self.sampler = sampler

    # ------------------------------------------------------------
    # Core tuning interface
    # ------------------------------------------------------------

    @abstractmethod
    def ask(self, trial=None) -> Dict[str, Any]:
        """
        Produce the next configuration to evaluate.
        Must return a dict mapping parameter names to values.
        """
        pass

    @abstractmethod
    def tell(self, params: Dict[str, Any], feedback: float):
        """
        Notify the tuner of the measured feedback for a configuration.

        params:   The configuration produced by ask()
        feedback: The evaluation metric (e.g., runtime)
        """
        pass

    # ------------------------------------------------------------
    # Evaluation hook (used later by hierarchical operators)
    # ------------------------------------------------------------

    @abstractmethod
    def evaluate(self, fixed_params: Dict[str, Any]) -> float:
        """
        Evaluate a fully-fixed configuration and return the feedback value.

        Leaf nodes will override this to perform actual kernel evaluation.
        Parent nodes will override this to:
            - expand fixed_params into child configurations
            - distribute evaluation to children
            - aggregate child results using a FeedbackReducer
        """
        pass


class PipelineParentTuner(TunerNode):
    def __init__(self, space, sampler, blockdim_tuner, reducer):
        super().__init__(space, sampler)
        self.child = blockdim_tuner
        self.reducer = reducer

    def ask(self, trial=None):
        return self.sampler.sample(self.space, trial)

    def tell(self, params, feedback):
        # simple bookkeeping for now
        pass

    def evaluate(self, fixed_params):
        # Evaluate using child tuner
        result = self.child.evaluate(fixed_params)

        # Reduce feedback
        reduced = self.reducer.reduce([result])

        self.tell(fixed_params, reduced)
        return reduced
