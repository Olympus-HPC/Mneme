from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List

from mneme.logging import logger
from mneme.tuning.search_space import SearchSpace


class SamplingStrategy(ABC):
    """
    A generic sampler that yields configuration dictionaries.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Return an iterator that yields configs."""
        pass


class ExhaustiveSamplingStrategy(SamplingStrategy):
    def __init__(self, search_space):
        self.space = search_space

    def __iter__(self):
        for params in self.space.sample_exhaustive():
            yield params


class RandomSamplingStrategy(SamplingStrategy):
    def __init__(self, search_space, num_samples: int):
        self.space = search_space
        self.num_samples = num_samples

    def __iter__(self):
        for _ in range(self.num_samples):
            yield self.space.sample_random()


class OptunaSamplingStrategy(SamplingStrategy):
    def __init__(self, search_space, study, n_trials):
        self.space = search_space
        self.study = study
        self.n_trials = n_trials
        logger.debug(
            f"{self.__class__.__name__} Total number of previously executed trials are {len(self.study.trials)} and total requested trials are {self.n_trials}"
        )

    def __iter__(self):
        while len(self.study.trials) < self.n_trials:
            params = self.space.sample_optuna(self.study)
            yield params
