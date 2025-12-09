from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List


class FeedbackReducer(ABC):
    """
    Reduces a list of (configuration, value) into a single configuration.

    The parent tuner uses this to decide which child configuration wins.
    """

    @abstractmethod
    def reduce(self, configs: List[Any], values: List[float]) -> Any:
        pass


class ArgMinReducer(FeedbackReducer):
    """Return the configuration with the minimum value."""

    def reduce(self, configs: List[Any], values: List[float]) -> Any:
        if not configs or not values or len(configs) != len(values):
            raise ValueError("ArgMinReducer requires matching configs and values.")
        idx = min(range(len(values)), key=lambda i: values[i])
        return configs[idx]


class ArgMaxReducer(FeedbackReducer):
    """Return the configuration with the maximum value."""

    def reduce(self, configs: List[Any], values: List[float]) -> Any:
        if not configs or not values or len(configs) != len(values):
            raise ValueError("ArgMaxReducer requires matching configs and values.")
        idx = max(range(len(values)), key=lambda i: values[i])
        return configs[idx]
