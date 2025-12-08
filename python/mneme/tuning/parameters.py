# mneme/tuning/parameters.py

from __future__ import annotations

from abc import ABC
from typing import Any, Callable, List, Optional


class BaseParam(ABC):
    """Abstract base class for all tuning parameter definitions."""

    def __init__(self, name: str):
        self.name = name


class FixedParam(BaseParam):
    """A parameter that always has a single, fixed value."""

    def __init__(self, name: str, value: Any):
        super().__init__(name)
        self.value = value


class BoolParam(BaseParam):
    """A boolean parameter with values True or False."""

    def __init__(self, name: str):
        super().__init__(name)
        # We keep the domain explicit in case we want to expose it later.
        self.choices: List[bool] = [True, False]


class CategoricalParam(BaseParam):
    """A parameter with an explicit finite set of choices."""

    def __init__(self, name: str, choices: List[Any]):
        super().__init__(name)
        if not choices:
            raise ValueError("CategoricalParam must have at least one choice.")
        self.choices: List[Any] = list(choices)


class IntRangeParam(BaseParam):
    """A parameter representing an inclusive integer range [low, high]."""

    def __init__(self, name: str, low: int, high: int, step: Optional[int] = 1):
        super().__init__(name)
        if low > high:
            raise ValueError("IntRangeParam low must be <= high.")
        self.low: int = low
        self.high: int = high
        self.step: Optional[int] = step


class PipelineParam(BaseParam):
    """
    Parameter representing LLVM pass pipelines.

    This supports two modes:

      1. Finite set of known pipelines:
           PipelineParam("pipeline", pipelines=[...])

      2. Generator-driven pipelines (non-enumerable space):
           PipelineParam("pipeline", generator=my_pipeline_generator)

         where `my_pipeline_generator: () -> str` returns a single pipeline
         string when called.

    """

    def __init__(
        self,
        name: str,
        pipelines: Optional[List[str]] = None,
        generator: Optional[Callable[[], str]] = None,
    ):
        super().__init__(name)

        if pipelines is None and generator is None:
            raise ValueError(
                "PipelineParam requires either a finite `pipelines` list "
                "or a `generator` callable."
            )

        if pipelines is not None and not pipelines:
            raise ValueError("PipelineParam pipelines list cannot be empty.")

        self.pipelines: Optional[List[str]] = (
            list(pipelines) if pipelines is not None else None
        )
        self.generator: Optional[Callable[[], str]] = generator

