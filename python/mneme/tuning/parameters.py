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
        self.choices: List[bool] = [True, False]


class CategoricalParam(BaseParam):
    """A parameter with an explicit finite set of choices."""

    def __init__(self, name: str, choices: List[Any]):
        super().__init__(name)
        if not choices:
            raise ValueError("CategoricalParam must have at least one choice.")
        self.choices: List[Any] = list(choices)


class IntRangeParam(BaseParam):
    """A parameter representing an inclusive integer range [low, high] with optional step."""

    def __init__(self, name: str, low: int, high: int, step: Optional[int] = 1):
        super().__init__(name)
        if low > high:
            raise ValueError("IntRangeParam low must be <= high.")
        if step is None or step <= 0:
            raise ValueError("IntRangeParam step must be a positive integer.")

        self.low: int = low
        self.high: int = high
        self.step: int = step


class PipelineParam(BaseParam):
    """
    Parameter representing LLVM pass pipelines.

    Supports two mutually-exclusive modes:

      1. Finite set of pipelines:
           PipelineParam("pipeline", pipelines=[...])

      2. Generator-driven infinite/unknown space:
           PipelineParam("pipeline", generator=fn)

    """

    def __init__(
        self,
        name: str,
        pipelines: Optional[List[str]] = None,
        generator: Optional[Callable[[], str]] = None,
    ):
        super().__init__(name)

        # If *both* are provided, this is an ambiguous configuration.
        if pipelines is not None and generator is not None:
            raise ValueError(
                "PipelineParam cannot define both `pipelines` and `generator`. "
                "Choose one mode."
            )

        # If neither is provided, nothing to sample from.
        if pipelines is None and generator is None:
            raise ValueError(
                "PipelineParam requires either a `pipelines` list or a `generator` callable."
            )

        if pipelines is not None and not pipelines:
            raise ValueError("PipelineParam pipelines list cannot be empty.")

        self.pipelines: Optional[List[str]] = (
            list(pipelines) if pipelines is not None else None
        )
        self.generator: Optional[Callable[[], str]] = generator

