from __future__ import annotations
import random
from typing import Any, Dict

from .parameters import (
    BaseParam,
    FixedParam,
    BoolParam,
    CategoricalParam,
    IntRangeParam,
    PipelineParam,
)
from .search_space import SearchSpace


# ---------------------------------------------------------
# Single parameter random sampler
# ---------------------------------------------------------

def sample_random_param(param: BaseParam) -> Any:
    """Generate one random sample for a single parameter."""

    if isinstance(param, FixedParam):
        return param.value

    elif isinstance(param, BoolParam):
        return random.choice(param.choices)

    elif isinstance(param, CategoricalParam):
        return random.choice(param.choices)

    elif isinstance(param, IntRangeParam):
        low = param.low
        high = param.high
        step = param.step
        # Sample from {low, low+step, ..., high}
        n = ((high - low) // step) + 1
        idx = random.randrange(n)
        return low + idx * step

    elif isinstance(param, PipelineParam):
        if param.pipelines is not None:
            return random.choice(param.pipelines)
        return param.generator()

    else:
        raise TypeError(f"Unsupported parameter type: {type(param)}")

