from __future__ import annotations
from typing import Any

import optuna

from .parameters import (
    BaseParam,
    FixedParam,
    BoolParam,
    CategoricalParam,
    IntRangeParam,
    PipelineParam,
)


def sample_optuna_param(trial: optuna.trial.Trial, param: BaseParam) -> Any:
    """
    Sample a single parameter using Optuna.
    """

    name = param.name

    if isinstance(param, FixedParam):
        return param.value

    if isinstance(param, BoolParam):
        return trial.suggest_categorical(name, param.choices)

    if isinstance(param, CategoricalParam):
        return trial.suggest_categorical(name, param.choices)

    if isinstance(param, IntRangeParam):
        # Use step-aware sampling
        return trial.suggest_int(name, param.low, param.high, step=param.step)

    if isinstance(param, PipelineParam):

        # Case 1 — finite list: we can use Optuna categorical
        if param.pipelines is not None:
            return trial.suggest_categorical(name, param.pipelines)

        # Case 2 — generator (non-enumerable): we must manually record the result
        value = param.generator()
        trial.set_user_attr(name, value)
        return value

    raise TypeError(f"Unsupported parameter type in Optuna: {type(param)}")

