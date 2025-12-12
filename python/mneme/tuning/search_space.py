from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Iterable

from optuna.trial import Trial
from mneme.mneme_types import ExperimentConfiguration
from mneme.pipeline import PipelineManager


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

    def __init__(self, name: str, low: int, high: int, step: int = 1):
        super().__init__(name)
        if low > high:
            raise ValueError("IntRangeParam low must be <= high.")
        if step is None or step <= 0:
            raise ValueError("IntRangeParam step must be a positive integer.")

        self.low: int = low
        self.high: int = high
        self.step: int = step


class PipelineParam(BaseParam):
    """A parameter representing a sequence of compiler optimization passes [low, high] with optional step."""

    def __init__(self, name: str, num_draws: int):
        super().__init__(name)
        self.pass_manager = PipelineManager()
        self.num_draws = num_draws
        self.available_passes = self.pass_manager.get_passes()
        self.available_passes.sort()


class RealRangeParam(BaseParam):
    """A parameter representing an inclusive integer range [low, high] with optional step."""

    def __init__(self, name: str, low: float, high: float):
        super().__init__(name)
        if low > high:
            raise ValueError("RealRangeParam low must be <= high.")
        self.low: float = low
        self.high: float = high


def sample_optuna_param(trial: Trial, param: BaseParam) -> Any:
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
        return trial.suggest_int(name, param.low, param.high, step=param.step)

    if isinstance(param, RealRangeParam):
        # Use step-aware sampling
        return trial.suggest_float(name, param.low, param.high)

    if isinstance(param, PipelineParam):
        selected_pipelines = []
        pass_id = 0
        pass_count = defaultdict(lambda: 0)
        for i in range(param.num_draws):
            pass_name = param.available_passes[pass_id]
            count = pass_count[pass_name]
            use_it = trial.suggest_categorical(f"use_{pass_name}_{count}", [0, 1])
            if not use_it:
                continue

            # Priority *within* this round
            prio = trial.suggest_int(f"prio_{pass_name}_{count}", 0, param.num_draws)

            # (round, prio, index, pass)
            selected_pipelines.append((pass_name, prio, pass_count[pass_name]))
            pass_count[pass_name] += 1
            pass_id = (pass_id + 1) % len(param.available_passes)

        selected_pipelines.sort(key=lambda x: (x[1], x[2]))
        concrete_passes = param.pass_manager.get_concrete_passes()
        return param.pass_manager.to_string(
            [concrete_passes[v[0]] for v in selected_pipelines]
        )

    raise TypeError(f"Unsupported parameter type in Optuna: {type(param)}")


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
    elif isinstance(param, RealRangeParam):
        return random.uniform(param.low, param.high)
    elif isinstance(param, PipelineParam):
        if param.pipelines is not None:
            return random.choice(param.pipelines)
        return param.generator()

    else:
        raise TypeError(f"Unsupported parameter type: {type(param)}")


class SearchSpace(ABC):
    """
    Declarative representation of a search space.

    A SearchSpace describes:
      • The **primary tunable dimensions** (dimensions())
      • Any **derived parameters** computed from these (derived())
      • Any **constraints** on valid configurations (constraints())
    """

    @abstractmethod
    def dimensions(self) -> Dict[str, BaseParam]:
        """
        Return the top-level parameters of this search space.
        Keys are parameter names, values are BaseParam instances.
        """
        pass

    @abstractmethod
    def derived(self, params: Dict[str, Any]) -> ExperimentConfiguration:
        """
        Compute derived parameters from the given primary parameters.

        Example:
          - compute launch_bounds if block_dim/max_threads implies it
          - compute grid_dim automatically from problem size
        """
        return {}

    @abstractmethod
    def constraints(self, params: Dict[str, Any]) -> bool:
        """
        Validate that this parameter assignment is legal.

        Returns:
            True if the assignment is valid, False otherwise.
        """
        return True

    def sample_random(self) -> Dict[str, Any]:
        """
        Generate one valid random configuration from this search space.
        """

        MAX_RETRIES = 1000
        dims = self.dimensions()

        for _ in range(MAX_RETRIES):
            result = {}

            # Step 1: sample primary dimensions
            for name, param in dims.items():
                result[name] = sample_random_param(param)

            # Step 2: constraints
            if self.constraints(result):
                return {"parameters": result}

        raise RuntimeError(
            f"Failed to produce a valid random sample after {MAX_RETRIES} attempts."
        )

    def sample_optuna(self, study) -> Tuple[ExperimentConfiguration, Trial]:
        """
        Generate a valid configuration using Optuna.

        Steps:
          1. Use trial.suggest_* to sample primary dimensions
          2. Compute derived parameters
          3. Enforce constraints
          4. Retry if needed
        """

        MAX_RETRIES = 1000
        dims = self.dimensions()

        for _ in range(MAX_RETRIES):
            trial = study.ask()
            config = {}

            # Step 1: primary dimension sampling
            for name, param in dims.items():
                config[name] = sample_optuna_param(trial, param)

            # Step 2: constraints
            derived_config = self.derived(config)
            if self.constraints(derived_config):
                return derived_config, trial
            else:
                study.tell(trial, (1 << 64) - 1)

        raise RuntimeError(
            f"Failed to generate a valid Optuna sample after {MAX_RETRIES} attempts."
        )

    def sample_exhaustive(self) -> Iterable[Dict[str, Any]]:
        """
        Exhaustively enumerate all configurations for dimensions that are finite.
        This is used only for Random-Exhaustive search.
        """
        dims = self.dimensions()
        keys = list(dims.keys())

        # Build value lists
        value_lists = []
        for dim in dims.values():
            if isinstance(dim, FixedParam):
                value_lists.append([dim.config])
            elif isinstance(dim, BoolParam):
                value_lists.append(list(dim.choices))
            elif isinstance(dim, IntRangeParam):
                lo = dim.low
                hi = dim.high
                step = dim.step
                value_lists.append(list(range(lo, hi + 1, step)))
            elif isinstance(dim, CategoricalParam):
                value_lists.append(list(dim.choices))
            elif isinstance(dim, RealRangeParam):
                raise ValueError("Cannot enumerate real space.")
            else:
                raise ValueError("Cannot enumerate custom dimension.")

        # Cartesian product
        from itertools import product

        for combo in product(*value_lists):
            params = dict(zip(keys, combo))
            if self.constraints(params):
                yield {"parameters": params}
