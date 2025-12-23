from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Iterable

from optuna.trial import Trial
from mneme.mneme_types import ExperimentConfiguration
from mneme.pipeline import PipelineManager


class BaseParam(ABC):
    """
    Abstract base class for tuning parameter definitions.

    A ``BaseParam`` represents a single tunable dimension in a :class:`SearchSpace`.
    Concrete subclasses define the domain (fixed, boolean, categorical, numeric range,
    pass-pipeline, etc.) and provide metadata needed by sampling backends.

    Attributes
    ----------
    name : str
        Logical name of the parameter. This name is typically used as the Optuna
        parameter name when sampling via :class:`optuna.trial.Trial`.
    """

    def __init__(self, name: str):
        self.name = name


class FixedParam(BaseParam):
    """
    A parameter with a single fixed value.

    This is useful for keeping a dimension present in the search space interface
    while effectively disabling tuning for that parameter.
    """

    def __init__(self, name: str, value: Any):
        """
        Parameters
        ----------
        name : str
            Name of the parameter.
        value : Any
            Fixed value returned by all samplers.
        """
        super().__init__(name)
        self.value = value


class BoolParam(BaseParam):
    """
    A boolean parameter.

    The domain is ``{True, False}``.
    """

    def __init__(self, name: str):
        """
        Parameters
        ----------
        name : str
            Name of the parameter.
        """
        super().__init__(name)
        self.choices: List[bool] = [True, False]


class CategoricalParam(BaseParam):
    """
    A parameter with an explicit finite set of choices.

    The domain is the provided list of choices.
    """

    def __init__(self, name: str, choices: List[Any]):
        """
        Parameters
        ----------
        name : str
            Name of the parameter.
        choices : list
            Finite set of allowed values.

        Raises
        ------
        ValueError
            If ``choices`` is empty.
        """
        super().__init__(name)
        if not choices:
            raise ValueError("CategoricalParam must have at least one choice.")
        self.choices: List[Any] = list(choices)


class IntRangeParam(BaseParam):
    """
    Integer range parameter.

    Represents an inclusive integer range ``[low, high]`` with a positive step.
    Sampling produces values from the discrete set:
    ``{low, low+step, ..., high}`` (assuming divisibility).

    Notes
    -----
    * This class models a discrete domain (even though it is expressed as a range).
    """

    def __init__(self, name: str, low: int, high: int, step: int = 1):
        """
        Parameters
        ----------
        name : str
            Name of the parameter.
        low : int
            Inclusive lower bound.
        high : int
            Inclusive upper bound.
        step : int, optional
            Step size (must be positive).

        Raises
        ------
        ValueError
            If ``low > high`` or ``step <= 0``.
        """
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
    Parameter representing a compiler optimization pipeline / pass sequence.

    This parameter is specialized: its domain is defined by the available passes
    provided by :class:`PipelineManager` and an internal sampling scheme that can
    select passes, order them, and optionally select multiple occurrences.

    Attributes
    ----------
    pass_manager : PipelineManager
        Helper that provides available passes and serialization to pipeline strings.
    available_passes : list of str
        Sorted list of pass identifiers that may be selected.
    num_draws : int
        Upper bound on how many pass-selection decisions are made when sampling.
        (Interpretation depends on the sampling backend.)
    Notes
    -----
    This parameter must be used cautiously, the pipeline sequence is a combinatorial space on each own
    composed of more than 100+ of dimensions. We mainly provide this for completeness, however applying
    `TPE` or `NGSAII` on this space is not recommended.

    """

    def __init__(self, name: str, num_draws: int):
        """
        Parameters
        ----------
        name : str
            Name of the parameter (used as a logical key in the search space).
        num_draws : int
            Number of sampling "draws" used when constructing a pipeline.

        Notes
        -----
        * The available pass list is obtained from :class:`PipelineManager` and is
          sorted to ensure stable iteration order.
        """
        super().__init__(name)
        self.pass_manager = PipelineManager()
        self.num_draws = num_draws
        self.available_passes = self.pass_manager.get_passes()
        self.available_passes.sort()


class RealRangeParam(BaseParam):
    """
    Real-valued range parameter.

    Represents an inclusive real range ``[low, high]``. This parameter is intended
    for continuous sampling backends (e.g., Optuna suggest_float).

    Notes
    -----
    * This parameter is not exhaustively enumerable.
    """

    def __init__(self, name: str, low: float, high: float):
        """
        Parameters
        ----------
        name : str
            Name of the parameter.
        low : float
            Inclusive lower bound.
        high : float
            Inclusive upper bound.

        Raises
        ------
        ValueError
            If ``low > high``.
        """
        super().__init__(name)
        if low > high:
            raise ValueError("RealRangeParam low must be <= high.")
        self.low: float = low
        self.high: float = high


def sample_optuna_param(trial: Trial, param: BaseParam) -> Any:
    """
    Sample a single parameter value using an Optuna trial.

    This function maps :class:`BaseParam` subclasses to the appropriate Optuna
    sampling primitive (e.g., ``suggest_int``, ``suggest_float``, or
    ``suggest_categorical``). For specialized parameter types (e.g.,
    :class:`PipelineParam`), it implements a custom sampling scheme that encodes
    selection and ordering via multiple Optuna decision variables.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial used to generate parameter suggestions.
    param : BaseParam
        Parameter definition describing the domain and sampling behavior.

    Returns
    -------
    Any
        Sampled value for the parameter.

    Raises
    ------
    TypeError
        If the parameter type is not supported.
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
    """
    Generate one random sample for a single parameter.

    Parameters
    ----------
    param : BaseParam
        Parameter definition describing the domain.

    Returns
    -------
    Any
        Randomly sampled value.

    Raises
    ------
    TypeError
        If the parameter type is not supported.
    """

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
    Declarative representation of a tuning search space.

    A :class:`SearchSpace` describes:

      1) The **primary tunable dimensions** (see :meth:`dimensions`)
      2) Any **derived configuration** computed from sampled parameters (see :meth:`derived`)
      3) **Constraints** that determine whether a sampled assignment is valid (see :meth:`constraints`)

    The base class also provides helper sampling routines for different backends
    (random sampling, Optuna sampling, and exhaustive enumeration of finite domains).

    Notes
    -----
    * Concrete subclasses should keep :meth:`dimensions` purely declarative and
      implement domain-specific logic inside :meth:`derived` and :meth:`constraints`.
    """

    @abstractmethod
    def dimensions(self) -> Dict[str, BaseParam]:
        """
        Return the top-level tunable parameters of this search space.

        Returns
        -------
        dict
            Mapping from parameter name to a :class:`BaseParam` instance describing
            that parameter’s domain.
        """
        pass

    @abstractmethod
    def derived(self, params: Dict[str, Any]) -> ExperimentConfiguration:
        """
        Compute a full experiment configuration from sampled primary parameters.

        Parameters
        ----------
        params : dict
            Dictionary mapping primary parameter names to sampled values.

        Returns
        -------
        ExperimentConfiguration
            Fully specified experiment configuration derived from the sampled values.

        Notes
        -----
        * Derived configuration may include both original parameters and additional
          fields computed from them (e.g., mapping a normalized fraction to an integer
          launch-bounds parameter).
        """
        return {}

    @abstractmethod
    def constraints(self, params: Dict[str, Any]) -> bool:
        """
        Validate that a parameter assignment or derived configuration is legal.

        Parameters
        ----------
        params : dict
            Parameter assignment to validate. Implementations may choose whether this
            expects only primary parameters or a derived configuration, depending on
            the calling context.

        Returns
        -------
        bool
            ``True`` if the assignment is valid, otherwise ``False``.
        """
        return True

    def sample_random(self) -> Dict[str, Any]:
        """
        Generate one valid random sample from this search space.

        This method repeatedly samples all primary dimensions using
        :func:`sample_random_param` and applies :meth:`constraints`. Sampling is retried
        up to ``MAX_RETRIES`` times.

        Returns
        -------
        dict
            A dictionary of the form ``{"parameters": <param-dict>}`` containing the
            sampled primary parameters.

        Raises
        ------
        RuntimeError
            If a valid configuration cannot be produced within the retry budget.
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

        This method uses an Optuna study to create trials and sample primary
        dimensions, then computes the derived experiment configuration and enforces
        constraints. Invalid samples are immediately reported back to the study with
        a sentinel objective value.

        Parameters
        ----------
        study : optuna.study.Study
            Optuna study used to create trials and manage search state.

        Returns
        -------
        (ExperimentConfiguration, Trial)
            A tuple containing:

            * **ExperimentConfiguration** – derived configuration produced by :meth:`derived`.
            * **Trial** – Optuna trial associated with the sampled configuration.

        Raises
        ------
        RuntimeError
            If a valid configuration cannot be produced within the retry budget.

        Notes
        -----
        * This routine uses ``study.ask()`` / ``study.tell()`` rather than Optuna’s
          higher-level objective API to support asynchronous evaluation.
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
        Exhaustively enumerate all configurations for finite domains.

        This helper enumerates the cartesian product of all dimension value lists
        for parameters with finite, enumerable domains (fixed, boolean, categorical,
        and integer ranges). Real-valued parameters are not enumerable.

        Yields
        ------
        dict
            Dictionaries of the form ``{"parameters": <param-dict>}`` for each valid
            parameter assignment satisfying :meth:`constraints`.

        Raises
        ------
        ValueError
            If an attempt is made to enumerate a non-enumerable parameter type.
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
            derived_config = self.derived(params)
            if self.constraints(derived_config):
                yield derived_config
