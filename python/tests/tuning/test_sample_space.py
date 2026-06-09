import pytest


# Adjust this import to your real module path
import mneme.tuning.search_space as mod
from mneme.mneme_types import ExperimentConfiguration


# -----------------------------------------------------------------------------
# Small fakes
# -----------------------------------------------------------------------------


class FakeTrial:
    def __init__(self):
        self.calls = []

    def suggest_categorical(self, name, choices):
        self.calls.append(("categorical", name, tuple(choices)))
        # Deterministic: pick first choice
        return choices[0]

    def suggest_int(self, name, low, high, step=1):
        self.calls.append(("int", name, low, high, step))
        # Deterministic: pick low
        return low

    def suggest_float(self, name, low, high):
        self.calls.append(("float", name, low, high))
        # Deterministic: pick mid-ish
        return (low + high) / 2.0


class FakeStudy:
    def __init__(self, trials):
        self._trials = list(trials)
        self.tells = []

    def ask(self):
        if not self._trials:
            raise RuntimeError("No more trials")
        return self._trials.pop(0)

    def tell(self, trial, value):
        self.tells.append((trial, value))


class FakePipelineManager:
    def __init__(self):
        self.to_string_calls = []

    def get_passes(self):
        # Unsorted on purpose; PipelineParam should sort these
        return ["bpass", "apass"]

    def get_concrete_passes(self):
        # Map pass-id -> concrete representation (e.g., pass objects/strings)
        return {"apass": "A", "bpass": "B"}

    def to_string(self, concrete_list):
        self.to_string_calls.append(list(concrete_list))
        # Make a stable string
        return "PIPE[" + ",".join(concrete_list) + "]"


# -----------------------------------------------------------------------------
# Constructors / validation
# -----------------------------------------------------------------------------


def test_categorical_param_requires_choices():
    with pytest.raises(ValueError):
        mod.CategoricalParam("x", [])


def test_int_range_param_validation():
    with pytest.raises(ValueError):
        mod.IntRangeParam("x", 5, 4)

    with pytest.raises(ValueError):
        mod.IntRangeParam("x", 0, 10, step=0)

    p = mod.IntRangeParam("x", 0, 10, step=2)
    assert p.low == 0
    assert p.high == 10
    assert p.step == 2


def test_real_range_param_validation():
    with pytest.raises(ValueError):
        mod.RealRangeParam("r", 2.0, 1.0)

    p = mod.RealRangeParam("r", 1.0, 2.0)
    assert p.low == 1.0
    assert p.high == 2.0


# -----------------------------------------------------------------------------
# sample_optuna_param: Fixed/Bool/Categorical/Int/Real
# -----------------------------------------------------------------------------


def test_sample_optuna_param_fixed():
    t = FakeTrial()
    p = mod.FixedParam("x", 123)
    assert mod.sample_optuna_param(t, p) == 123
    assert t.calls == []


def test_sample_optuna_param_bool():
    t = FakeTrial()
    p = mod.BoolParam("flag")
    v = mod.sample_optuna_param(t, p)
    assert v in [True, False]
    assert t.calls[0][0] == "categorical"
    assert t.calls[0][1] == "flag"


def test_sample_optuna_param_categorical():
    t = FakeTrial()
    p = mod.CategoricalParam("cat", ["a", "b"])
    v = mod.sample_optuna_param(t, p)
    assert v == "a"
    assert ("categorical", "cat", ("a", "b")) in t.calls


def test_sample_optuna_param_int_range():
    t = FakeTrial()
    p = mod.IntRangeParam("i", 4, 10, step=2)
    v = mod.sample_optuna_param(t, p)
    assert v == 4
    assert ("int", "i", 4, 10, 2) in t.calls


def test_sample_optuna_param_real_range():
    t = FakeTrial()
    p = mod.RealRangeParam("r", 0.0, 1.0)
    v = mod.sample_optuna_param(t, p)
    assert v == 0.5
    assert ("float", "r", 0.0, 1.0) in t.calls


# -----------------------------------------------------------------------------
# PipelineParam + sample_optuna_param(PipelineParam)
# -----------------------------------------------------------------------------


def test_pipeline_param_sorts_passes(monkeypatch):
    monkeypatch.setattr(mod, "PipelineManager", FakePipelineManager, raising=True)
    p = mod.PipelineParam("passes", num_draws=2)
    assert p.available_passes == ["apass", "bpass"]


def test_sample_optuna_param_pipeline_param_builds_pipeline(monkeypatch):
    # Ensure PipelineParam uses our fake manager
    monkeypatch.setattr(mod, "PipelineManager", FakePipelineManager, raising=True)

    trial = FakeTrial()
    p = mod.PipelineParam("passes", num_draws=3)

    # We want at least one "use_it" to be true.
    # FakeTrial.suggest_categorical returns first choice; so make first choice = 1.
    # Patch trial method only for those "use_" queries.
    orig_cat = trial.suggest_categorical

    def suggest_categorical(name, choices):
        if name.startswith("use_"):
            # Return 1 to "enable" pass
            trial.calls.append(("categorical", name, tuple(choices)))
            return 1
        return orig_cat(name, choices)

    trial.suggest_categorical = suggest_categorical

    out = mod.sample_optuna_param(trial, p)
    assert out.startswith("PIPE[")
    # Ensure PipelineManager.to_string_


class ConcreteSpace(mod.SearchSpace):
    def __init__(self, valid_after=1, dimensions=None):
        self.valid_after = valid_after
        self.constraint_calls = 0
        self._dimensions = dimensions or {
            "fixed": mod.FixedParam("fixed", 7),
            "flag": mod.BoolParam("flag"),
            "level": mod.IntRangeParam("level", 1, 3),
            "mode": mod.CategoricalParam("mode", ["a", "b"]),
        }

    def dimensions(self):
        return self._dimensions

    def derived(self, params):
        return ExperimentConfiguration(codegen_opt=int(params.get("level", 1)))

    def constraints(self, params):
        self.constraint_calls += 1
        return self.constraint_calls >= self.valid_after


def test_sample_random_param_covers_all_non_pipeline_types(monkeypatch):
    monkeypatch.setattr(mod.random, "choice", lambda choices: choices[-1])
    monkeypatch.setattr(mod.random, "randrange", lambda n: n - 1)
    monkeypatch.setattr(mod.random, "uniform", lambda low, high: high)

    assert mod.sample_random_param(mod.FixedParam("x", "fixed")) == "fixed"
    assert mod.sample_random_param(mod.BoolParam("flag")) is False
    assert mod.sample_random_param(mod.CategoricalParam("cat", ["a", "b"])) == "b"
    assert mod.sample_random_param(mod.IntRangeParam("i", 2, 6, step=2)) == 6
    assert mod.sample_random_param(mod.RealRangeParam("r", 1.0, 2.0)) == 2.0


def test_sample_random_param_pipeline_param(monkeypatch):
    monkeypatch.setattr(mod, "PipelineManager", FakePipelineManager, raising=True)
    monkeypatch.setattr(mod.random, "choice", lambda choices: True)

    pipeline = mod.PipelineParam("passes", num_draws=2)

    assert mod.sample_random_param(pipeline) == "PIPE[A,B]"


def test_sample_random_param_rejects_unknown_param():
    class UnknownParam(mod.BaseParam):
        pass

    with pytest.raises(TypeError, match="Unsupported parameter type"):
        mod.sample_random_param(UnknownParam("unknown"))


def test_search_space_sample_random_retries_until_valid(monkeypatch):
    monkeypatch.setattr(mod.random, "choice", lambda choices: choices[0])
    monkeypatch.setattr(mod.random, "randrange", lambda n: 0)
    space = ConcreteSpace(valid_after=2)

    sample = space.sample_random()

    assert sample == {"parameters": {"fixed": 7, "flag": True, "level": 1, "mode": "a"}}
    assert space.constraint_calls == 2


def test_search_space_sample_random_raises_after_retry_budget(monkeypatch):
    monkeypatch.setattr(mod.random, "choice", lambda choices: choices[0])
    monkeypatch.setattr(mod.random, "randrange", lambda n: 0)
    space = ConcreteSpace(valid_after=1001)

    with pytest.raises(RuntimeError, match="Failed to produce a valid random sample"):
        space.sample_random()

    assert space.constraint_calls == 1000


def test_search_space_sample_optuna_tells_invalid_then_returns_valid():
    trials = [FakeTrial(), FakeTrial()]
    study = FakeStudy(trials)
    space = ConcreteSpace(valid_after=2)

    config, trial = space.sample_optuna(study)

    assert config.codegen_opt == 1
    assert trial is trials[1]
    assert study.tells == [(trials[0], (1 << 64) - 1)]


def test_search_space_sample_optuna_raises_after_retry_budget():
    trials = [FakeTrial() for _ in range(1000)]
    study = FakeStudy(trials)
    space = ConcreteSpace(valid_after=1001)

    with pytest.raises(RuntimeError, match="Failed to generate a valid Optuna sample"):
        space.sample_optuna(study)

    assert len(study.tells) == 1000


def test_search_space_sample_exhaustive_yields_valid_derived_configs():
    space = ConcreteSpace(valid_after=1)

    configs = list(space.sample_exhaustive())

    assert len(configs) == 12
    assert {config.codegen_opt for config in configs} == {1, 2, 3}


def test_search_space_sample_exhaustive_rejects_non_enumerable_params():
    real_space = ConcreteSpace(dimensions={"r": mod.RealRangeParam("r", 0.0, 1.0)})
    with pytest.raises(ValueError, match="Cannot enumerate real space"):
        list(real_space.sample_exhaustive())

    class UnknownParam(mod.BaseParam):
        pass

    custom_space = ConcreteSpace(dimensions={"x": UnknownParam("x")})
    with pytest.raises(ValueError, match="Cannot enumerate custom dimension"):
        list(custom_space.sample_exhaustive())
