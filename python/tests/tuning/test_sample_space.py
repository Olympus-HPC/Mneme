import types
import pytest


# Adjust this import to your real module path
import mneme.tuning.search_space as mod


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
