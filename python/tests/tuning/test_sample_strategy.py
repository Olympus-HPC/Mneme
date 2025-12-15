import pytest

import mneme.tuning.sample_strategy as mod


class FakeSpace:
    def __init__(self):
        self.random_calls = 0
        self.optuna_calls = 0

    def sample_exhaustive(self):
        # return iterable of items the strategy should yield
        yield {"parameters": {"a": 1}}
        yield {"parameters": {"a": 2}}

    def sample_random(self):
        self.random_calls += 1
        return {"parameters": {"rand": self.random_calls}}

    def sample_optuna(self, study):
        self.optuna_calls += 1
        # Return whatever your SearchSpace.sample_optuna returns.
        # In your implementation earlier it returns (derived_config, trial),
        # so we mimic that shape here.
        return ({"cfg": self.optuna_calls}, f"trial-{self.optuna_calls}")


class FakeStudy:
    def __init__(self, n_existing=0):
        # Only len(study.trials) matters for the strategy loop
        self.trials = [object() for _ in range(n_existing)]


def test_exhaustive_sampling_strategy_yields_space_items():
    space = FakeSpace()
    strat = mod.ExhaustiveSamplingStrategy(space)

    out = list(iter(strat))
    assert out == [{"parameters": {"a": 1}}, {"parameters": {"a": 2}}]


def test_random_sampling_strategy_calls_sample_random_n_times():
    space = FakeSpace()
    strat = mod.RandomSamplingStrategy(space, num_samples=5)

    out = list(iter(strat))
    assert len(out) == 5
    assert space.random_calls == 5
    # Deterministic outputs from FakeSpace.sample_random()
    assert out[0] == {"parameters": {"rand": 1}}
    assert out[-1] == {"parameters": {"rand": 5}}


def test_optuna_sampling_strategy_yields_until_n_trials_reached(monkeypatch):
    space = FakeSpace()
    study = FakeStudy(n_existing=0)

    orig_sample_optuna = space.sample_optuna  # capture BEFORE monkeypatching

    def wrapped_sample_optuna(study_obj):
        study_obj.trials.append(object())  # simulate trial count increasing
        return orig_sample_optuna(study_obj)

    monkeypatch.setattr(space, "sample_optuna", wrapped_sample_optuna, raising=True)

    strat = mod.OptunaSamplingStrategy(space, study=study, n_trials=3)
    out = list(iter(strat))

    assert len(out) == 3
    assert len(study.trials) == 3
    assert out[0] == ({"cfg": 1}, "trial-1")
    assert out[-1] == ({"cfg": 3}, "trial-3")
