import pytest
from mneme.tuning.parameters import *
from mneme.tuning.search_space import SearchSpace


class SimpleSpace(SearchSpace):
    def dimensions(self):
        return {
            "size": IntRangeParam("size", 0, 10, step=2),
            "flag": BoolParam("flag"),
            "cat": CategoricalParam("cat", ["A", "B"]),
        }

    def derived(self, params):
        return {"double": params["size"] * 2}

    def constraints(self, params):
        # Reject size == 4
        return params["size"] != 4


def test_sample_random_single_param():
    dim = IntRangeParam("x", 0, 6, step=3)
    from mneme.tuning.random_sampler import sample_random_param
    values = {sample_random_param(dim) for _ in range(20)}
    assert values.issubset({0, 3, 6})
    assert len(values) >= 2  # Should see variation


def test_random_sampling_full_space():
    space = SimpleSpace()

    # Try a bunch of samples
    for _ in range(20):
        cfg = space.sample_random()
        assert "size" in cfg
        assert cfg["size"] != 4               # constraint
        assert cfg["double"] == cfg["size"] * 2
        assert cfg["flag"] in [True, False]
        assert cfg["cat"] in ["A", "B"]

def test_random_samples_iterator():
    space = SimpleSpace()
    gen = space.random_samples()

    # Take 5 samples
    samples = [next(gen) for _ in range(5)]

    assert len(samples) == 5
    for cfg in samples:
        assert "size" in cfg
        assert cfg["size"] != 4

