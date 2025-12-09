import pytest
from mneme.tuning.parameters import IntRangeParam
from mneme.tuning.search_space import SearchSpace
from mneme.tuning.tuner_node import TunerNode


class DummySpace(SearchSpace):
    def dimensions(self):
        return {"x": IntRangeParam("x", 0, 10)}


class DummyNode(TunerNode):
    def ask(self):
        return {"x": 5}

    def tell(self, params, feedback):
        self.last = (params, feedback)

    def evaluate(self, fixed_params):
        # trivial evaluation for test purposes
        return fixed_params["x"] * 2


def test_tuner_node_basic():
    node = DummyNode(DummySpace())

    cfg = node.ask()
    assert cfg == {"x": 5}

    node.tell(cfg, 1.23)
    assert node.last == (cfg, 1.23)

    out = node.evaluate({"x": 3})
    assert out == 6
