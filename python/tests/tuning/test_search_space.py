import pytest
from mneme.tuning.parameters import IntRangeParam, BoolParam
from mneme.tuning.search_space import SearchSpace


class MySpace(SearchSpace):
    def dimensions(self):
        return {
            "size": IntRangeParam("size", 1, 8, step=1),
            "flag": BoolParam("flag"),
        }

    def derived(self, params):
        return {"double": params["size"] * 2}

    def constraints(self, params):
        return params["size"] != 4


def test_dimensions_return_params():
    space = MySpace()
    dims = space.dimensions()
    assert "size" in dims and "flag" in dims
    assert isinstance(dims["size"], IntRangeParam)


def test_derived_parameters():
    space = MySpace()
    d = space.derived({"size": 3, "flag": True})
    assert d["double"] == 6


def test_constraints_filter():
    space = MySpace()
    assert space.constraints({"size": 3, "flag": True})
    assert not space.constraints({"size": 4, "flag": False})

