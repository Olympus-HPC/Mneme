import pytest

from mneme.tuning.parameters import (
    BaseParam,
    FixedParam,
    BoolParam,
    CategoricalParam,
    IntRangeParam,
    PipelineParam,
)


def test_fixed_param_basic():
    p = FixedParam("x", 42)
    assert p.name == "x"
    assert p.value == 42
    assert isinstance(p, BaseParam)


def test_bool_param_basic():
    p = BoolParam("flag")
    assert p.choices == [True, False]


def test_categorical_param_requires_values():
    with pytest.raises(ValueError):
        CategoricalParam("cat", [])


def test_int_range_param():
    p = IntRangeParam("r", 1, 10)
    assert p.low == 1
    assert p.high == 10
    assert p.step == 1


def test_int_range_param_custom_step():
    p = IntRangeParam("r", 0, 8, step=2)
    assert p.step == 2


def test_int_range_param_step_invalid():
    with pytest.raises(ValueError):
        IntRangeParam("r", 0, 10, step=0)
    with pytest.raises(ValueError):
        IntRangeParam("r", 0, 10, step=-3)
    with pytest.raises(ValueError):
        IntRangeParam("r", 10, 0)  # low > high


def test_pipeline_param_with_list():
    p = PipelineParam("pipeline", pipelines=["O1", "O2"])
    assert p.pipelines == ["O1", "O2"]
    assert p.generator is None


def test_pipeline_param_with_generator():
    def gen():
        return "random_pipeline"

    p = PipelineParam("pipeline", generator=gen)
    assert p.pipelines is None
    assert p.generator is gen


def test_pipeline_param_empty_list():
    with pytest.raises(ValueError):
        PipelineParam("pipeline", pipelines=[])


def test_pipeline_param_requires_one_mode():
    # neither pipelines nor generator
    with pytest.raises(ValueError):
        PipelineParam("pipeline")


def test_pipeline_param_rejects_both():
    def gen():
        return "x"

    with pytest.raises(ValueError):
        PipelineParam("pipeline", pipelines=["O3"], generator=gen)

