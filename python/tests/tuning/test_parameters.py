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
    assert p.name == "flag"
    assert p.choices == [True, False]


def test_categorical_param_requires_choices():
    try:
        CategoricalParam("x", [])
        assert False, "Expected ValueError for empty choices."
    except ValueError:
        pass


def test_int_range_param_bounds():
    p = IntRangeParam("x", 1, 4)
    assert p.low == 1
    assert p.high == 4

    try:
        IntRangeParam("y", 5, 2)
        assert False, "Expected ValueError when low > high."
    except ValueError:
        pass


def test_pipeline_param_with_pipelines():
    p = PipelineParam("pipe", pipelines=["O3", "O2"])
    assert p.name == "pipe"
    assert p.pipelines == ["O3", "O2"]
    assert p.generator is None


def test_pipeline_param_with_generator():
    def gen():
        return "some_pipeline"

    p = PipelineParam("pipe", generator=gen)
    assert p.pipelines is None
    assert p.generator is gen


def test_pipeline_param_requires_mode():
    try:
        PipelineParam("pipe")
        assert False, "Expected ValueError if neither pipelines nor generator is given."
    except ValueError:
        pass

