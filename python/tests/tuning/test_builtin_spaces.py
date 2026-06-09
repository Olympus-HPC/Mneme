import json

import pytest

import mneme.tuning.builtin_spaces as mod
from mneme.mneme_types import ExperimentConfiguration, dim3
from mneme.tuning.search_space import (
    BoolParam,
    CategoricalParam,
    FixedParam,
    IntRangeParam,
    RealRangeParam,
)


class FakeKernel:
    kernel_name = "KERNEL"
    shared_mem = 16
    block_dim = dim3(128, 2, 1)
    grid_dim = dim3(8, 4, 1)


def test_parse_range_accepts_two_or_three_part_specs():
    assert mod.parse_range("1:4") == (1, 4, 1)
    assert mod.parse_range("2:8:2") == (2, 8, 2)


@pytest.mark.parametrize("spec", ["1", "1:2:3:4", "1:4:0", "4:1"])
def test_parse_range_rejects_invalid_specs(spec):
    with pytest.raises(ValueError):
        mod.parse_range(spec)


def test_parse_key_value_args_decodes_json_values_and_raw_strings():
    parsed = mod.parse_key_value_args(
        ["alpha=1", 'name="kernel"', "flag=true", "raw=default<O3>"]
    )

    assert parsed == {
        "alpha": 1,
        "name": "kernel",
        "flag": True,
        "raw": "default<O3>",
    }


@pytest.mark.parametrize("value", ["missing_equals", "=empty"])
def test_parse_key_value_args_rejects_bad_values(value):
    with pytest.raises(ValueError):
        mod.parse_key_value_args([value])


def test_shape_round_trip_and_validation():
    shape = mod.dim3_to_shape(dim3(4, 2, 1))

    assert shape == "4x2x1"
    assert mod.dim3_to_shape(mod.shape_to_dim3(shape)) == shape

    with pytest.raises(ValueError, match="expected XxYxZ"):
        mod.shape_to_dim3("4x2")


def test_active_axes_modes_and_auto_detection():
    kernel = FakeKernel()

    assert mod._active_axes(kernel, "none") == (False, False, False)
    assert mod._active_axes(kernel, "x") == (True, False, False)
    assert mod._active_axes(kernel, "xy") == (True, True, False)
    assert mod._active_axes(kernel, "xyz") == (True, True, True)
    assert mod._active_axes(kernel, "auto") == (True, True, False)

    with pytest.raises(ValueError, match="Unknown launch_dim"):
        mod._active_axes(kernel, "bad")


def test_launch_budget_modes_and_validation():
    assert mod._launch_budget(2048, "conservative") == mod.MAX_THREADS_PER_BLOCK
    assert mod._launch_budget(128, "balanced") == 256
    assert mod._launch_budget(1, "aggressive") == mod.MAX_THREADS_PER_BLOCK

    with pytest.raises(ValueError, match="Unknown launch_safety"):
        mod._launch_budget(128, "bad")


def test_candidate_block_shapes_include_baseline_and_respect_launch_dim():
    kernel = FakeKernel()

    none_shapes = mod._candidate_block_shapes(kernel, "none", "balanced")
    x_shapes = mod._candidate_block_shapes(kernel, "x", "balanced")

    assert mod.dim3_to_shape(kernel.block_dim) in none_shapes
    assert none_shapes == ["128x2x1"]
    assert "128x2x1" in x_shapes
    assert all(int(shape.split("x")[1]) == 2 for shape in x_shapes)


def test_pipeline_choices_from_fixed_explicit_file_and_presets(tmp_path):
    pipeline_file = tmp_path / "pipelines.txt"
    pipeline_file.write_text("# comment\nfile-pass\n\nother-pass\n")

    assert mod._pipeline_choices("standard", None, "fixed-pass", None) == (
        None,
        "fixed-pass",
    )
    assert mod._pipeline_choices("standard", ["a", "b", "a"], None, None) == (
        ["a", "b"],
        "a",
    )
    assert mod._pipeline_choices("standard", None, None, str(pipeline_file)) == (
        ["file-pass", "other-pass"],
        "file-pass",
    )
    assert mod._pipeline_choices("quick", None, None, None) == (
        mod.QUICK_PIPELINES,
        mod.QUICK_PIPELINES[0],
    )
    assert mod._pipeline_choices("launch", None, None, None) == (None, "default<O3>")
    assert mod._pipeline_choices("standard", None, None, None) == (
        mod.DEFAULT_PIPELINES,
        mod.DEFAULT_PIPELINES[0],
    )


def test_space_param_modes_and_param_descriptions():
    assert mod.param_to_description(mod._space_param("x", "fixed", True)) == {
        "type": "fixed",
        "value": True,
    }
    assert mod.param_to_description(mod._space_param("x", "on")) == {
        "type": "fixed",
        "value": True,
    }
    assert mod.param_to_description(mod._space_param("x", "off")) == {
        "type": "fixed",
        "value": False,
    }
    assert mod.param_to_description(mod._space_param("x", "on-off")) == {
        "type": "bool",
        "choices": [True, False],
    }
    assert mod.param_to_description(CategoricalParam("cat", ["a"])) == {
        "type": "categorical",
        "choices": ["a"],
    }
    assert mod.param_to_description(IntRangeParam("i", 1, 3, 2)) == {
        "type": "int_range",
        "low": 1,
        "high": 3,
        "step": 2,
    }
    assert mod.param_to_description(RealRangeParam("r", 0.0, 1.0)) == {
        "type": "real_range",
        "low": 0.0,
        "high": 1.0,
    }

    with pytest.raises(ValueError, match="Unknown x mode"):
        mod._space_param("x", "sometimes")


def test_describe_search_space_includes_metadata_and_preset():
    class Space:
        metadata = {"source": "test"}

        def dimensions(self):
            return {"passes": FixedParam("passes", "default<O3>")}

    description = mod.describe_search_space(Space(), preset="quick")

    assert description == {
        "type": "builtin",
        "preset": "quick",
        "metadata": {"source": "test"},
        "dimensions": {"passes": {"type": "fixed", "value": "default<O3>"}},
    }
    assert mod.describe_search_space(Space(), custom=True)["type"] == "custom"


def test_builtin_space_quick_preset_derives_baseline_and_candidate_config():
    space = mod.BuiltinTuneSearchSpace(FakeKernel(), preset="quick", launch_dim="x")
    dims = space.dimensions()

    assert isinstance(dims["block_shape"], CategoricalParam)
    assert isinstance(dims["passes"], CategoricalParam)
    assert isinstance(dims["codegen_opt"], IntRangeParam)
    assert space.metadata["baseline_passes"] == "default<O3>"
    assert space.metadata["baseline_codegen_opt"] == 3

    baseline = space.baseline()
    assert baseline.grid.to_dict() == FakeKernel.grid_dim.to_dict()
    assert baseline.block.to_dict() == FakeKernel.block_dim.to_dict()
    assert baseline.shared_mem == 16

    config = space.derived(
        {
            "block_shape": "64x2x1",
            "passes": "default<O2>",
            "codegen_opt": 2,
            "specialize": True,
            "specialize_dims": True,
            "set_launch_bounds": True,
            "min_blocks_per_sm": 2,
        }
    )

    assert config.grid.to_dict() == {"x": 16, "y": 4, "z": 1}
    assert config.block.to_dict() == {"x": 64, "y": 2, "z": 1}
    assert config.max_threads == 128
    assert config.min_blocks_per_sm == 2
    assert config.passes == "default<O2>"
    assert space.constraints(config) is True


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        ("recorded", 256),
        ("hardware", mod.MAX_THREADS_PER_BLOCK),
        ("block-threads", 64),
    ],
)
def test_builtin_space_max_threads_policies(policy, expected):
    space = mod.BuiltinTuneSearchSpace(
        FakeKernel(),
        preset="standard",
        launch_dim="none",
        launch_bounds_space="on",
        max_threads_policy=policy,
    )

    config = space.derived(
        {
            "block_shape": "64x1x1",
            "passes": "default<O3>",
            "codegen_opt": 3,
            "set_launch_bounds": True,
        }
    )

    assert config.max_threads == expected


def test_builtin_space_rejects_bad_constructor_options():
    with pytest.raises(ValueError, match="Unknown tuning preset"):
        mod.BuiltinTuneSearchSpace(FakeKernel(), preset="bad")

    with pytest.raises(ValueError, match="Unknown max_threads_policy"):
        mod.BuiltinTuneSearchSpace(FakeKernel(), max_threads_policy="bad")


def test_builtin_space_compiler_and_launch_presets_fix_expected_dimensions():
    compiler = mod.BuiltinTuneSearchSpace(FakeKernel(), preset="compiler")
    launch = mod.BuiltinTuneSearchSpace(FakeKernel(), preset="launch")

    assert isinstance(compiler.dimensions()["block_shape"], FixedParam)
    assert isinstance(launch.dimensions()["passes"], FixedParam)
    assert isinstance(launch.dimensions()["codegen_opt"], FixedParam)


def test_builtin_space_constraints_reject_bad_configurations():
    space = mod.BuiltinTuneSearchSpace(FakeKernel(), preset="standard", launch_dim="none")

    assert space.constraints(
        ExperimentConfiguration(block=dim3(2048, 1, 1), grid=dim3(1, 1, 1))
    ) is False
    assert space.constraints(
        ExperimentConfiguration(block=dim3(1, 1, 1), grid=dim3(0, 1, 1))
    ) is False
    assert space.constraints(
        ExperimentConfiguration(
            block=dim3(128, 1, 1),
            grid=dim3(1, 1, 1),
            set_launch_bounds=True,
            max_threads=64,
        )
    ) is False


def test_load_custom_space_from_file_and_module(tmp_path, monkeypatch):
    module_path = tmp_path / "custom_space.py"
    module_path.write_text(
        "\n".join(
            [
                "class CustomSpace:",
                "    def __init__(self, kernel, scale=1):",
                "        self.kernel = kernel",
                "        self.scale = scale",
            ]
        )
    )

    loaded = mod.load_custom_space(f"{module_path}:CustomSpace", FakeKernel(), {"scale": 3})
    assert loaded.kernel is not None
    assert loaded.scale == 3

    class ModuleSpace:
        def __init__(self, kernel, name="default"):
            self.kernel = kernel
            self.name = name

    fake_module = type("FakeModule", (), {"ModuleSpace": ModuleSpace})
    monkeypatch.setattr(mod.importlib, "import_module", lambda name: fake_module)

    loaded = mod.load_custom_space("fake.module:ModuleSpace", FakeKernel(), {"name": "custom"})
    assert loaded.name == "custom"

    with pytest.raises(ValueError, match="MODULE_OR_PATH:CLASS"):
        mod.load_custom_space("missing-class", FakeKernel())


def test_finite_param_values_and_iter_finite_params():
    assert mod.finite_param_values(FixedParam("x", 1)) == [1]
    assert mod.finite_param_values(BoolParam("b")) == [True, False]
    assert mod.finite_param_values(CategoricalParam("c", ["a", "b"])) == ["a", "b"]
    assert mod.finite_param_values(IntRangeParam("i", 1, 3)) == [1, 2, 3]

    with pytest.raises(ValueError, match="not finite"):
        mod.finite_param_values(RealRangeParam("r", 0.0, 1.0))

    class Space:
        def dimensions(self):
            return {
                "x": FixedParam("x", 1),
                "y": CategoricalParam("y", ["a", "b"]),
            }

    assert list(mod.iter_finite_params(Space())) == [
        {"x": 1, "y": "a"},
        {"x": 1, "y": "b"},
    ]
