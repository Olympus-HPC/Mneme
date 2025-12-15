import pytest
import json
import copy
from mneme.mneme_types import (
    dim3,
    ExperimentConfiguration,
    ExperimentResult,
    _to_serializable,
)


def test_dim3_to_dict_and_from_dict_roundtrip():
    d = dim3(7, 8, 9)
    as_dict = d.to_dict()
    assert as_dict == {"x": 7, "y": 8, "z": 9}

    d2 = dim3.from_dict(as_dict)
    assert isinstance(d2, dim3)
    assert (d2.x, d2.y, d2.z) == (7, 8, 9)


def test_to_serializable_dim3():
    d = dim3(1, 2, 3)
    assert _to_serializable(d) == {"x": 1, "y": 2, "z": 3}


def test_to_serializable_sorts_dict_keys_deterministically():
    # Key property: dict keys are sorted in the output mapping.
    obj = {"b": 2, "a": 1, "c": 3}
    ser = _to_serializable(obj)
    assert list(ser.keys()) == ["a", "b", "c"]
    assert ser == {"a": 1, "b": 2, "c": 3}


def test_to_serializable_handles_nested_containers_and_dataclasses():
    cfg = ExperimentConfiguration(
        grid=dim3(2, 1, 1),
        block=dim3(16, 2, 1),
        shared_mem=123,
        specialize=True,
        set_launch_bounds=True,
        max_threads=256,
        min_blocks_per_sm=2,
        specialize_dims=True,
        passes="default<O2>",
        codegen_opt=2,
        codegen_method="serial",
        prune=True,
        internalize=True,
    )

    ser = _to_serializable(
        {
            "cfg": cfg,
            "list": [dim3(1, 1, 1), {"z": 3, "y": 2, "x": 1}],
            "tuple": (1, 2, dim3(9, 9, 9)),
        }
    )

    assert ser["cfg"]["grid"] == {"x": 2, "y": 1, "z": 1}
    assert ser["cfg"]["block"] == {"x": 16, "y": 2, "z": 1}
    # Ensure inner dict got sorted
    assert list(ser["list"][1].keys()) == ["x", "y", "z"]
    assert ser["tuple"] == [1, 2, {"x": 9, "y": 9, "z": 9}]


def test_experiment_configuration_to_dict_shape_and_types():
    cfg = ExperimentConfiguration(grid=dim3(4, 5, 6), block=dim3(7, 8, 9))
    d = cfg.to_dict()

    # Must be JSON-serializable (no dim3 objects left)
    json.dumps(d)

    assert d["grid"] == {"x": 4, "y": 5, "z": 6}
    assert d["block"] == {"x": 7, "y": 8, "z": 9}
    assert isinstance(d["shared_mem"], int)
    assert isinstance(d["specialize"], bool)


def test_experiment_configuration_from_dict_accepts_dim3_or_dict():
    base = ExperimentConfiguration(grid=dim3(10, 1, 1), block=dim3(32, 1, 1))
    d = base.to_dict()

    # Case 1: grid/block encoded as dicts
    cfg1 = ExperimentConfiguration.from_dict(d)
    assert (cfg1.grid.x, cfg1.grid.y, cfg1.grid.z) == (10, 1, 1)
    assert (cfg1.block.x, cfg1.block.y, cfg1.block.z) == (32, 1, 1)

    # Case 2: grid/block as actual dim3 objects in the input dict
    d2 = dict(d)
    d2["grid"] = dim3(11, 2, 3)
    d2["block"] = dim3(64, 1, 1)
    cfg2 = ExperimentConfiguration.from_dict(d2)
    assert (cfg2.grid.x, cfg2.grid.y, cfg2.grid.z) == (11, 2, 3)
    assert (cfg2.block.x, cfg2.block.y, cfg2.block.z) == (64, 1, 1)


def test_experiment_configuration_is_valid_launch_bounds_constraint():
    # Invalid: set_launch_bounds=True but max_threads smaller than block volume
    cfg = ExperimentConfiguration(
        block=dim3(16, 16, 1),  # 256 threads
        set_launch_bounds=True,
        max_threads=128,
    )
    assert cfg.is_valid() is False

    # Valid: max_threads >= block volume
    cfg2 = ExperimentConfiguration(
        block=dim3(16, 16, 1),
        set_launch_bounds=True,
        max_threads=256,
    )
    assert cfg2.is_valid() is True

    # Valid: when set_launch_bounds is False, constraint is ignored
    cfg3 = ExperimentConfiguration(
        block=dim3(32, 32, 1),
        set_launch_bounds=False,
        max_threads=1,  # meaningless in this mode
    )
    assert cfg3.is_valid() is True


def test_experiment_configuration_ground_zeroes_unused_fields_only():
    cfg = ExperimentConfiguration(
        set_launch_bounds=False,
        max_threads=999,
        min_blocks_per_sm=7,
    )
    cfg.ground()
    assert cfg.max_threads == 0
    assert cfg.min_blocks_per_sm == 0

    cfg2 = ExperimentConfiguration(
        set_launch_bounds=True,
        max_threads=888,
        min_blocks_per_sm=6,
    )
    cfg2.ground()
    assert cfg2.max_threads == 888
    assert cfg2.min_blocks_per_sm == 6


def test_experiment_configuration_hash_stable_for_equivalent_configs():
    cfg = ExperimentConfiguration(
        grid=dim3(4096, 1, 1),
        block=dim3(256, 1, 1),
        set_launch_bounds=False,
        max_threads=123,  # should be ignored after grounding
        min_blocks_per_sm=456,  # should be ignored after grounding
    )

    cfg_a = copy.deepcopy(cfg)
    cfg_b = copy.deepcopy(cfg)

    # If you *intend* hashes to be stable across "unused" fields, caller must ground().
    cfg_a.ground()
    cfg_b.ground()

    assert cfg_a.hash() == cfg_b.hash()

    # Mutating a used field should change the hash
    cfg_b.shared_mem = cfg_b.shared_mem + 1
    assert cfg_a.hash() != cfg_b.hash()


def test_experiment_configuration_hash_deterministic_across_dict_key_order():
    cfg = ExperimentConfiguration(grid=dim3(1, 2, 3), block=dim3(4, 5, 6))
    h1 = cfg.hash()

    # Build a dict with the same content but shuffled insertion order,
    # reconstruct config, and ensure hash matches.
    d = cfg.to_dict()
    shuffled = {}
    for k in ["block", "grid"] + [kk for kk in d.keys() if kk not in ("grid", "block")]:
        shuffled[k] = d[k]

    cfg2 = ExperimentConfiguration.from_dict(shuffled)
    h2 = cfg2.hash()
    assert h1 == h2


def test_experiment_result_to_dict_from_dict_roundtrip_and_defaults():
    r = ExperimentResult(
        preprocess_ir_time=1.0,
        opt_time=2.0,
        codegen_time=3.0,
        obj_size=1234,
        exec_time=[10, 11, 12],
        verified=True,
        executed=True,
        failed=False,
        start_time="2025-01-01T00:00:00",
        end_time="2025-01-01T00:00:01",
        gpu_id=0,
        const_mem_usage=1,
        local_mem_usage=2,
        reg_usage=3,
        error="",
    )
    d = r.to_dict()
    json.dumps(d)  # serializable

    r2 = ExperimentResult.from_dict(d)
    assert r2 == r

    # Ensure exec_time default_factory gives independent lists
    a = ExperimentResult()
    b = ExperimentResult()
    a.exec_time.append(1)
    assert b.exec_time == []
