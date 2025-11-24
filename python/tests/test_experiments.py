import pytest
import numpy as np
from mneme.experiment import _analyze_measurements, Experiment


# -----------------------------------------------------------------------------
# analyze_measurements
# -----------------------------------------------------------------------------

def test_analyze_measurements_basic():
    times = [10, 12, 14, 16, 18, 20]
    std, avg, median, R, Rp, iqr, iqrp, q25, q75 = _analyze_measurements(times)

    arr = np.array(times, float)

    assert pytest.approx(std) == np.std(arr, ddof=1)
    assert pytest.approx(avg) == np.mean(arr)
    assert pytest.approx(median) == np.median(arr)
    assert pytest.approx(q25) == np.percentile(arr, 25)
    assert pytest.approx(q75) == np.percentile(arr, 75)
    assert pytest.approx(iqr) == (q75 - q25)
    assert pytest.approx(iqrp) == (iqr / median)

    # R and R% defined only if at least 4 samples
    sorted_arr = np.sort(arr)
    assert pytest.approx(R) == sorted_arr[-2] / sorted_arr[1]
    assert pytest.approx(Rp) == (R - 1.0) * 100.0


def test_analyze_measurements_small_list():
    # N < 4 → R and Rp become None
    times = [1.0, 2.0, 3.0]
    std, avg, med, R, Rp, *_ = _analyze_measurements(times)

    assert std >= 0
    assert avg == pytest.approx(2.0)
    assert med == pytest.approx(2.0)
    assert R is None
    assert Rp is None


# -----------------------------------------------------------------------------
# Experiment construction helpers
# -----------------------------------------------------------------------------

def make_exp(**overrides):
    """Helper factory with defaults."""
    base = dict(
        specialize=False,
        max_threads=128,
        min_blocks_per_sm=0,
        specialize_dims=False,
        passes="default<O3>",
        prune=False,
        internalize=False,
        codegen_opt=3,
        codegen_method="llc",
        device_arch="gfx90a",
    )
    base.update(overrides)
    return Experiment(**base)


# -----------------------------------------------------------------------------
# Boolean coercion tests
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("true", True), ("True", True), ("TRUE", True),
    ("false", False), ("False", False), ("no", False),
    (1, True), (0, False),
])
def test_specialize_coercion(value, expected):
    e = make_exp(specialize=value)
    assert e.specialize is expected


@pytest.mark.parametrize("value,expected", [
    ("true", True), ("false", False), (1, True), (0, False)
])
def test_prune_coercion(value, expected):
    e = make_exp(prune=value)
    assert e.prune is expected


@pytest.mark.parametrize("value,expected", [
    ("true", True), ("false", False), (1, True), (0, False)
])
def test_internalize_coercion(value, expected):
    e = make_exp(internalize=value)
    assert e.internalize is expected


# -----------------------------------------------------------------------------
# Integer coercion
# -----------------------------------------------------------------------------

def test_int_fields():
    e = make_exp(max_threads="256", min_blocks_per_sm="3", codegen_opt="2")
    assert e.max_threads == 256
    assert e.min_blocks_per_sm == 3
    assert e.codegen_opt == 2


def test_int_fields_none():
    e = make_exp(max_threads=None, min_blocks_per_sm=None, codegen_opt=None)
    assert e.max_threads is None
    assert e.min_blocks_per_sm is None
    assert e.codegen_opt is None


# -----------------------------------------------------------------------------
# exec_time setter
# -----------------------------------------------------------------------------

def test_exec_time_requires_list():
    e = make_exp()
    with pytest.raises(TypeError):
        e.exec_time = 5  # not a list


def test_exec_time_sets_measurements_correctly():
    e = make_exp()
    times = [1, 2, 3, 40, 50, 60]    # first 2 dropped → analyze [3,40,50,60]

    e.exec_time = times
    arr = np.array(times[2:], float)

    assert e._exec_time_std == pytest.approx(np.std(arr, ddof=1))
    assert e._exec_time_avg == pytest.approx(np.mean(arr))
    assert e._exec_time_median == pytest.approx(np.median(arr))


# -----------------------------------------------------------------------------
# hash() determinism
# -----------------------------------------------------------------------------

def test_hash_is_deterministic():
    e1 = make_exp()
    e2 = make_exp()
    assert e1.hash() == e2.hash()

    # Mutate something → hash changes
    e2.max_threads = 999
    assert e1.hash() != e2.hash()


# -----------------------------------------------------------------------------
# to_dict / from_dict
# -----------------------------------------------------------------------------

def test_to_dict_contains_all_fields():
    e = make_exp()
    d = e.to_dict()

    required = [
        "specialize", "max_threads", "min_blocks_per_sm",
        "specialize_dims", "passes", "prune", "internalize",
        "codegen_opt", "codegen_method", "device_arch",
        "failed", "start_time", "end_time", "commit_id",
        "start_id", "gpu_id", "opt_time", "codegen_time",
        "verified", "obj_size", "exec_time_std",
        "exec_time_avg", "exec_time_median", "exec_time_r",
        "exec_time_rp", "exec_time_iqr", "exec_time_iqrp",
        "exec_time_q25", "exec_time_q75", "executed",
        "hash", "reg_usage", "const_mem", "local_mem",
    ]

    for k in required:
        assert k in d


def test_from_dict_roundtrip():
    e1 = make_exp()
    d = e1.to_dict()
    e2 = Experiment.from_dict(**d)

    assert e1.hash() == e2.hash()
    assert e1.to_dict() == e2.to_dict()


# -----------------------------------------------------------------------------
# simple property tests
# -----------------------------------------------------------------------------

def test_property_setters():
    e = make_exp()

    e.const_mem = 123
    e.local_mem = 456
    e.reg_usage = 789
    e.gpu_id = 3
    e.start_id = 9
    e.commit_id = 7
    e.start_time = 100
    e.end_time = 200
    e.obj_size = 512
    e.verified = True
    e.codegen_time = 2.5
    e.opt_time = 1.5
    e.executed = True
    e.failed = False

    assert e.const_mem == 123
    assert e.local_mem == 456
    assert e.reg_usage == 789
    assert e.gpu_id == 3
    assert e.start_id == 9
    assert e.commit_id == 7
    assert e.start_time == 100
    assert e.end_time == 200
    assert e.obj_size == 512
    assert e.verified is True
    assert e.codegen_time == 2.5
    assert e.opt_time == 1.5
    assert e.executed is True
    assert e.failed is False

