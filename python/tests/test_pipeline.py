import pytest

from mneme.pipeline import (
    split_top_level,
    flatten_passes,
    parse_llvm_pass_options,
    PipelineManager,
    AbstractPass,
)

# ------------------------------------------------------------
# split_top_level
# ------------------------------------------------------------

def test_split_top_level_simple():
    s = "a,b,c"
    assert split_top_level(s) == ["a", "b", "c"]


def test_split_top_level_nested():
    s = "a(b,c), d, e(f(g,h),i)"
    assert split_top_level(s) == [
        "a(b,c)",
        "d",
        "e(f(g,h),i)",
    ]


# ------------------------------------------------------------
# flatten_passes
# ------------------------------------------------------------

def test_flatten_passes_basic():
    s = "a(b,c),d"
    out = flatten_passes(s)
    assert out == ["b", "c", "d"]


def test_flatten_passes_deep():
    s = "aa(bb(cc,dd),ee),ff"
    out = flatten_passes(s)
    assert out == ["cc", "dd", "ee", "ff"]


# ------------------------------------------------------------
# parse_llvm_pass_options
# ------------------------------------------------------------

def test_parse_llvm_pass_options_toggle():
    name, opts = parse_llvm_pass_options("myPass<foo;no-foo>")
    assert name == "myPass"
    assert "foo" in opts
    assert opts["foo"].is_toggle()


def test_parse_llvm_pass_options_range():
    name, opts = parse_llvm_pass_options("myPass<threshold=N>")
    assert "threshold" in opts
    assert opts["threshold"].is_range()


def test_parse_llvm_pass_options_setting_list():
    name, opts = parse_llvm_pass_options("myPass<optA;optB;optC>")
    assert name == "myPass"
    assert "myPass" in opts
    assert opts["myPass"].is_setting()
    assert opts["myPass"].options == ["optA", "optB", "optC"]


# ------------------------------------------------------------
# PipelineManager basic construction
# ------------------------------------------------------------

def test_pipeline_manager_constructs():
    pm = PipelineManager()
    assert isinstance(pm._passes, list)
    assert len(pm._passes) > 10


# ------------------------------------------------------------
# from_string
# ------------------------------------------------------------

def test_from_string_simple():
    pm = PipelineManager()
    p = pm.from_string("mem2reg")   # a known-good pass
    assert len(p) == 1
    assert isinstance(p[0], AbstractPass.ConcretePass)
    assert p[0]._apass.pass_name == "mem2reg"


def test_from_string_with_options():
    pm = PipelineManager()
    p = pm.from_string("mem2reg<optA>")  # mem2reg always exists
    assert p[0]._apass.pass_name == "mem2reg"
    assert p[0].options == ["optA"]


# ------------------------------------------------------------
# to_string roundtrip structural
# ------------------------------------------------------------

def test_pipeline_roundtrip():
    pm = PipelineManager()
    parsed = pm.from_string("mem2reg")
    s = pm.to_string(parsed)
    assert "mem2reg" in s


def test_pipeline_roundtrip_generic_o3_passes_with_options():
    pm = PipelineManager()
    pipeline = (
        "function<eager-inv>("
        "instcombine<max-iterations=1;no-use-loop-info;no-verify-fixpoint>,"
        "loop-unroll<O3>,"
        "mldst-motion<no-split-footer-bb>,"
        "speculative-execution<only-if-divergent-target>,"
        "correlated-propagation)"
    )
    parsed = pm.from_string(pipeline)
    s = pm.to_string(parsed)

    assert len(parsed) == 5
    assert "instcombine<max-iterations=1;no-use-loop-info;no-verify-fixpoint>" in s
    assert "loop-unroll<O3>" in s
    assert "mldst-motion<no-split-footer-bb>" in s
    assert "speculative-execution<only-if-divergent-target>" in s
    assert "correlated-propagation" in s


def test_pipeline_roundtrip_amdgpu_middle_end_passes():
    pm = PipelineManager()
    pipeline = (
        "amdgpu-printf-runtime-binding,"
        "amdgpu-unify-metadata,"
        "function("
        "amdgpu-promote-kernel-arguments,"
        "amdgpu-lower-kernel-attributes,"
        "amdgpu-promote-alloca-to-vector,"
        "amdgpu-usenative,"
        "amdgpu-simplifylib)"
    )
    parsed = pm.from_string(pipeline)
    s = pm.to_string(parsed)

    assert len(parsed) == 7
    assert "amdgpu-printf-runtime-binding" in s
    assert "amdgpu-unify-metadata" in s
    assert "function<eager-inv>(" in s
    assert "amdgpu-promote-kernel-arguments" in s
    assert "amdgpu-lower-kernel-attributes" in s
    assert "amdgpu-promote-alloca-to-vector" in s
    assert "amdgpu-usenative" in s
    assert "amdgpu-simplifylib" in s


# ------------------------------------------------------------
# Random pipeline generation
# ------------------------------------------------------------

def test_generate_deterministic():
    pm = PipelineManager()
    p1 = pm.generate(samples=2, mean_passes=3, std=1, seed=123)
    p2 = pm.generate(samples=2, mean_passes=3, std=1, seed=123)
    assert p1[0][0].__str__() == p2[0][0].__str__()


def test_generate_length_clamped_min():
    pm = PipelineManager()
    p = pm._generate_random_pipeline(mean_passes=0.1, std_passes=0.1)
    assert len(p) == 1


def test_generate_length_clamped_max():
    pm = PipelineManager()
    max_len = len(pm._passes)
    p = pm._generate_random_pipeline(mean_passes=max_len * 10, std_passes=1)
    assert len(p) == max_len
