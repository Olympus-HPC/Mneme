# test_convert_to_json.py
import json
from pathlib import Path
import types
import pytest

from mneme import convert as m


class DummyFunc:
    def __init__(self, root, src, loc):
        self._root = root
        self._src = src
        self._loc = loc

    def get_function_location(self):
        return self._root, self._src, self._loc


class DummyMod:
    def __init__(self, kernel_name_to_func):
        self._map = kernel_name_to_func

    def get_function(self, name):
        if name not in self._map:
            raise NameError("not found")
        return self._map[name]


def _mk_recorded_execution(tmp_path, llvm_files, demangled_name="my::kernel()"):
    # RecordedExecution is only used by attribute access, so a SimpleNamespace is enough.
    return types.SimpleNamespace(
        llvm_files=[str(p) for p in llvm_files],
        demangled_name=demangled_name,
    )


def _mk_kernel_descr(kernel_name="K", specializations=None):
    if specializations is None:
        specializations = []
    return types.SimpleNamespace(
        kernel_name=kernel_name,
        specializations=specializations,
    )


def _mk_config():
    # ExperimentConfiguration is only used by attribute access.
    return types.SimpleNamespace(
        set_launch_bounds=True,
        specialize_dims=[1, 2, 3],
        passes=["p1", "p2"],
        codegen_opt=2,
        max_threads=256,
        min_blocks_per_sm=2,
    )


def test_attribute_line_basic():
    assert m._attribute_line([16, 17]) == '__attribute__((annotate("jit", 16,17)))'


def test_attribute_line_empty_params():
    # Current behavior: will produce annotate("jit", ) which is arguably odd,
    # but test documents the existing behavior.
    assert m._attribute_line([]) == '__attribute__((annotate("jit", )))'


def test_convert_to_json_writes_expected_json_no_location(
    monkeypatch, tmp_path, capsys
):
    # Setup: one llvm file, kernel found, but location returns (None, None, -1)
    llvm = tmp_path / "a.bc"
    llvm.write_bytes(b"dummy")

    rec = _mk_recorded_execution(tmp_path, [llvm], demangled_name="demangled::name")
    kd = _mk_kernel_descr(kernel_name="KERNEL", specializations=[True, False, True])
    cfg = _mk_config()

    # Mock module.parse_bitcode -> returns a mod that has get_function("KERNEL")
    func = DummyFunc(root=None, src=None, loc=-1)
    mod = DummyMod({"KERNEL": func})

    monkeypatch.setattr(m.module, "parse_bitcode", lambda ir: mod)

    out_json = tmp_path / "out.json"
    m.convert_to_json(str(out_json), rec, kd, cfg)

    # JSON file content
    data = json.loads(out_json.read_text())
    assert "KERNEL" in data
    res = data["KERNEL"]
    assert res["LaunchBounds"] == cfg.set_launch_bounds
    assert res["SpecializeDims"] == cfg.specialize_dims
    assert res["SpecializeDimsRange"] == cfg.specialize_dims  # matches current code
    assert res["OptLevel"] == 1
    assert res["Pipeline"] == cfg.passes
    assert res["CodeGenOptLevel"] == cfg.codegen_opt
    assert res["TunedMaxThreads"] == cfg.max_threads
    assert res["MinBlocksPerSM"] == cfg.min_blocks_per_sm

    # Stdout: attribute line should include indices where specializations are True (1-based)
    printed = capsys.readouterr().out
    assert '__attribute__((annotate("jit", 1,3)))' in printed
    assert "The kernel is defined in:" not in printed  # because src/root None


def test_convert_to_json_prints_location_when_available(monkeypatch, tmp_path, capsys):
    llvm = tmp_path / "a.bc"
    llvm.write_bytes(b"dummy")

    rec = _mk_recorded_execution(tmp_path, [llvm], demangled_name="demangled::name")
    kd = _mk_kernel_descr(kernel_name="KERNEL", specializations=[False, True])
    cfg = _mk_config()

    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    src = "src/kernel.cu"
    (root / src).write_text("// kernel")

    func = DummyFunc(root=str(root), src=src, loc=123)
    mod = DummyMod({"KERNEL": func})
    monkeypatch.setattr(m.module, "parse_bitcode", lambda ir: mod)

    out_json = tmp_path / "out.json"
    m.convert_to_json(str(out_json), rec, kd, cfg)

    printed = capsys.readouterr().out
    # attribute should pick specialization index 2 (1-based)
    assert '__attribute__((annotate("jit", 2)))' in printed

    # location line should include resolved path and :loc
    resolved = str((Path(root) / Path(src)).resolve())
    assert resolved in printed
    assert f"{resolved}:123" in printed


def test_convert_to_json_skips_files_where_kernel_missing(
    monkeypatch, tmp_path, capsys
):
    # Two llvm files: first missing kernel -> should log debug and continue, second has it.
    llvm1 = tmp_path / "a.bc"
    llvm2 = tmp_path / "b.bc"
    llvm1.write_bytes(b"dummy1")
    llvm2.write_bytes(b"dummy2")

    rec = _mk_recorded_execution(
        tmp_path, [llvm1, llvm2], demangled_name="demangled::name"
    )
    kd = _mk_kernel_descr(kernel_name="KERNEL", specializations=[True])
    cfg = _mk_config()

    mod_missing = DummyMod({})
    func = DummyFunc(root=None, src=None, loc=-1)
    mod_found = DummyMod({"KERNEL": func})

    calls = {"n": 0}

    def parse_bitcode(_ir):
        calls["n"] += 1
        return mod_missing if calls["n"] == 1 else mod_found

    monkeypatch.setattr(m.module, "parse_bitcode", parse_bitcode)

    # Capture logger.debug without depending on the real logger implementation
    debug_msgs = []
    monkeypatch.setattr(m.logger, "debug", lambda msg: debug_msgs.append(msg))

    out_json = tmp_path / "out.json"
    m.convert_to_json(str(out_json), rec, kd, cfg)

    assert any("Could not find function" in s for s in debug_msgs)
    # It should have tried both files
    assert calls["n"] == 2

    printed = capsys.readouterr().out
    assert '__attribute__((annotate("jit", 1)))' in printed


def test_convert_to_json_always_writes_json_even_if_kernel_never_found(
    monkeypatch, tmp_path, capsys
):
    # No file contains the kernel; function location remains default (None, None, -1)
    llvm1 = tmp_path / "a.bc"
    llvm1.write_bytes(b"dummy")

    rec = _mk_recorded_execution(tmp_path, [llvm1], demangled_name="demangled::name")
    kd = _mk_kernel_descr(kernel_name="KERNEL", specializations=[True, True])
    cfg = _mk_config()

    mod_missing = DummyMod({})
    monkeypatch.setattr(m.module, "parse_bitcode", lambda ir: mod_missing)
    monkeypatch.setattr(m.logger, "debug", lambda msg: None)

    out_json = tmp_path / "out.json"
    m.convert_to_json(str(out_json), rec, kd, cfg)

    data = json.loads(out_json.read_text())
    assert "KERNEL" in data  # JSON is still produced

    printed = capsys.readouterr().out
    assert '__attribute__((annotate("jit", 1,2)))' in printed
    assert "The kernel is defined in:" not in printed


def test_export_proteus_tuned_kernel_writes_usage_file(monkeypatch, tmp_path, capsys):
    """ Test that export_proteus_tuned_kernel writes a usage file with the expected content, including the attribute line and kernel source location if available."""
    llvm = tmp_path / "a.bc"
    llvm.write_bytes(b"dummy")

    rec = _mk_recorded_execution(tmp_path, [llvm], demangled_name="demangled::name")
    kd = _mk_kernel_descr(kernel_name="KERNEL", specializations=[True, False, True])
    cfg = _mk_config()

    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    src = "src/kernel.cu"
    (root / src).write_text("// kernel")

    func = DummyFunc(root=str(root), src=src, loc=123)
    monkeypatch.setattr(m.module, "parse_bitcode", lambda ir: DummyMod({"KERNEL": func}))

    out_json = tmp_path / "out.json"
    usage = tmp_path / "usage.txt"
    m.export_proteus_tuned_kernel(str(out_json), rec, kd, cfg, usage_filename=str(usage))

    assert capsys.readouterr().out == ""
    usage_text = usage.read_text()
    assert '__attribute__((annotate("jit", 1,3)))' in usage_text
    assert "Kernel source location:" in usage_text
    assert "export PROTEUS_TUNED_KERNELS=" in usage_text
