import json
import tempfile
from unittest.mock import patch, MagicMock
import pytest

from mneme.recorded_execution import (
    MemStateRef,
    RecordedExecution,
    SnapshotType,
)
from mneme.mneme_types import dim3


# ======================================================================
#                           MemStateRef Tests
# ======================================================================

def test_memstate_constructor_checks_file_exists():
    with patch("mneme.recorded_execution.Path.exists", return_value=False):
        with pytest.raises(RuntimeError):
            MemStateRef("missing.pro", "kernel", SnapshotType.PROLOGUE)


def test_memstate_open_initializes_and_loads():
    with patch("mneme.recorded_execution.Path.exists", return_value=True):
        with patch("mneme.recorded_execution.ffi.lib") as fake:
            fake.MnemePy_initializeMemState.return_value = "STATE"

            m = MemStateRef("snap.pro", "kernelA", SnapshotType.PROLOGUE)
            m.open()

            fake.MnemePy_initializeMemState.assert_called_once()
            fake.MnemePy_LoadMemState.assert_called_once_with("STATE")


def test_memstate_args_lazy_loaded_once():
    with patch("mneme.recorded_execution.Path.exists", return_value=True):
        with patch("mneme.recorded_execution.ffi.lib") as fake:
            fake.MnemePy_initializeMemState.return_value = "S"
            fake.MnemePy_getArgs.return_value = ["A", "B"]

            m = MemStateRef("snap.pro", "kernel", SnapshotType.PROLOGUE)
            m.open()

            a1 = m.args
            a2 = m.args

            assert a1 == a2
            fake.MnemePy_getArgs.assert_called_once()


def test_memstate_num_args_lazy_loaded_once():
    with patch("mneme.recorded_execution.Path.exists", return_value=True):
        with patch("mneme.recorded_execution.ffi.lib") as fake:
            fake.MnemePy_initializeMemState.return_value = "S"
            fake.MnemePy_getNumArgs.return_value = 7

            m = MemStateRef("snap.pro", "kernel", SnapshotType.PROLOGUE)
            m.open()

            n1 = m.num_args
            n2 = m.num_args

            assert n1 == n2 == 7
            fake.MnemePy_getNumArgs.assert_called_once()


def test_memstate_reset_requires_load():
    with patch("mneme.recorded_execution.Path.exists", return_value=True):
        m = MemStateRef("snap.pro", "kernel", SnapshotType.PROLOGUE)
        with pytest.raises(RuntimeError):
            m.reset()


def test_memstate_context_manager_calls_open_and_close():
    with patch("mneme.recorded_execution.Path.exists", return_value=True):
        with patch("mneme.recorded_execution.ffi.lib") as fake:
            fake.MnemePy_initializeMemState.return_value = "ST"

            m = MemStateRef("snap.pro", "kernel", SnapshotType.PROLOGUE)

            with m:
                fake.MnemePy_LoadMemState.assert_called_once_with("ST")

            fake.MnemePy_DisposeMemState.assert_called_once_with("ST")


def test_memstate_equality_uses_ffi_compare():
    with patch("mneme.recorded_execution.Path.exists", return_value=True):
        with patch("mneme.recorded_execution.ffi.lib") as fake:
            fake.MnemePy_initializeMemState.side_effect = ["A", "B"]
            fake.MnemePy_CompareMemState.return_value = True

            m1 = MemStateRef("p1.pro", "k", SnapshotType.PROLOGUE).open()
            m2 = MemStateRef("p2.pro", "k", SnapshotType.PROLOGUE).open()

            assert m1 == m2
            fake.MnemePy_CompareMemState.assert_called_once_with("A", "B")


# ======================================================================
#                      RecordedExecution Tests
# ======================================================================

def test_recorded_execution_to_dict():
    fake_instance = MagicMock()
    fake_instance.to_dict.return_value = {"X": 1}

    r = RecordedExecution(
        static_hash="S",
        kernel_name="K",
        demangled_name="DK",
        llvm_files=["a.ll"],
        arg_names=["a0"],
        specializations=[True],
        va_addr="0x100",
        va_size=32,
        kernel_instances={"hashX": fake_instance},
        captured_globals=["g_data"],
    )

    d = r.to_dict()

    assert d["KernelName"] == "K"
    assert d["Modules"] == ["a.ll"]
    assert d["Globals"] == ["g_data"]
    assert d["instances"]["hashX"] == {"X": 1}


def test_recorded_execution_link_llvm_modules_calls_jit():
    with patch("mneme.recorded_execution.jit.link_llvm_modules") as fake_link:
        fake_link.return_value = "MOD"

        r = RecordedExecution(
            "S", "K", "DK",
            ["a.ll", "b.ll"],
            ["x"],
            [True, False],
            "0x100",
            128,
            {},
        )

        out = r.link_llvm_modules(prune=True, internalize=False)

        fake_link.assert_called_once_with(
            ["a.ll", "b.ll"], "K", True, False, preserve_globals=[]
        )
        assert out == "MOD"


def test_recorded_execution_from_json_reconstructs():
    data = {
        "StaticHash": "S",
        "KernelName": "K",
        "DemangledName": "DK",
        "Modules": ["modA.ll"],
        "Globals": ["g_data"],
        "ArgNames": ["x"],
        "Specializations": [True, True, False],
        "VAddr": "ADDR",
        "VASize": 64,
        "instances": {
            "H": {
                "Args": [True, False],
                "SharedMem": 0,
                "BlockDims": {"x": 1, "y": 2, "z": 3},
                "GridDims": {"x": 4, "y": 5, "z": 6},
                "Occurrences": 3,
                "Prologue": "file.pro",
                "Epilogue": "file.epi",
            }
        },
    }

    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        json.dump(data, f)
        fname = f.name

    # Pretend all the module/prologue/epilogue files exist
    with patch("mneme.recorded_execution.Path.exists", return_value=True):
        r = RecordedExecution.from_json(fname)

    assert r.kernel_name == "K"
    assert r.captured_globals == ["g_data"]
    assert "H" in r.kernel_instances
    inst = r.kernel_instances["H"]
    assert inst.block_dim.x == 1
    assert inst.grid_dim.z == 6
    assert inst.occ == 3
