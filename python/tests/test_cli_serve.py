import argparse
import json
from pathlib import Path

import pandas as pd
import pytest
from mneme.commands import Serve
from mneme.llvm import module
from mneme.proteus import jit
from mneme.recorded_execution import RecordedExecution


@pytest.fixture
def serve_parser():
    parser = argparse.ArgumentParser(prog="mneme serve")
    Serve.set_cli_args(parser)
    return parser


def test_serve_happy_path(tmp_path, serve_parser, monkeypatch):
    """
    Validate:
    - CLI parses arguments
    - RecordedExecution.from_json is called
    - bitcode is parsed
    - dataframe is filtered
    - proteus-json file is written
    - render_output is called
    """

    # ----------------------------------------------------------------------
    # Create a real dummy bitcode file so open(...) does not crash
    # ----------------------------------------------------------------------
    dummy_bc = tmp_path / "dummy.bc"
    dummy_bc.write_bytes(b"\xDE\xAD\xBE\xEF")

    # ----------------------------------------------------------------------
    # Fake RecordedExecution object
    # ----------------------------------------------------------------------
    class FakeExec:
        kernel_name = "vecAdd"
        llvm_files = [str(dummy_bc)]
        demangled_name = "vecAdd"
        specializations = [True, False, True]

        @staticmethod
        def items():
            return []

    monkeypatch.setattr(
        RecordedExecution, "from_json", staticmethod(lambda path: FakeExec)
    )

    # ----------------------------------------------------------------------
    # Fake LLVM bitcode → module → function → location
    # ----------------------------------------------------------------------
    class FakeFunc:
        @staticmethod
        def get_function_location():
            return ("/root/path", "file.cu", 42)

    class FakeMod:
        @staticmethod
        def get_function(name):
            return FakeFunc()

    monkeypatch.setattr(module, "parse_bitcode", lambda _: FakeMod)

    # ----------------------------------------------------------------------
    # Fake jit linker
    # ----------------------------------------------------------------------
    monkeypatch.setattr(jit, "link_llvm_modules", lambda *a, **kw: None)

    # ----------------------------------------------------------------------
    # CSV input file with tuning results
    # ----------------------------------------------------------------------
    csv_file = tmp_path / "results.csv"
    df_input = pd.DataFrame(
        {
            "hash": ["A", "B"],
            "exec_time_median": [50, 20],  # B is best
            "verified": [True, True],
            "failed": [False, False],
            "max_threads": [0, 128],
            "min_blocks_per_sm": [0, 2],
            "passes": ["pA", "pB"],
            "codegen_method": ["cA", "cB"],
            "specialize": [False, True],
            "specialize_dims": [False, True],
            "codegen_opt": [2, 3],
        }
    )
    df_input.to_csv(csv_file, index=False)

    # ----------------------------------------------------------------------
    # Fake render_output — we only verify it was called
    # ----------------------------------------------------------------------
    render_calls = {}

    def fake_render(mneme_cfg, exec_time, console, func, fpath, line, params, raw):
        render_calls["mneme_cfg"] = mneme_cfg
        render_calls["exec_time"] = exec_time
        render_calls["func"] = func
        render_calls["fpath"] = fpath
        render_calls["line"] = line
        render_calls["params"] = params

    monkeypatch.setattr(Serve, "render_output", fake_render)

    # ----------------------------------------------------------------------
    # Target JSON file that Serve should write
    # ----------------------------------------------------------------------
    json_out = tmp_path / "proteus.json"

    # ----------------------------------------------------------------------
    # Run CLI
    # ----------------------------------------------------------------------
    args = serve_parser.parse_args(
        [
            "--record_database",
            "fake_db.json",
            "--results",
            str(csv_file),
            str(json_out),  # proteus-json positional arg
        ]
    )

    Serve.serve(args)

    # ----------------------------------------------------------------------
    # Validate JSON file
    # ----------------------------------------------------------------------
    assert json_out.exists()

    with open(json_out, "r") as fd:
        data = json.load(fd)

    # Should have one kernel config
    assert "vecAdd" in data
    cfg = data["vecAdd"]

    assert cfg["TunedMaxThreads"] == 128
    assert cfg["MinBlocksPerSM"] == 2
    assert cfg["Pipeline"] == "pB"
    assert cfg["CodeGen"] == "cB"
    assert cfg["SpecializeArgs"] is True
    assert cfg["SpecializeDims"] is True
    assert cfg["LaunchBounds"] is True  # max_threads != 0
    assert cfg["CodeGenOptLevel"] == 3

    # ----------------------------------------------------------------------
    # Validate render_output invocation
    # ----------------------------------------------------------------------
    assert render_calls["func"] == "vecAdd"
    assert render_calls["line"] == 42
    assert render_calls["params"] == [1, 3]  # specializations True at indices 0 and 2
    assert render_calls["exec_time"] == 20  # best exec_time
    assert "mneme_cfg" in render_calls
