# python/tests/test_cli_summary.py

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest
from mneme.commands import Summary
from mneme.llvm import module
from mneme.proteus import jit
from mneme.recorded_execution import RecordedExecution


@pytest.fixture
def summary_parser():
    parser = argparse.ArgumentParser(prog="mneme summary")
    Summary.set_cli_args(parser)
    return parser


def test_summary_cli_invokes_compute_speedups(tmp_path, summary_parser, monkeypatch):
    """
    Ensure:
    - CLI parses arguments
    - compute_speedups is called with parsed CSV
    - ALL LLVM operations (bitcode parse, linking, get_function) are mocked
    """

    # ------------------------------------------------------------------
    # Create dummy bitcode file so open(...) does not fail
    # ------------------------------------------------------------------
    dummy_bc = tmp_path / "dummy.bc"
    dummy_bc.write_bytes(b"\xDE\xAD\xBE\xEF")

    # ------------------------------------------------------------------
    # Fake RecordedExecution
    # ------------------------------------------------------------------
    class FakeExec:
        kernel_name = "vec_add"
        llvm_files = [str(dummy_bc)]  # IMPORTANT
        demangled_name = "vec_add"
        specializations = []

        @staticmethod
        def items():
            return []

    monkeypatch.setattr(
        RecordedExecution, "from_json", staticmethod(lambda p: FakeExec)
    )

    # ------------------------------------------------------------------
    # Fake bitcode parsing + fake LLVM Function
    # ------------------------------------------------------------------
    class FakeFunc:
        @staticmethod
        def get_function_location():
            # root, src, line
            return ("/tmp", "fake.cu", 123)

    class FakeMod:
        @staticmethod
        def get_function(name):
            return FakeFunc()

    monkeypatch.setattr(module, "parse_bitcode", lambda ir: FakeMod)

    # ------------------------------------------------------------------
    # Fake jit linker
    # ------------------------------------------------------------------
    monkeypatch.setattr(jit, "link_llvm_modules", lambda *a, **kw: None)

    # ------------------------------------------------------------------
    # Fake compute_speedups
    # ------------------------------------------------------------------
    called = {}

    def fake_compute(df):
        called["df"] = df
        return True, {"Baseline": [123.0, "H"]}

    monkeypatch.setattr(Summary, "compute_speedups", staticmethod(fake_compute))

    # ------------------------------------------------------------------
    # Fake render_report
    # ------------------------------------------------------------------
    monkeypatch.setattr(Summary, "render_report", lambda *a, **kw: None)

    # ------------------------------------------------------------------
    # Create fake CSV
    # ------------------------------------------------------------------
    csv_file = tmp_path / "results.csv"
    df = pd.DataFrame(
        {
            "specialize": [False],
            "max_threads": [0],
            "min_blocks_per_sm": [0],
            "exec_time_median": [100],
            "hash": ["H"],
            "passes": ["default<O3>,globaldce"],
        }
    )
    df.to_csv(csv_file, index=False)

    # ------------------------------------------------------------------
    # Create fake DB JSON
    # ------------------------------------------------------------------
    db_file = tmp_path / "fake_db.json"
    db_file.write_text(
        json.dumps(
            {
                "kernel_name": "vec_add",
                "llvm_files": [str(dummy_bc)],
                "demangled_name": "vec_add",
            }
        )
    )

    # ------------------------------------------------------------------
    # Run CLI
    # ------------------------------------------------------------------
    args = summary_parser.parse_args(["-rdb", str(db_file), str(csv_file)])
    Summary.analyze(args)

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    assert "df" in called
    assert isinstance(called["df"], pd.DataFrame)
