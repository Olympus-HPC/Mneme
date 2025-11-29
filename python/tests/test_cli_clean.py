# python/tests/test_clean_cli.py

import argparse
from pathlib import Path

import pytest
from mneme.commands import Clean  # adjust if needed
from mneme.recorded_execution import RecordedExecution

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def clean_parser():
    parser = argparse.ArgumentParser(prog="mneme clean")
    Clean.set_cli_args(parser)
    return parser


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_clean_happy_path(tmp_path, clean_parser, monkeypatch):
    """
    Test that Clean.run correctly deletes:
    - the DB json file(s)
    - llvm_files IR artifacts
    - kernel prologue/epilogue files
    """

    # ----------------------------------------------------
    # Setup: create fake files on disk
    # ----------------------------------------------------
    db1 = tmp_path / "rec1.json"
    db1.write_text("{}")
    db2 = tmp_path / "rec2.json"
    db2.write_text("{}")

    # Fake IR + kernel files
    ir1 = tmp_path / "ir1.ll"
    ir1.write_text("IR1")
    ir2 = tmp_path / "ir2.ll"
    ir2.write_text("IR2")

    k1_pro = tmp_path / "kernel1_pro.ll"
    k1_pro.write_text("K1P")
    k1_epi = tmp_path / "kernel1_epi.ll"
    k1_epi.write_text("K1E")

    k2_pro = tmp_path / "kernel2_pro.ll"
    k2_pro.write_text("K2P")
    k2_epi = tmp_path / "kernel2_epi.ll"
    k2_epi.write_text("K2E")

    # ----------------------------------------------------
    # Monkeypatch RecordedExecution.from_json
    # ----------------------------------------------------
    class FakeKernel:
        def __init__(self, pro, epi):
            class Pro:
                fn = str(pro)

            class Epi:
                fn = str(epi)

            self.prologue = Pro()
            self.epilogue = Epi()

    class FakeExecution:
        def __init__(self, ir_files, kernels):
            self.llvm_files = ir_files
            self._kernels = kernels

        def items(self):
            # simulate dict-like iteration
            return [(f"k{i}", k) for i, k in enumerate(self._kernels)]

    # Map db file path → FakeExecution instance
    fake_map = {
        str(db1): FakeExecution([str(ir1)], [FakeKernel(k1_pro, k1_epi)]),
        str(db2): FakeExecution([str(ir2)], [FakeKernel(k2_pro, k2_epi)]),
    }

    def fake_from_json(path):
        return fake_map[path]

    monkeypatch.setattr(RecordedExecution, "from_json", staticmethod(fake_from_json))

    # ----------------------------------------------------
    # Execute Clean.run
    # ----------------------------------------------------
    args = clean_parser.parse_args([str(db1), str(db2)])
    Clean.run(args, verbosity=None)

    # ----------------------------------------------------
    # Assertions: all files must be deleted
    # ----------------------------------------------------
    for f in [db1, db2, ir1, ir2, k1_pro, k1_epi, k2_pro, k2_epi]:
        assert not f.exists(), f"{f} should have been deleted"


def test_clean_missing_db_file_raises(clean_parser):
    """A missing DB file should raise FileNotFoundError."""
    args = clean_parser.parse_args(["does_not_exist.json"])
    with pytest.raises(FileNotFoundError):
        Clean.run(args, verbosity=None)


def test_clean_only_deletes_expected_files(tmp_path, clean_parser, monkeypatch):
    """
    Ensure Clean only deletes:
    - the DB json file
    - the IR files in .llvm_files
    - the kernel prologue/epilogue files
    and does NOT delete other unrelated files.
    """

    db = tmp_path / "rec.json"
    db.write_text("{}")

    ir = tmp_path / "ir.ll"
    ir.write_text("IR")

    pro = tmp_path / "pro.ll"
    pro.write_text("PRO")

    epi = tmp_path / "epi.ll"
    epi.write_text("EPI")

    other = tmp_path / "keep_me.txt"
    other.write_text("KEEP")

    class FakeKernel:
        def __init__(self, p, e):
            class Pro:
                fn = str(p)

            class Epi:
                fn = str(e)

            self.prologue = Pro()
            self.epilogue = Epi()

    class FakeExecution:
        def __init__(self):
            self.llvm_files = [str(ir)]
            self._kernels = [FakeKernel(pro, epi)]

        def items(self):
            return [("k", self._kernels[0])]

    monkeypatch.setattr(
        RecordedExecution, "from_json", staticmethod(lambda path: FakeExecution())
    )

    args = clean_parser.parse_args([str(db)])
    Clean.run(args, verbosity=None)

    # Expected deletions
    for f in [db, ir, pro, epi]:
        assert not f.exists()

    # Should still exist
    assert other.exists()
