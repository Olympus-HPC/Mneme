import builtins
import csv
import json
import pathlib
import sys

import pytest
from unittest.mock import MagicMock, mock_open, patch

from mneme.db import MnemeDB
from mneme.experiment import Experiment


# ---------------------------------------------------------
# Helper Dummy Experiment
# ---------------------------------------------------------
class DummyExp:
    def __init__(self):
        self._hash = "H123"
        self.exec_time = 10.0
        self.start_id = 1
        self.failed = False
        self.verified = True
        self.executed = True

        # baseline fields:
        self.passes = "default<O3>"
        self.specialize = False
        self.specialize_dims = 0
        self.max_threads = 0
        self.min_blocks_per_sm = 0

    def hash(self):
        return self._hash

    def to_dict(self):
        # minimal fields needed by MnemeDB.write
        return {
            "specialize": self.specialize,
            "max_threads": self.max_threads,
            "min_blocks_per_sm": self.min_blocks_per_sm,
            "specialize_dims": self.specialize_dims,
            "passes": self.passes,
            "prune": False,
            "internalize": False,
            "codegen_opt": 3,
            "codegen_method": "serial",
            "device_arch": "gfx90a",
            "opt_time": None,
            "codegen_time": None,
            "verified": self.verified,
            "obj_size": None,
            "exec_time_std": None,
            "exec_time_avg": self.exec_time,
            "exec_time_median": self.exec_time,
            "exec_time_r": None,
            "exec_time_rp": None,
            "exec_time_iqr": None,
            "exec_time_iqrp": None,
            "exec_time_q25": None,
            "exec_time_q75": None,
            "executed": self.executed,
            "failed": self.failed,
            "start_time": None,
            "end_time": None,
            "start_id": self.start_id,
            "commit_id": "",
            "gpu_id": "",
            "reg_usage": "",
            "const_mem": "",
            "local_mem": "",
        }


# ---------------------------------------------------------
# TEST: open() creates directory and CSV if missing
# ---------------------------------------------------------
def test_db_open_creates_dir_and_file(tmp_path):
    db_dir = tmp_path / "results"
    static_hash = "S"
    dynamic_hash = "D"

    db = MnemeDB(db_dir, static_hash, dynamic_hash)

    # Patch file existence checks and open
    with patch.object(pathlib.Path, "exists", return_value=False), \
         patch.object(pathlib.Path, "mkdir") as mkdir, \
         patch("builtins.open", mock_open()) as m:

        out = db.open()

        assert out is db
        mkdir.assert_called_once()
        m.assert_called_once()  # file opened for header creation
        assert db.is_open is True


# ---------------------------------------------------------
# TEST: should_execute respects cached experiment hashes
# ---------------------------------------------------------
def test_db_should_execute():
    db = MnemeDB("/tmp", "S", "D")
    db._experiments = {"H123": True}

    exp = DummyExp()
    assert db.should_execute(exp) is False

    exp._hash = "OTHER"
    assert db.should_execute(exp) is True


# ---------------------------------------------------------
# TEST: suggest_ir_fn_name formats correctly
# ---------------------------------------------------------
def test_db_suggest_ir_fn_name():
    db = MnemeDB("/tmp/dir", "S", "D")
    exp = DummyExp()

    fn = db.suggest_ir_fn_name(exp)
    assert fn.endswith("/H123.ll")


# ---------------------------------------------------------
# TEST: _is_baseline logic
# ---------------------------------------------------------
def test_db_is_baseline_true():
    db = MnemeDB("/tmp", "S", "D")
    exp = DummyExp()  # baseline fields already set for DummyExp

    assert db._is_baseline(exp) is True


def test_db_is_baseline_false():
    db = MnemeDB("/tmp", "S", "D")
    exp = DummyExp()
    exp.passes = "not_o3"
    assert db._is_baseline(exp) is False


# ---------------------------------------------------------
# TEST: add() writes a row and updates best baseline
# ---------------------------------------------------------
def test_db_add_writes_row(tmp_path):
    db_dir = tmp_path / "results"
    db = MnemeDB(db_dir, "S", "D")
    db._open = True  # simulate db.open()

    exp = DummyExp()

    mo = mock_open()
    with patch("builtins.open", mo):

        db.add("orig.ll", "out.ll", exp)

        mo.assert_called_once()  # CSV open for append

        handle = mo()
        written = handle.write.call_args[0][0]

        # Writer writes a JSON-like row → minimally assert the hash is in output
        assert "H123" in written

    assert "H123" in db._experiments


# ---------------------------------------------------------
# TEST: save_ir writes bitcode
# ---------------------------------------------------------
def test_db_save_ir_calls_to_bitcode(tmp_path):
    class IR:
        def __init__(self):
            self.called = False

        def to_bitcode(self, fn):
            self.called = True
            self.fn = fn

    db = MnemeDB(tmp_path, "S", "D")
    ir = IR()

    out_fn = db.save_ir(ir, 11)

    assert ir.called is True
    assert out_fn.endswith(".bc")


# ---------------------------------------------------------
# TEST: verify_db detects missing file
# ---------------------------------------------------------
def test_db_verify_missing_file():
    with pytest.raises(RuntimeError):
        MnemeDB.verify_db("/does/not/exist.csv")


# ---------------------------------------------------------
# TEST: verify_db detects missing headers
# ---------------------------------------------------------
def test_db_verify_headers_missing(tmp_path):
    p = tmp_path / "file.csv"

    # write CSV with incomplete headers
    with open(p, "w") as fd:
        fd.write("hash,orig_ir,compiled_ir\n")

    assert MnemeDB.verify_db(str(p)) is False


# ---------------------------------------------------------
# TEST: verify_db OK
# ---------------------------------------------------------
def test_db_verify_complete(tmp_path):
    p = tmp_path / "file.csv"

    with open(p, "w") as fd:
        writer = csv.DictWriter(fd, fieldnames=MnemeDB._columns)
        writer.writeheader()

    assert MnemeDB.verify_db(str(p)) is True

