import json
import pathlib
import re

from mneme.cli import main as mneme_main
from mneme.recorded_execution import RecordedExecution


def test_copy(recorded_execution, tmp_path, monkeypatch):
    src = recorded_execution
    dst = tmp_path / "copy"
    dst.mkdir()

    result = mneme_main(["copy", str(src), str(dst)])
    assert result == 0
    src_records = set(p.name for p in src.parent.glob("*.json"))
    dest_records = set(p.name for p in dst.glob("*.json"))
    diff = src_records - dest_records
    assert not diff, f"The sets should be empty {diff}"

    dest_json = next(dst.glob("*.json"))
    written = json.loads(dest_json.read_text())
    for m in written["Modules"]:
        assert "/" not in m, f"Module path should be a basename: {m}"
    for inst in written["instances"].values():
        assert "/" not in inst["Prologue"], inst["Prologue"]
        assert "/" not in inst["Epilogue"], inst["Epilogue"]

    foreign_cwd = tmp_path / "elsewhere"
    foreign_cwd.mkdir()
    monkeypatch.chdir(foreign_cwd)
    RecordedExecution.from_json(str(dest_json))
