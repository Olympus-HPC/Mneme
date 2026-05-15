import json
import pathlib
import re

from mneme.cli import main as mneme_main
from mneme.recorded_execution import RecordedExecution


def test_move(recorded_execution, tmp_path, monkeypatch):
    src = recorded_execution
    src_dir = recorded_execution.parent
    src_files = set([p.name for p in src_dir.iterdir()])
    dst = tmp_path / "move"
    dst.mkdir()

    result = mneme_main(["move", str(src), str(dst)])
    assert result == 0
    dest_files = set([p.name for p in dst.iterdir()])
    diff = dest_files - src_files
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
