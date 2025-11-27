import json
import pathlib
import re

from mneme.cli import main as mneme_main


def test_move(recorded_execution, tmp_path):
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
