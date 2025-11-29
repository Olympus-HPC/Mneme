import json
import pathlib
import re

from mneme.cli import main as mneme_main


def test_copy(recorded_execution, tmp_path):
    src = recorded_execution
    dst = tmp_path / "copy"
    dst.mkdir()

    result = mneme_main(["copy", str(src), str(dst)])
    assert result == 0
    src_records = set(src.glob("*.json"))
    dest_records = set(dst.glob("*.json"))
    diff = src_records - dest_records
    assert not diff, f"The sets should be empty {diff}"
