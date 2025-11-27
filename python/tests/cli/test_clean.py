import json
import pathlib
import re

from mneme.cli import main as mneme_main


def test_clean(recorded_execution, tmp_path):
    src = recorded_execution
    src_dir = recorded_execution.parent

    result = mneme_main(["clean", str(src)])
    assert result == 0
    assert not any(src_dir.iterdir()), "Directory should be empty"
