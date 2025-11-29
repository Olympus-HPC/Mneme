import json
import pathlib
import re

import pandas as pd
from mneme.cli import main as mneme_main
from mneme.db import MnemeDB
from mneme.recorded_execution import RecordedExecution


def test_execute(recorded_execution, tmp_path):
    recorded_execution_descr = RecordedExecution.from_json(recorded_execution)
    key = list(recorded_execution_descr.kernel_instances.keys())[0]
    tmp_dir = tmp_path / "execute"

    result = mneme_main(
        [
            "execute",
            "-db",
            str(recorded_execution),
            "-rid",
            str(key),
            "--results-db-dir",
            str(tmp_dir),
            "--specialize",
            "--dims",
            "--max-threads",
            "--min-threads-per-block",
            "2",
            "default<Oz>",
        ]
    )

    results_csv = tmp_dir / "results.csv"
    assert MnemeDB.verify_db(str(results_csv))

    result = mneme_main(
        [
            "execute",
            "-db",
            str(recorded_execution),
            "-rid",
            str(key),
            "--results-db-dir",
            str(tmp_dir),
            "--specialize",
            "--dims",
            "--max-threads",
            "--min-threads-per-block",
            "2",
            "default<O1>",
        ]
    )
    assert MnemeDB.verify_db(str(results_csv))
    results_csv.unlink()
    result = mneme_main(
        [
            "execute",
            "-db",
            str(recorded_execution),
            "-rid",
            str(key),
            "--results-db-dir",
            str(tmp_dir),
            "--specialize",
            "--dims",
            "--apply-increamentally",
            "--max-threads",
            "--min-threads-per-block",
            "2",
            "mem2reg,loop-load-elim,dce",
        ]
    )
    print(results_csv)
    assert MnemeDB.verify_db(str(results_csv))
    num_lines = sum(1 for _ in results_csv.open())
    # We apply 1 for empty pipeline + 3 for the passes + 1 for the header
    assert num_lines == 5, "Expected 4 experiments when applying increamentally"
