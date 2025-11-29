import json
import pathlib
import re

import pandas as pd
from mneme.cli import main as mneme_main
from mneme.db import MnemeDB
from mneme.recorded_execution import RecordedExecution


def test_execute_random(recorded_execution, tmp_path):
    recorded_execution_descr = RecordedExecution.from_json(recorded_execution)
    key = list(recorded_execution_descr.kernel_instances.keys())[0]
    tmp_dir = tmp_path / "execute"

    result = mneme_main(
        [
            "tune",
            "-db",
            str(recorded_execution),
            "--num-trials",
            "2",
            "--num-workers",
            "1",
            "--tuner-type",
            "random",
            "--average-pipeline-length",
            "30",
            "-rid",
            str(key),
            "--results-db-dir",
            str(tmp_dir),
        ]
    )
    results_csv = tmp_dir / "results.csv"
    assert MnemeDB.verify_db(str(results_csv))
    num_lines = sum(1 for _ in results_csv.open())
    # 140 are a fixe number of experiments I run on this settings. I need to find a better
    # way of defining a search space. This is a little "too" fixed.
    assert num_lines == 141, "Expected 141 experiments when running random"


def test_execute_optuna(recorded_execution, tmp_path):
    recorded_execution_descr = RecordedExecution.from_json(recorded_execution)
    key = list(recorded_execution_descr.kernel_instances.keys())[0]
    tmp_dir = tmp_path / "execute"

    result = mneme_main(
        [
            "tune",
            "-db",
            str(recorded_execution),
            "--num-trials",
            "2",
            "--num-workers",
            "1",
            "--tuner-type",
            "optuna",
            "--average-pipeline-length",
            "30",
            "-rid",
            str(key),
            "--results-db-dir",
            str(tmp_dir),
            "-s",
            "NSGAIISampler",
        ]
    )
    results_csv = tmp_dir / "results.csv"
    print(results_csv)
    assert MnemeDB.verify_db(str(results_csv))
    num_lines = sum(1 for _ in results_csv.open())
    # 140 are a fixe number of experiments I run on this settings. I need to find a better
    # way of defining a search space. This is a little "too" fixed.
    assert num_lines == 18, "Expected 18 experiments when using GA"
