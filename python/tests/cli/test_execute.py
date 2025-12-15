import json
import pathlib
import re

import pandas as pd
from mneme.cli import main as mneme_main
from mneme.recorded_execution import RecordedExecution


def test_execute(recorded_execution, tmp_path):
    recorded_execution_descr = RecordedExecution.from_json(recorded_execution)
    key = list(recorded_execution_descr.kernel_instances.keys())[0]
    tmp_dir = tmp_path / "execute"

    result = mneme_main(
        [
            "execute",
            "-rdb",
            str(recorded_execution),
            "-rid",
            str(key),
            "--specialize",
            "-sdims",
            "--max-threads",
            "5",
            "--min-threads-per-block",
            "3",
            "default<Oz>",
        ]
    )

    result = mneme_main(
        [
            "execute",
            "-rdb",
            str(recorded_execution),
            "-rid",
            str(key),
            "--specialize",
            "default<O1>",
        ]
    )
    print(
        " ".join(
            [
                "execute",
                "-rdb",
                str(recorded_execution),
                "-rid",
                str(key),
                "--specialize",
                "-sdims",
                "--max-threads",
                "256",
                "--min-threads-per-block",
                "2",
                "mem2reg,loop-load-elim,dce",
            ]
        )
    )
    result = mneme_main(
        [
            "execute",
            "-rdb",
            str(recorded_execution),
            "-rid",
            str(key),
            "--specialize",
            "-sdims",
            "--max-threads",
            "256",
            "--min-threads-per-block",
            "2",
            "mem2reg,loop-load-elim,dce",
        ]
    )
