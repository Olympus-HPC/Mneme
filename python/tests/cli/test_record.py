import json
import pathlib
import re

from mneme.cli import main as mneme_main
from mneme.recorded_execution import RecordedExecution


def test_record(recorded_execution):
    assert recorded_execution.exists()
