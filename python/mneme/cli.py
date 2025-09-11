import argparse
import csv
import hashlib
import json
import math
import multiprocessing

try:
    multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass

import os
import sys
import time
from typing import Tuple

from mneme.commands import Clean, Copy, Move
from mneme.device import DeviceModule, dim3, get_device_arch
from mneme.llvm.module import ModuleRef
from mneme.page_manager import PageManagerRef
from mneme.pipeline import PipelineManager
from mneme.proteus import jit
from mneme.recorded_execution import MemStateRef, RecordedExecution
from mneme.replay_executor import BaseExecutor, CLIExecutor
from mneme.tuner import ReplayTuner


def main():
    parser = argparse.ArgumentParser(prog="mneme", description="Mneme Tool")
    parent_executor_parser = BaseExecutor.get_base_parser()

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        required=True,
        description="The mneme available commands",
    )

    p_exec = subparsers.add_parser(
        "execute", parents=[parent_executor_parser], help="Run a recorded kernel"
    )

    p_tune = subparsers.add_parser(
        "tune",
        parents=[parent_executor_parser],
        help="Tune the pipeline of a recorded kernel",
    )

    p_clean = subparsers.add_parser(
        "Clean", parents=[], help="Clean mneme generated files"
    )

    p_copy = subparsers.add_parser(
        "copy",
        parents=[],
        help="Copy mneme generated files and update database accordingly",
    )

    p_move = subparsers.add_parser(
        "move",
        parents=[],
        help="Move mneme generated files and update database accordingly",
    )

    CLIExecutor.set_cli_args(p_exec)
    ReplayTuner.set_cli_args(p_tune)
    Clean.set_cli_args(p_clean)
    Copy.set_cli_args(p_copy)
    Move.set_cli_args(p_move)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    # We need this to avoid copying the context
    sys.exit(main())
