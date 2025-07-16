import argparse
import csv
import hashlib
import json
import math
import multiprocessing
import os
import sys
import time
from typing import Tuple

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
    parent_parser = BaseExecutor.get_base_parser()

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        required=True,
        description="The mneme available commands",
    )

    p_exec = subparsers.add_parser(
        "execute", parents=[parent_parser], help="run a recorded kernel"
    )

    p_tune = subparsers.add_parser(
        "tune", parents=[parent_parser], help="tune the pipeline of a recorded kernel"
    )

    CLIExecutor.set_cli_args(p_exec)
    ReplayTuner.set_cli_args(p_tune)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    # We need this to avoid copying the context
    multiprocessing.set_start_method("spawn")

    sys.exit(main())
