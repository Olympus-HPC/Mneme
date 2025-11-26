import argparse
import csv
import hashlib
import json
import logging
import math
import multiprocessing

try:
    multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass

import os
import sys
import time
from typing import Tuple, Union

from mneme.commands import Clean, Copy, Detail, Move, Serve, Summary, Record, Config
from mneme.device import DeviceModule, dim3, get_device_arch
from mneme.llvm.module import ModuleRef
from mneme.logging import logger as replay_logger
from mneme.page_manager import PageManagerRef
from mneme.pipeline import PipelineManager
from mneme.proteus import jit
from mneme.recorded_execution import MemStateRef, RecordedExecution
from mneme.replay_executor import BaseExecutor, CLIExecutor
from mneme.tuner import ReplayTuner

_LEVELS = {
    "critical": logging.CRITICAL,
    "warn": logging.WARNING,  # accept 'warn' per your spec
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


def configure_replay_logging(level_name: Union[str, None]) -> None:
    """Attach a real handler only when -v is provided."""
    if not level_name:
        # Stay silent: leave only the NullHandler in place
        return

    level = _LEVELS[level_name.lower()]
    # Avoid double-adding if called twice
    if not any(isinstance(h, logging.StreamHandler) for h in replay_logger.handlers):
        h = logging.StreamHandler()  # stderr by default
        # Short, friendly format; tweak as you like
        fmt = logging.Formatter("[mneme:%(levelname).1s] %(message)s")
        h.setFormatter(fmt)
        replay_logger.addHandler(h)

    replay_logger.setLevel(level)


def main():
    parser = argparse.ArgumentParser(
        prog="mneme", description="Mneme Tool for Autotuning GPU kernels"
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        choices=["critical", "warn", "info", "debug"],
        help="Set replay logger level (default: silent)",
    )
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
        "clean", parents=[], help="clean mneme generated files"
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

    p_summary = subparsers.add_parser(
        "summary",
        parents=[],
        help="summarize the best run of a tune campaign",
    )

    p_serve = subparsers.add_parser(
        "serve",
        parents=[],
        help="Export optimal configuration to some json file for Proteus to digest and use",
    )

    p_detail = subparsers.add_parser(
        "detail",
        parents=[],
        help="Provide details of a specific experiment",
    )

    p_record = subparsers.add_parser(
        "record",
        parents=[],
        help="Record the execution of an application"
    )

    p_config = subparsers.add_parser(
        "config",
        parents=[],
        help="Get mneme config options"
    )

    CLIExecutor.set_cli_args(p_exec)
    ReplayTuner.set_cli_args(p_tune)
    Clean.set_cli_args(p_clean)
    Copy.set_cli_args(p_copy)
    Move.set_cli_args(p_move)
    Summary.set_cli_args(p_summary)
    Serve.set_cli_args(p_serve)
    Detail.set_cli_args(p_detail)
    Record.set_cli_args(p_record)
    Config.set_cli_args(p_config)

    args = parser.parse_args()
    verbosity = vars(args).pop("verbosity", None)
    configure_replay_logging(verbosity)

    return args.func(args, verbosity)


if __name__ == "__main__":
    # We need this to avoid copying the context
    sys.exit(main())
