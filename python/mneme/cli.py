import argparse
import multiprocessing

try:
    multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass

import sys

from mneme.commands import (
    Clean,
    Config,
    Copy,
    Execute,
    Move,
    Record,
)
from mneme.logging import configure_replay_logging


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="mneme", description="Mneme Tool for Autotuning GPU kernels"
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        choices=["critical", "warn", "info", "debug"],
        help="Set replay logger level (default: silent)",
    )

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        required=True,
        description="The mneme available commands",
    )

    p_exec = subparsers.add_parser("execute", parents=[], help="Run a recorded kernel")

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

    p_record = subparsers.add_parser(
        "record", parents=[], help="Record the execution of an application"
    )

    p_config = subparsers.add_parser(
        "config", parents=[], help="Get mneme config options"
    )

    Execute.set_cli_args(p_exec)
    Clean.set_cli_args(p_clean)
    Copy.set_cli_args(p_copy)
    Move.set_cli_args(p_move)
    Record.set_cli_args(p_record)
    Config.set_cli_args(p_config)

    args = parser.parse_args(argv)
    verbosity = vars(args).pop("verbosity", None)
    configure_replay_logging(verbosity)

    return args.func(args, verbosity)


if __name__ == "__main__":
    # We need this to avoid copying the context
    sys.exit(main())
