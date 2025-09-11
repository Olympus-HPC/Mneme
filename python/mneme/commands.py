import argparse
import os
import shutil
import sys
import time
from pathlib import Path

from mneme.logging import logger
from mneme.recorded_execution import RecordedExecution


def _copy_or_move(sources, dest, move=False):
    for s in sources:
        if not s.exists():
            continue

        exec = RecordedExecution.from_json(str(s))
        new_ir_files = []
        for ll in exec.llvm_files:
            dest_ll = dest / Path(ll).name
            if move:
                shutil.move(ll, dest_ll)
            else:
                shutil.copy(ll, dest_ll)
            new_ir_files.append(str(dest_ll))

        exec.llvm_files = new_ir_files

        for _, kernel in exec.items():
            kfn = Path(kernel.prologue.fn)
            dest_kfn = dest / kfn.name
            if move:
                shutil.move(kfn, dest_kfn)
            else:
                shutil.copy(kfn, dest_kfn)
            kernel.prologue.fn = str(dest_kfn)

            kfn = Path(kernel.epilogue.fn)
            dest_kfn = dest / kfn.name
            if move:
                shutil.move(kfn, dest_kfn)
            else:
                shutil.copy(kfn, dest_kfn)
            kernel.epilogue.fn = str(dest_kfn)

        json_fn = dest / s.name
        exec.to_json(str(json_fn))
        if move:
            s.unlink()


class Clean:
    @staticmethod
    def set_cli_args(parser):
        parser.add_argument("dbs", help="Recorded Mneme DB files", nargs="+")
        parser.set_defaults(func=Clean.run)

    @staticmethod
    def run(args):
        dbs = args.dbs
        for db in dbs:
            if not Path(db).exists():
                raise FileNotFoundError(f"Mneme Database '{db}' does not exist")

        ir_records = set()
        mneme_db_files = set()
        for db in dbs:
            exec = RecordedExecution.from_json(db)
            ir_records |= {Path(e) for e in exec.llvm_files}
            for _, kernel in exec.items():
                mneme_db_files.add(Path(kernel.prologue.fn))
                mneme_db_files.add(Path(kernel.epilogue.fn))
            Path(db).unlink()

        for fn in ir_records:
            if fn.exists():
                fn.unlink()

        for fn in mneme_db_files:
            if fn.exists():
                fn.unlink()


class Copy:
    @staticmethod
    def set_cli_args(parser):
        parser.add_argument(
            "dbs",
            nargs="+",
            help="Source paths of Mneme DB records followed by a destination path to which data will be copied to",
        )
        parser.set_defaults(func=Copy.run)

    @staticmethod
    def run(args):
        paths = args.dbs
        if len(paths) < 2:
            raise ValueError(
                f"Please provide both source and destination directories {paths}"
            )

        *_sources, _dest = paths
        dest = Path(_dest).absolute()
        if not dest.exists():
            raise RuntimeError(f"Destination target '{str(dest)}' does not exist")
        if not dest.is_dir():
            raise NotADirectoryError(
                f"Destination target '{str(dest)}' is not a directory"
            )

        sources = [Path(s) for s in _sources]
        _copy_or_move(sources, dest)


class Move:
    @staticmethod
    def set_cli_args(parser):
        parser.add_argument(
            "dbs",
            nargs="+",
            help="Source paths of Mneme DB records followed by a destination path to which data will be moved to",
        )
        parser.set_defaults(func=Move.run)

    @staticmethod
    def run(args):
        paths = args.dbs
        if len(paths) < 2:
            raise ValueError(
                f"Please provide both source and destination directories {paths}"
            )

        *_sources, _dest = paths
        dest = Path(_dest).absolute()
        if not dest.exists():
            raise RuntimeError(f"Destination target '{str(dest)}' does not exist")
        if not dest.is_dir():
            raise NotADirectoryError(
                f"Destination target '{str(dest)}' is not a directory"
            )

        sources = [Path(s) for s in _sources]

        _copy_or_move(sources, dest, move=True)
