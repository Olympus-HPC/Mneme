import csv
import json
import pathlib
import sys

from mneme.experiment import Experiment
from mneme.fancy_out import print_experiment_status
from mneme.logging import logger


class MnemeDB:
    def __init__(self, dir, static_hash, dynamic_hash, suffix=None):
        self._dir = pathlib.Path(dir)
        self._static_hash = static_hash
        self._dynamic_hash = dynamic_hash
        self._experiments = {}
        self._open = False
        self._suffix = suffix
        self._best = sys.float_info.max
        self._o3 = None
        if suffix is not None:
            self._prefix = (
                f"mneme.{self._static_hash}.{self._dynamic_hash}.{self._suffix}"
            )
        else:
            self._prefix = f"mneme.{self._static_hash}.{self._dynamic_hash}"

        self._filename = f"{self._prefix}.csv"

        self._columns = [
            "hash",
            "orig_ir",
            "compiled_ir",
            "specialize",
            "max_threads",
            "min_blocks_per_sm",
            "specialize_dims",
            "passes",
            "prune",
            "internalize",
            "codegen_opt",
            "codegen_method",
            "device_arch",
            "opt_time",
            "codegen_time",
            "verified",
            "obj_size",
            "exec_time_std",
            "exec_time_avg",
            "exec_time_median",
            "exec_time_r",
            "exec_time_rp",
            "exec_time_iqr",
            "exec_time_iqrp",
            "exec_time_q25",
            "exec_time_q75",
            "executed",
            "failed",
            "start_time",
            "end_time",
            "start_id",
            "commit_id",
            "gpu_id",
            "reg_usage",
            "const_mem",
            "local_mem",
        ]

    @property
    def prefix(self):
        return self._prefix

    @property
    def db_dir(self):
        return self._dir

    @property
    def is_open(self):
        return self._open

    def __len__(self):
        return len(self._experiments)

    def open(self):
        logger.debug(f"Opening database under directory: '{str(self._dir)}'")
        if not self._dir.exists():
            logger.debug(f"Making directory: '{str(self._dir)}'")
            self._dir.mkdir(parents=True, exist_ok=True)
        self._dir = self._dir.resolve()
        self._open = True

        self._filename = self._dir / pathlib.Path(self._filename)
        logger.debug(f"Database file is {str(self._filename)}")
        if not self._filename.exists():
            with open(self._filename, mode="w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self._columns)
                writer.writeheader()
            return self

        with open(self._filename, mode="r") as fd:
            reader = csv.DictReader(fd)
            values = []
            for row in reader:
                values.append((row["hash"], row["executed"]))
                if row["failed"].lower() == "false":
                    exp = Experiment.from_dict(**row)
                    if self._is_baseline(exp):
                        self._o3 = float(row["exec_time_median"])
                        logger.debug(f"Set baseline from csv baseline {self._o3}")

                    if self._best > float(row["exec_time_median"]):
                        self._best = float(row["exec_time_median"])
                        logger.debug(f"Set BEST from csv baseline {self._best}")

            for v in values:
                if v[1]:
                    self._experiments[v[0]] = v[1]

        if self._o3 is not None and self._best != sys.float_info.max:
            print(f"Optimal is {self._best} while O3 is {self._o3}")

        return self

    def close(self):
        self._open = False
        return

    def __enter__(self):
        return self.open()

    def __exit__(self):
        return

    def should_execute(self, exp: Experiment):
        _hash = exp.hash()
        if _hash in self._experiments:
            return False
        return True

    def suggest_ir_fn_name(self, exp: Experiment):
        return f"{self._dir}/{exp.hash()}.ll"

    def _is_baseline(self, exp):
        logger.debug(
            f"Testing whether this is a baseline {json.dumps(exp.to_dict(), indent=2)}"
        )
        if exp.passes != "default<O3>" and exp.passes != "default<O3>,globaldce":
            return False
        if exp.specialize != False:
            return False
        if exp.specialize_dims != 0:
            return False
        if exp.max_threads != 0:
            return False
        if exp.min_blocks_per_sm != 0:
            return False

        return True

    def add(self, src_ir: str, dst_ir: str, exp: Experiment):
        if not self._open:
            raise RuntimeError("Expected database to be open")

        _hash = exp.hash()

        if self._is_baseline(exp):
            self._o3 = exp.exec_time

        with open(self._filename, mode="a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self._columns)
            values = [str(_hash), src_ir, dst_ir]
            _exp = exp.to_dict()
            _exp["hash"] = str(_hash)
            _exp["orig_ir"] = src_ir
            _exp["compiled_ir"] = dst_ir
            writer.writerow(_exp)

        if self._o3 is not None and exp.exec_time is not None:
            speedup = self._o3 / exp.exec_time
            best_speedup = self._o3 / self._best

            print_experiment_status(
                _hash,
                not exp.failed,
                exp.verified,
                speedup,
                best_speedup,
            )

        if self._best > exp.exec_time:
            self._best = exp.exec_time

        self._experiments[_hash] = exp.executed

    def save_ir(self, ir, _id):
        fn = f"{str(self._dir)}/{self._static_hash}.{self._dynamic_hash}.{_id}.bc"
        ir.to_bitcode(fn)
        return fn
