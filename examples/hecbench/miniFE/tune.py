#!/usr/bin/env python3
"""
Mneme tuning example (Optuna)

This example demonstrates how to run a tuning session on a
previously recorded kernel execution.

Workflow:
  1) Load a recorded execution (record-db) and select a kernel (record-id).
  2) Define a (Exhaustive) SearchSpace that exposes tunable parameters.
  3) Run a baseline configuration to verify replay correctness and measure baseline time.
  4) Print the best configuration and its result.

Notes:
  - This example intentionally keeps the API usage explicit and minimal.
  - The Optuna objective is configured as direction="minimize" and the script
    reports a speedup value to Optuna, exactly as shown in the original example.
"""

import argparse
import json
import time
import math
import statistics
import sys
import optuna

from mneme.async_executor import AsyncReplayExecutor
from mneme.mneme_types import ExperimentConfiguration
from mneme.recorded_execution import RecordedExecution
from mneme.tuning.sample_strategy import OptunaSamplingStrategy
from mneme.tuning.search_space import SearchSpace, BoolParam, RealRangeParam


class EntireSpace(SearchSpace):
    def __init__(self, recorded_kernel: RecordedExecution.KernelInstance):
        self.recorded_kernel = recorded_kernel
        self.original_size = recorded_kernel.grid_dim.x * recorded_kernel.block_dim.x
        _pos_min_val = sys.float_info.min * sys.float_info.epsilon
        self._search_space = {
            "warp_fraction": RealRangeParam("warp_fraction", _pos_min_val, 1.0),
            "grid_fraction": RealRangeParam("grid_fraction", _pos_min_val, 1.0),
            "max_threads": RealRangeParam("max_threads", _pos_min_val, 1.0),
        }

    def dimensions(self):
        return self._search_space

    def derived(self, params) -> ExperimentConfiguration:
        # Compute the number of active Warps and map that to the numThreads.
        maxWarpsInBlock = 1024 / 64
        numActiveWarps = min(
            math.ceil(params["warp_fraction"] * maxWarpsInBlock), maxWarpsInBlock
        )
        numThreads = int(numActiveWarps * 64)

        # How many blocks exist out of the total size taking into account the new block size
        maxNumBlocks = int(math.ceil(self.original_size / numThreads))

        # Compute the fraction of that maximum
        gridDimx = int(
            min(math.ceil(maxNumBlocks * params["grid_fraction"]), maxNumBlocks)
        )

        # max threads needs to always be >= numThreads and smaller than 1024.
        # This is a constraint provided by the vendors.
        # So here we compute the value:
        max_threads_range = 1024 - numThreads
        max_thread_value = numThreads + int(
            min(math.ceil(max_threads_range * params["max_threads"]), max_threads_range)
        )

        derived_config = {
            "block": {"x": numThreads, "y": 1, "z": 1},
            "grid": {"x": gridDimx, "y": 1, "z": 1},
            "shared_mem": 0,
            "specialize": True,
            "max_threads": max_thread_value,
            "set_launch_bounds": True,
            "specialize_dims": True,
        }

        return ExperimentConfiguration.from_dict(derived_config)

    def constraints(self, params):
        return True

    def baseline(self) -> ExperimentConfiguration:
        """
        Baseline configuration: replay with the recorded launch dims and shared memory.
        """
        return ExperimentConfiguration.from_dict(
            {
                "block": {
                    "x": self.recorded_kernel.block_dim.x,
                    "y": self.recorded_kernel.block_dim.y,
                    "z": self.recorded_kernel.block_dim.z,
                },
                "grid": {
                    "x": self.recorded_kernel.grid_dim.x,
                    "y": self.recorded_kernel.grid_dim.y,
                    "z": self.recorded_kernel.grid_dim.z,
                },
                "shared_mem": self.recorded_kernel.shared_mem,
            }
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    """
    Parse CLI arguments.

    Required:
      --record-db   Path to the recording database (or recording JSON, depending on your setup).
      --record-id   Identifier of the recorded kernel instance within the recording.
    """
    p = argparse.ArgumentParser(
        prog="mneme-tune-example",
        description="Run Optuna tuning on a Mneme recorded kernel.",
    )
    p.add_argument(
        "--record-db",
        required=True,
        help="Path to the recorded execution database/file.",
    )
    p.add_argument(
        "--record-id",
        required=True,
        help="Kernel record identifier within the recorded execution.",
    )
    return p.parse_args(argv)


def tune(executor, record_db, record_id):
    recorded_kernel = RecordedExecution.from_json(record_db)
    kernel = recorded_kernel[record_id]
    space = EntireSpace(kernel)

    # Baseline execution (verify correctness first).
    baseline_config = space.baseline()
    print(json.dumps(baseline_config.to_dict()))

    baseline_result = executor.evaluate(baseline_config)
    if not baseline_result.verified:
        print("Cannot verify baseline execution exiting")
        return 1

    baseline_time = statistics.mean(baseline_result.exec_time)
    print(f"Average baseline time {baseline_time}, {baseline_result.exec_time}")

    start = time.time()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
    )

    SS = OptunaSamplingStrategy(space, study, 200)

    for i, (config, ctrial) in enumerate(SS):
        if not config.is_valid():
            continue

        val = executor.evaluate(config)
        if val.verified:
            avg_time = statistics.mean(val.exec_time)
            speedup = baseline_time / avg_time

            ctrial.set_user_attr("mneme.config", config.to_dict())
            ctrial.set_user_attr("mneme.result", val.to_dict())

            study.tell(ctrial, speedup)
            print(
                f"\tExperiment {i+1} with options MaxThreads:{config.max_threads} GridX: {config.grid.x} BlockX: {config.block.x} shows speedup over  base line : {speedup} and total time: {avg_time}"
            )
        else:
            study.tell(ctrial, (1 << 64) - 1)
            print(i, config.hash(), f"Experiment failed with {val.error}")

    end = time.time()
    best = study.best_trial
    print(f"Optimal speedup is {study.best_trial.value}")
    print(
        f"Tuning time was {end - start} to perform 200 samples, each sample was executed {executor.iterations+1} times"
    )
    return 0


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    executor = AsyncReplayExecutor(
        record_db=args.record_db,
        record_id=args.record_id,
        iterations=5,
        results_db_dir="./",
        num_workers=1,
    )
    try:
        return tune(executor, args.record_db, args.record_id)
    except KeyboardInterrupt:
        print("Interrupted by user")
        raise
    except Exception as e:
        print("Fatal error during tuning:")
        print(f"{type(e).__name__}: {e}")
        raise
    finally:
        executor.shutdown()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
