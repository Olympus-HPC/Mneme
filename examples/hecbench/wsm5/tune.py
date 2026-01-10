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
import statistics
import sys
import time

from mneme.async_executor import AsyncReplayExecutor
from mneme.mneme_types import ExperimentConfiguration
from mneme.recorded_execution import RecordedExecution
from mneme.tuning.sample_strategy import ExhaustiveSamplingStrategy
from mneme.tuning.search_space import (
    SearchSpace,
    BoolParam,
)


class EntireSpace(SearchSpace):
    """
    Example SearchSpace for tuning a recorded kernel.

    This SearchSpace:
      - Uses the recorded grid/block dims as fixed reference values where needed.
      - Exposes a set of tunable parameters (block_dim_x, specialization toggles,
        min_blocks_per_sm, launch bounds, codegen choices, and pass pipeline).
      - Produces an ExperimentConfiguration via derived(params).
    """

    def __init__(self, recorded_kernel: RecordedExecution.KernelInstance):
        self.grid_dim_x = recorded_kernel.grid_dim.x
        self.grid_dim_y = recorded_kernel.grid_dim.y
        self.grid_dim_z = recorded_kernel.grid_dim.z

        self.block_dim_x = recorded_kernel.block_dim.x
        self.block_dim_y = recorded_kernel.block_dim.y
        self.block_dim_z = recorded_kernel.block_dim.z

        self.shared_mem = recorded_kernel.shared_mem

        self._search_space = {
            "specialize": BoolParam("specialize"),
            "set_launch_bounds": BoolParam("set_launch_bounds"),
            "specialize_dims": BoolParam("specialize_dims"),
        }

    def dimensions(self):
        return self._search_space

    def derived(self, params) -> ExperimentConfiguration:
        """
        Convert sampled parameters into a concrete ExperimentConfiguration.

        If launch bounds are enabled, this example maps max_threads_fraction into a
        proper max_threads integer in [block_dim_x, 1024] (rounded to multiples of 64).
        """
        derived_config = {
            "block": {
                "x": self.block_dim_x,
                "y": self.block_dim_y,
                "z": self.block_dim_z,
            },
            "grid": {"x": self.grid_dim_x, "y": self.grid_dim_y, "z": self.grid_dim_z},
            "shared_mem": self.shared_mem,
            "specialize": params["specialize"],
            "set_launch_bounds": params["set_launch_bounds"],
            "specialize_dims": params["specialize_dims"],
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
                    "x": self.block_dim_x,
                    "y": self.block_dim_y,
                    "z": self.block_dim_z,
                },
                "grid": {
                    "x": self.grid_dim_x,
                    "y": self.grid_dim_y,
                    "z": self.grid_dim_z,
                },
                "shared_mem": self.shared_mem,
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
    SS = ExhaustiveSamplingStrategy(space)

    start = time.time()
    results = []
    total_executed_experiments = 0
    for i, config in enumerate(SS):
        if not config.is_valid():
            continue

        val = executor.evaluate(config)
        if val.verified:
            avg_time = statistics.mean(val.exec_time)
            speedup = baseline_time / avg_time
            results.append((config, val))
            print(
                f"\tExperiment {i+1} with options specialize:{config.specialize} specialize_dims: {config.specialize_dims} launch_bounds: {config.specialize_dims} shows speedup over base line : {speedup} and total time: {avg_time}"
            )
        else:
            print(
                f"\tExperiment {i+1} with options specialize:{config.specialize} specialize_dims: {config.specialize_dims} launch_bounds: {config.specialize_dims} shows error {val.error}"
            )
        total_executed_experiments += 1

    end = time.time()
    print(f"Total num experiments performed {total_executed_experiments}")
    print(
        f"Total number of executions performed {total_executed_experiments * (executor.iterations + 1)}"
    )
    best = min(results, key=lambda x: sum(x[1].exec_time) / len(x[1].exec_time))

    total_gpu_kernel_time = sum(sum(x[1].exec_time) for x in results) / 1_000_000_000
    total_mid_end_comp_time = sum(x[1].opt_time for x in results)
    total_back_end_comp_time = sum(x[1].codegen_time for x in results)
    total_preprocess_comp_time = sum(x[1].preprocess_ir_time for x in results)
    total_count = 0
    for _, r in results:
        total_count += len(r.exec_time)
    total_verify_time = (total_gpu_kernel_time / total_count) * executor.iterations
    print(f"Time spend executing configs on GPU {total_gpu_kernel_time}")
    print(
        f"Time spend executing verification kernel configs on GPU {total_verify_time}"
    )
    print(f"Time spend in optimizing code (middle end)  {total_mid_end_comp_time}")
    print(f"Time spend in codegen code (back end)  {total_back_end_comp_time}")
    print(f"Time spend in preprocess IR {total_preprocess_comp_time}")
    total_tune_time = end - start
    total_useful_time = (
        total_preprocess_comp_time
        + total_mid_end_comp_time
        + total_back_end_comp_time
        + total_gpu_kernel_time
        + total_verify_time
    )
    overhead = (total_tune_time - total_useful_time) / total_tune_time
    print(
        f"Time spend in performing actions {total_useful_time} total tune time {total_tune_time} mneme overhead {overhead}"
    )

    avg_time = statistics.mean(best[1].exec_time)
    speedup = baseline_time / avg_time

    if len(results) <= 0:
        print("No valid experiment was executed, returning")
        return 1

    print(
        f"Best config has specialize: {best[0].specialize} and specilize_dims: {best[0].specialize_dims} and set launch bounds: {best[0].set_launch_bounds} shows speedup over base line : {speedup} and total time: {avg_time}"
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
