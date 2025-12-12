import json
import sys
import time
import statistics

import optuna
from mneme.async_executor import AsyncReplayExecutor
from mneme.mneme_types import ExperimentConfiguration
from mneme.recorded_execution import RecordedExecution
from mneme.tuning.sample_strategy import (
    OptunaSamplingStrategy,
)
from mneme.tuning.search_space import (
    BoolParam,
    CategoricalParam,
    IntRangeParam,
    PipelineParam,
    RealRangeParam,
    SearchSpace,
)


class EntireSpace(SearchSpace):
    def __init__(self, recorded_kernel: RecordedExecution.KernelInstance):
        self.grid_dim_x = recorded_kernel.grid_dim.x
        self.grid_dim_y = recorded_kernel.grid_dim.y
        self.grid_dim_z = recorded_kernel.grid_dim.z
        self.block_dim_x = recorded_kernel.block_dim.x
        self.block_dim_y = recorded_kernel.block_dim.y
        self.block_dim_z = recorded_kernel.block_dim.z
        print(self.grid_dim_x)
        print(self.grid_dim_y)
        print(self.grid_dim_z)
        print(self.block_dim_x)
        print(self.block_dim_x)
        print(self.block_dim_x)
        self.shared_mem = recorded_kernel.shared_mem
        self._search_space = {
            "block_dim_x": IntRangeParam("block_dim_x", low=64, high=1024, step=64),
            "specialize": BoolParam("specialize"),
            "specialize_dims": BoolParam("specialize_dims"),
            "max_threads_fraction": RealRangeParam(
                "max_threads_fraction", low=0.0, high=1.0
            ),
            "min_blocks_per_sm": IntRangeParam("min_blocks_per_sm", low=1, high=8),
            "set_launch_bounds": BoolParam("set_launch_bounds"),
            "codegen_opt": IntRangeParam("codegen_opt", low=1, high=3, step=1),
            "codegen_method": CategoricalParam("codegen_method", ["serial"]),
            "passes": PipelineParam("passes", 120),
        }

    def dimensions(self):
        return self._search_space

    def derived(self, params) -> ExperimentConfiguration:
        derived_config = {
            "block": {
                "x": params["block_dim_x"],
                "y": self.block_dim_y,
                "z": self.block_dim_z,
            },
            "grid": {"x": self.grid_dim_x, "y": self.grid_dim_y, "z": self.grid_dim_z},
            "shared_mem": self.shared_mem,
            "specialize": params["specialize"],
            "specialize_dims": params["specialize_dims"],
            "min_blocks_per_sm": params["min_blocks_per_sm"],
            "set_launch_bounds": params["set_launch_bounds"],
            "max_threads": params["max_threads_fraction"],
            "codegen_opt": params["codegen_opt"],
            "codegen_method": params["codegen_method"],
            "passes": params["passes"],
        }
        if not params["set_launch_bounds"]:
            return ExperimentConfiguration.from_dict(derived_config)

        # NOTE: max_threads value is a real number. In reality max threads can take values
        # from "block_dim_x*block_dim_y*block_dim_z" until 1024 (1024 is the maximum threads
        # within a block in GPUs both (AMD and NVIDIA)
        proper_range = (
            params["block_dim_x"]
            + round(
                (1024 - params["block_dim_x"]) * params["max_threads_fraction"] / 64
            )
            * 64
        )
        derived_config["max_threads"] = proper_range

        return ExperimentConfiguration.from_dict(derived_config)

    def constraints(self, params):
        # No constraint for now
        return True

    def baseline(self) -> ExperimentConfiguration:
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


executor = AsyncReplayExecutor(
    record_db=sys.argv[1],
    record_id=sys.argv[2],
    iterations=5,
    results_db_dir="./results",
    num_workers=4,
)

recorded_kernel = RecordedExecution.from_json(sys.argv[1])
kernel = recorded_kernel[sys.argv[2]]
space = EntireSpace(kernel)

# I have an async executor, I will generate a baseline and execute it.
baseline_config = space.baseline()
print(json.dumps(baseline_config.to_dict()))
baseline_result = executor.evaluate(baseline_config)
if not baseline_result.verified:
    print("Cannot verify baseline execution exiting")
    sys.exit(0)
baseline_time = statistics.mean(baseline_result.exec_time)
print(f"Average baseline time {baseline_time}, {baseline_result.exec_time}")


study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.RandomSampler(),
)
SS = OptunaSamplingStrategy(space, study, 100)
pending = []
start = time.time()
for i, (config, ctrial) in enumerate(SS):
    if not config.is_valid():
        study.tell(ctrial, (1 << 64) - 1)
        continue

    val = executor.evaluate(config)
    if val.verified:
        print("ID:", id(ctrial))
        avg_time = statistics.mean(val.exec_time)
        speedup = baseline_time / avg_time
        ctrial.set_user_attr("mneme.config", config.to_dict())
        ctrial.set_user_attr("mneme.result", val.to_dict())
        study.tell(ctrial, speedup)
        print(
            f" Experiment {i} with hash:{config.hash()} has speedup of {speedup} and total time is {avg_time}"
        )
    else:
        study.tell(ctrial, (1 << 64) - 1)
        print(i, config.hash(), f"Experiment failed with {val.error}")

best = study.best_trial
print(json.dumps(best.user_attrs.get("mneme.config"), indent=2))
print(json.dumps(best.user_attrs.get("mneme.result"), indent=2))
end = time.time()
print("total process time is ", end - start)
executor.shutdown()
