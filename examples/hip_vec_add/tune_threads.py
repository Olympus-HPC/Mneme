import json
import logging
import sys
import time
from typing import Union

import optuna
from mneme.async_executor import AsyncReplayExecutor
from mneme.experiment import Experiment
from mneme.logging import logger
from mneme.logging import logger as replay_logger
from mneme.recorded_execution import RecordedExecution
from mneme.tuning.reducers import ArgMinReducer
from mneme.tuning.sample_strategy import (
    ExhaustiveSamplingStrategy,
    OptunaSamplingStrategy,
    RandomSamplingStrategy,
)
from mneme.tuning.search_space import (
    BoolParam,
    CategoricalParam,
    FixedParam,
    IntRangeParam,
    PipelineParam,
    RealRangeParam,
    SearchSpace,
)
from mneme.tuning.tuner_node import TunerNode

_LEVELS = {
    "critical": logging.CRITICAL,
    "warn": logging.WARNING,  # accept 'warn' per your spec
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


class EntireSpace(SearchSpace):
    def __init__(self, recorded_kernel: RecordedExecution.KernelInstance):
        self.grid_dim_x = recorded_kernel.grid_dim.x
        self.grid_dim_y = recorded_kernel.grid_dim.y
        self.grid_dim_z = recorded_kernel.grid_dim.z
        self.block_dim_x = recorded_kernel.block_dim.x
        self.block_dim_y = recorded_kernel.block_dim.y
        self.block_dim_z = recorded_kernel.block_dim.z
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

    def derived(self, params):
        derived_config = {
            "block_dim_x": params["block_dim_x"],
            "block_dim_y": self.block_dim_y,
            "block_dim_z": self.block_dim_z,
            "grid_dim_x": self.grid_dim_x,
            "grid_dim_y": self.grid_dim_y,
            "grid_dim_z": self.grid_dim_z,
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
            return derived_config

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

        return derived_config

    def constraints(self, params):
        # No constraint for now
        return True


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


configure_replay_logging("debug")

# study = optuna.create_study(
#    direction="minimize",  # or "maximize"
#    sampler=optuna.samplers.TPESampler(),
# )

executor = AsyncReplayExecutor(
    record_db=sys.argv[1],
    record_id=sys.argv[2],
    iterations=5,
    results_db_dir="./results",
    num_workers=4,
)

recorded_kernel = RecordedExecution.from_json(sys.argv[1])
kernel = recorded_kernel[sys.argv[2]]
population_size = 1000
study = optuna.create_study(
    storage="sqlite:///example_study.db",
    study_name="demo_sqlite",
    direction="minimize",
    # sampler=optuna.samplers.RandomSampler(),
    # sampler=optuna.samplers.NSGAIISampler(population_size=population_size),
    sampler=optuna.samplers.TPESampler(),
    load_if_exists=True,
)

space = EntireSpace(kernel)
SS = OptunaSamplingStrategy(space, study, 1000)
pending = []
start = time.time()
if sys.argv[3] != "parallel":
    for i, config in enumerate(SS):
        s = config["parameters"]
        ctrial = config["__optuna_trial__"]
        exp = Experiment.from_dict(**s)
        if not exp.is_valid():
            study.tell(config["__optuna_trial__"], (1 << 64) - 1)
            logger.debug(f"Skipping {i}")
            continue

        # Since many values are depend to each other and in practice
        # will not modify the execution we 'ground' them. This will allow
        # later the DB to pick the previous execution and skip the current one.
        exp.ground()
        val = executor.evaluate(exp)
        study.tell(config["__optuna_trial__"], val["data"]["exec_time_median"])
        print(i, s, val["data"]["exec_time_median"])
else:
    for i, config in enumerate(SS):
        s = config["parameters"]
        ctrial = config["__optuna_trial__"]
        exp = Experiment.from_dict(**s)
        if not exp.is_valid():
            study.tell(config["__optuna_trial__"], (1 << 64) - 1)
            logger.debug(f"Skipping {i}")
            continue

        # Since many values are depend to each other and in practice
        # will not modify the execution we 'ground' them. This will allow
        # later the DB to pick the previous execution and skip the current one.
        exp.ground()

        val = executor.submit(exp.to_dict())
        pending.append((config["__optuna_trial__"], val))

    for i, p in enumerate(pending):
        ctrial, future = p
        res = future.result()
        if res.get("payload", None) != "result":
            raise RuntimeError(f"Unexpected worker message: {json.dumps(res)}")
        study.tell(ctrial, res["data"]["exec_time_median"])
        print(res["exp_id"], res["data"]["exec_time_median"])
    pending.clear()

end = time.time()
print("total process time is ", end - start)
executor.shutdown()
