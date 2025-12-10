import argparse
import copy
import json
import math
import os
import sys
import threading
from multiprocessing import Event, Process, Queue

from mneme.db import MnemeDB
from mneme.device import (
    DeviceModule,
    dim3,
    get_device_arch,
    get_device_count,
    get_max_blocks_per_sm,
    set_device,
)
from mneme.experiment import Experiment
from mneme.llvm.utils import get_profile_library
from mneme.logging import logger
from mneme.pipeline import PipelineManager
from mneme.replay_executor import BaseExecutor, TuneWorker
from mneme.tuner_optuna import run_optuna_tune


# `ReplayTuner` coordinates the end-to-end tuning workflow for a single
# recorded kernel. It builds experiment configurations, launches a pool of
# persistent `TuneWorkerHandle` subprocesses, schedules experiments across
# them, and records results into the Mneme database.
#
# Responsibilities:
#   • Parse CLI arguments controlling sampling strategy, pipeline length,
#     specialization options, and number of workers/GPUs to use.
#   • Generate experiment objects (random or optuna-driven) describing
#     codegen options, launch-bound settings, and LLVM pass pipelines.
#   • Dispatch experiments to workers and collect completed results from a
#     shared queue, including handling failures or worker restarts.
#   • Record all outcomes (IR, metadata, performance) into the results DB.
#
# In short, `ReplayTuner` is the top-level orchestrator that turns a recorded
# dynamic GPU kernel into a large set of experiments and executes them
# efficiently across all available GPUs.
class ReplayTuner(BaseExecutor):
    @staticmethod
    def set_cli_args(parser):
        parser.add_argument(
            "--results-db-dir",
            required=True,
            default=None,
            help="Directory to store the collected data to",
        )

        parser.add_argument(
            "--specialize",
            default=False,
            required=False,
            action=argparse.BooleanOptionalAction,
            dest="specialize",
            help="Apply argument and dim specialization on the kernel",
        )

        parser.add_argument(
            "--seed",
            required=False,
            type=int,
            default=0,
            dest="seed",
            help="Seed number to initialize the random pass generation module",
        )

        parser.add_argument(
            "--average-pipeline-length",
            "-p",
            required=False,
            dest="average_pipeline_length",
            type=int,
            default=50,
            help="The average length of the generated pipeline",
        )

        parser.add_argument(
            "--num-trials",
            "-n",
            required=True,
            type=int,
            default=100,
            help="The number of random pipelines to generate",
        )

        parser.add_argument(
            "--num-workers",
            "-w",
            required=False,
            dest="num_workers",
            type=int,
            help="The number of random workers to use. The value is always ceiled to the available number of GPUs",
        )

        parser.add_argument(
            "--tuner-type",
            "-t",
            required=False,
            dest="tuner",
            choices=["random", "optuna"],
            default="random",
            help="Which tuner to use to sample the space",
        )

        parser.add_argument(
            "--search-sampler",
            "-s",
            required=False,
            dest="sampler",
            choices=[
                "RandomSampler",
                "TPESampler",
                #                "GPSampler",
                "NSGAIISampler",
                "QMCSampler",
            ],
            help="Which sampler to use, used only by optuna",
        )

        parser.set_defaults(func=ReplayTuner.run)

    def __init__(self, *args, **kwargs):
        self.specialize = kwargs.pop("specialize", False)
        self.results_db_dir = kwargs.pop("results_db_dir")
        self.num_trials = kwargs.pop("num_trials")
        self.seed = kwargs.pop("seed", 0)
        self.mean_size = kwargs.pop("average_pipeline_length", 50)
        self.num_devices = get_device_count()
        self.num_workers = kwargs.pop("num_workers")
        self.sampler = kwargs.pop("sampler")
        if self.num_workers is None or self.num_workers > self.num_devices:
            self.num_workers = self.num_devices
        super().__init__(**kwargs)
        logger.debug(
            f"Detected {self.num_devices} gpu devices, we will use {self.num_workers} workers"
        )
        self.LLVMPassManager = PipelineManager()

    # Generate all experiment configurations for a given LLVM pass pipeline.
    # This expands the pipeline into a full set of tuning candidates by
    # enumerating:
    #   • specialization on/off
    #   • launch-bound settings (max threads per block and min blocks per SM)
    #
    # For each combination, an `Experiment` object is constructed and filtered
    # against the results database so we only execute experiments that have not
    # already been replayed. The returned list therefore represents the unique
    # set of experiments that still need to be evaluated for this pipeline.
    def generate_experiments_for_pipeline(self, pipeline, db):
        # NOTE: Some of the fields of the experiment are misguiding.
        # For example, the 'internalize' field is ignored, cause this
        # happens earlier regardless of the value of the field itself.
        # it jus exists to setup properly tracking of experiments.
        experiments = []
        for spec in [True, False]:
            exp = Experiment(
                specialize=spec,
                max_threads=0,
                min_blocks_per_sm=0,
                specialize_dims=self.specialize,
                passes=pipeline,
                prune=self.prune,
                internalize=self.internalize,
                codegen_opt=self.codegen_opt,
                codegen_method=self.codegen_method,
                device_arch=self.device_arch,
            )
            if not db.should_execute(exp):
                logger.debug(
                    f"Skipping experiment {str(exp.hash())}, already in replayed"
                )

                continue
            experiments.append(exp)

        max_threads = int(
            self.kernel_descr.block_dim.x
            * self.kernel_descr.block_dim.y
            * self.kernel_descr.block_dim.z
        )

        min_blocks_per_sm = [i for i in range(0, get_max_blocks_per_sm() + 1)]

        # 0 indicates do not set launch bounds
        for mb in min_blocks_per_sm:
            for spec in [True, False]:
                exp = Experiment(
                    specialize=spec,
                    max_threads=max_threads,
                    min_blocks_per_sm=mb,
                    specialize_dims=self.specialize,
                    passes=pipeline,
                    prune=self.prune,
                    internalize=self.internalize,
                    codegen_opt=self.codegen_opt,
                    codegen_method=self.codegen_method,
                    device_arch=self.device_arch,
                )
                if not db.should_execute(exp):
                    logger.debug(
                        f"Skipping experiment {str(exp.hash())}, already in replayed"
                    )

                    continue
                experiments.append(exp)

        return experiments

    @staticmethod
    def run(cli_args, verbosity):
        kwargs = vars(cli_args)
        tuner = kwargs.pop("tuner")
        kwargs.pop("command")
        kwargs.pop("func")
        executor = ReplayTuner(**kwargs)
        # NOTE: This needs to take place early, before launching the workers. Otherwise we get issues with dlopen and profiling executions
        os.environ["ROCP_TOOL_LIBRARIES"] = (
            get_profile_library()
        )  # /path/to/libmneme_profile.so
        if tuner == "random":
            kwargs.pop("sampler")
            return ReplayTuner.run_random(executor)
        elif tuner == "optuna":
            return ReplayTuner.run_optuna(executor)

    @staticmethod
    def run_optuna(executor):
        completed_jobs_q = Queue()

        workers = [
            TuneWorkerHandle(
                i,
                Queue(),
                completed_jobs_q,
                executor.db,
                executor.record_id,
                i,
                executor.prune,
                executor.internalize,
                executor.codegen_opt,
                executor.codegen_method,
                executor._iterations,
                executor.results_db_dir,
            )
            for i in range(executor.num_workers)
        ]

        db = MnemeDB(
            executor.results_db_dir,
            executor.kernel_descr.static_hash,
            executor.kernel_descr.dynamic_hash,
        ).open()

        run_optuna_tune(
            ReplayTuner.execute_list_of_experiments,
            db,
            executor,
            workers,
            completed_jobs_q,
            executor.num_trials,
            executor.sampler,
            executor.seed,
        )

        for w in workers:
            w.shutdown_process()
        for w in workers:
            w.join_monitor()

        return 0

    @staticmethod
    def execute_list_of_experiments(
        orig, total_experiments, workers, completed_jobs_q, db, exp_id, start_id=0
    ):
        def schedule_job(db, pending_experiments, exp_id, worker):
            while True:
                if len(pending_experiments) > 0:
                    e = pending_experiments.pop()
                    worker.assign(
                        {
                            "payload": "process",
                            "data": e.to_dict(),
                            "exp_id": exp_id + 1,
                        }
                    )
                    return (e, exp_id + 1)
                else:
                    return None

        count = start_id
        in_flight = {}
        for w in workers:
            vals = schedule_job(db, total_experiments, exp_id, w)
            if vals is None:
                continue
            exp, exp_id = vals[0], vals[1]
            in_flight[exp_id] = (exp, w)

        while len(in_flight) != 0:
            res = completed_jobs_q.get()
            if res["payload"] == "result":
                count += 1
                done_exp_id = res["exp_id"]
                exp1, worker = in_flight.pop(done_exp_id)
                exp2 = Experiment.from_dict(**res["data"])
                exp2.start_id = done_exp_id
                exp2.commit_id = count
                if exp1.hash() != exp2.hash():
                    raise RuntimeError(
                        f"Received experiment should have same hash with workers experiment {exp1.hash()} {exp2.hash()}"
                    )
                if not exp2.failed:
                    if "llvm_ir" not in res:
                        raise RuntimeError(
                            "Expected llvm ir to exist on non failed experiment"
                        )
                db.add(orig, res["llvm_ir"], exp2)
                logger.debug(
                    f"Worker {worker.idx} Done with {done_exp_id} had {exp2.failed} and took {exp2.exec_time}"
                )

                if len(total_experiments) != 0:
                    vals = schedule_job(db, total_experiments, exp_id, worker)
                    if vals is None:
                        continue

                    exp, exp_id = vals[0], vals[1]
                    in_flight[exp_id] = (exp, worker)
            elif res["payload"] == "exit":
                logger.debug("Exiting cause we received exit request")
                return -1
            else:
                logger.debug("Exiting cause we received 'unknown payload' request")
                return -1

            sys.stdout.flush()
        return 0

    @staticmethod
    def run_random(executor):
        completed_jobs_q = Queue()
        exp_id = 0

        workers = [
            TuneWorkerHandle(
                i,
                Queue(),
                completed_jobs_q,
                executor.db,
                executor.record_id,
                i,
                executor.prune,
                executor.internalize,
                executor.codegen_opt,
                executor.codegen_method,
                executor._iterations,
                executor.results_db_dir,
            )
            for i in range(executor.num_workers)
        ]

        passes = executor.LLVMPassManager.generate(
            executor.num_trials, executor.mean_size, 33, True, executor.seed
        )

        total_experiments = []
        default_pipelines = [
            "default<O3>",
            "default<O2>",
            "default<O1>",
            "default<Os>",
            "default<Oz>",
        ]

        db = MnemeDB(
            executor.results_db_dir,
            executor.kernel_descr.static_hash,
            executor.kernel_descr.dynamic_hash,
        ).open()

        logger.info(f"Database contains {len(db)} experiments")

        # NOTE: In all pipelines we always append "globaldce" we have observed this
        # reduces the size of the generated binary and can lead into issues. We perform
        # this for apple to apple comparisons across sizes.
        for p in passes:
            total_experiments += executor.generate_experiments_for_pipeline(
                executor.LLVMPassManager.to_string(p) + ",globaldce", db
            )

        # Schedule happens in reverse order (pop) thus we want default pipelines to rin first.
        requested_num_exp = len(total_experiments)
        for p in default_pipelines:
            total_experiments += executor.generate_experiments_for_pipeline(
                p + ",globaldce", db
            )

        root_ir = executor.link_ir()
        orig = db.save_ir(root_ir, "orig")
        logger.info(f"Original IR is at: {orig}")

        print(
            f"Will execute {len(total_experiments) - requested_num_exp} corresponding to possible baselines"
        )
        print(
            f"Will execute {requested_num_exp} to explore pipelines and several options"
        )

        ret = ReplayTuner.execute_list_of_experiments(
            orig, total_experiments, workers, completed_jobs_q, db, 0
        )

        logger.debug(f"Mneme is done with code {ret}")

        logger.debug("Requesting workers to shutdown")
        for w in workers:
            w.shutdown_process()

        logger.debug("Waiting for shadow monitors to terminate")
        for w in workers:
            w.join_monitor()

        return ret
