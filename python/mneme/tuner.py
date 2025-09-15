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
    set_device,
)
from mneme.experiment import Experiment
from mneme.llvm.utils import get_profile_library
from mneme.logging import logger
from mneme.pipeline import PipelineManager
from mneme.replay_executor import BaseExecutor, TuneWorker
from mneme.tuner_optuna import run_optuna_tune


# Handle a single persistent subprocess and its monitor thread
class TuneWorkerHandle:
    def __init__(
        self,
        idx,
        request_q,
        response_q,
        db: str,
        record_id: str,
        device_id: int,
        prune: bool,
        internalize: bool,
        codegen_opt: int,
        rtc: bool,
        iterations: int,
        db_dir: str,
        suffix: str,
    ):

        self.idx = idx
        self.request_q = request_q
        self.response_q = response_q
        self.shutdown_event = threading.Event()
        self.process = None
        self.current = None  # (req_id, cfg)
        # We need these for the worker
        # The Record Database
        self.db = db
        # The id of the dynamic kernel to replay
        self.record_id = record_id
        # Device to run on
        self.device_id = device_id
        self.prune = prune
        self.internalize = internalize
        self.codegen_opt = codegen_opt
        self.rtc = rtc
        self.iterations = iterations
        self.db_dir = db_dir
        self.suffix = suffix

        # Start the subprocess and its monitor thread
        self._state = None
        self._spawn_process()
        self.monitor_thread = threading.Thread(target=self._monitor)
        self.monitor_thread.start()
        self._lock = threading.Lock()

    def _spawn_process(self):
        # Launch a fresh replay subprocess
        self._state = Event()
        self.process = Process(
            target=TuneWorker.run,
            args=(
                self.request_q,
                self.response_q,
                self.db,
                self.record_id,
                self.device_id,
                self.prune,
                self.internalize,
                self.codegen_opt,
                self.rtc,
                self.iterations,
                self.db_dir,
                self.suffix,
                self._state,
            ),
            daemon=False,
        )
        self.process.start()
        logger.debug(f"Worker {self.idx} started (pid {self.process.pid})")

    def _monitor(self):
        while True:
            # Wait for subprocess to exit
            self.process.join()
            if not self._state.is_set():
                logger.debug("[TuneWorkerHandle] Event has not been set")
                if self.current is not None:
                    req = self.current
                    req["payload"] = "exit"
                    req["llvm_ir"] = ""
                    req["exp_id"] = self.current["exp_id"]
                    self.response_q.put(req)
                else:
                    req = dict()
                    req["payload"] = "exit"
                    self.response_q.put(req)
                return

            exitcode = self.process.exitcode

            ## If shutdown requested, exit monitor
            if self.shutdown_event.is_set():
                logger.debug(
                    f"[TuneWorkerHandle] Monitor for worker {self.idx} shutting down."
                )
                break

            with self._lock:
                # Non-zero exit = crash
                if exitcode != 0:
                    logger.warning(
                        f"[TuneWorkerHandle] Worker {self.idx} crashed with exit code {exitcode}. Respawning..."
                    )

                    if self.current is not None:
                        req = self.current
                        req["data"]["failed"] = exitcode
                        req["payload"] = "result"
                        req["llvm_ir"] = ""
                        req["exp_id"] = self.current["exp_id"]
                        logger.warning(
                            f"[TuneWorkerHandle]: {self.idx} An experiment failed received {exitcode} for experiment {req['exp_id']}"
                        )

                        self.current = None
                        self.response_q.put(req)
                    self._spawn_process()
                    continue

        logger.debug(f"[TuneWorkerHandle]: {self.idx} Monitor thread ended.")

    def assign(self, experiment):
        with self._lock:
            # Track and send config
            self.current = copy.deepcopy(experiment)
            logger.debug(
                f"[TuneWorkerHandle]:{self.idx} Responsible for experiment with id: {self.current['exp_id']}"
            )
            self.request_q.put(experiment)

    def shutdown_process(self):  # Signal monitor thread
        self.shutdown_event.set()
        # Tell worker to exit cleanly
        logger.debug(f"[TuneWorkerHandle]:{self.idx} Shutting down Tuner")
        self.request_q.put({"payload": "terminate"})

    def join_monitor(self):
        self.monitor_thread.join()
        logger.debug(f"[TuneWorkerHandle]:{self.idx} Worker shutdown")


class ReplayTuner(BaseExecutor):
    @staticmethod
    def set_cli_args(parser):
        parser.add_argument(
            "--db-dir",
            required=True,
            default=None,
            help="Directory to store the collected data to",
        )

        parser.add_argument(
            "--suffix",
            required=True,
            default=None,
            help="Suffix of the database file (e.g. <args.db_dir><static_hash><dynamic_hash><suffix>.csv)",
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
        self.db_dir = kwargs.pop("db_dir")
        self.suffix = kwargs.pop("suffix")
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

    def create_experiments(self, pipeline, db):
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
                rtc=self.rtc,
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

        min_blocks_per_sm = [
            i for i in range(0, int(math.floor(1024 / max_threads)) + 1)
        ]

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
                    rtc=self.rtc,
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
    def run(cli_args):
        kwargs = vars(cli_args)
        tuner = kwargs.pop("tuner")
        kwargs.pop("command")
        kwargs.pop("func")
        executor = ReplayTuner(**kwargs)
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

        suffix = f"{executor.specialize}.{executor.prune}.{executor.internalize}.{executor.rtc}.{executor.codegen_opt}.{executor.sampler}.{executor.seed}"

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
                executor.rtc,
                executor._iterations,
                executor.db_dir,
                suffix,
            )
            for i in range(executor.num_workers)
        ]

        db = MnemeDB(
            executor.db_dir,
            executor.kernel_descr.static_hash,
            executor.kernel_descr.dynamic_hash,
            suffix,
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
                print(
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

        suffix = f"{executor.specialize}.{executor.prune}.{executor.internalize}.{executor.rtc}.{executor.codegen_opt}.{executor.seed}"
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
                executor.rtc,
                executor._iterations,
                executor.db_dir,
                suffix,
            )
            for i in range(executor.num_workers)
        ]

        passes = executor.LLVMPassManager.generate(
            executor.num_trials, executor.mean_size, 33, True, executor.seed
        )

        total_experiments = []
        default_pipelines = [
            "default<O1>",
            "default<O2>",
            "default<O3>",
            "default<Os>",
            "default<Oz>",
        ]

        db = MnemeDB(
            executor.db_dir,
            executor.kernel_descr.static_hash,
            executor.kernel_descr.dynamic_hash,
            executor.suffix,
        ).open()

        logger.info(f"Database contains {len(db)} experiments")

        for p in passes:
            total_experiments += executor.create_experiments(
                executor.LLVMPassManager.to_string(p) + ",globaldce", db
            )

        # Schedule happens in reverse order (pop) thus we want default pipelines to rin first.
        for p in default_pipelines:
            total_experiments += executor.create_experiments(p + ",globaldce", db)

        root_ir = executor.link_ir()
        orig = db.save_ir(root_ir, "orig")
        logger.info(f"Original IR is at: {orig}")

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
