import argparse
import copy
import json
import math
import sys
import threading
from multiprocessing import Process, Queue

from mneme.db import MnemeDB
from mneme.device import (
    DeviceModule,
    dim3,
    get_device_arch,
    get_device_count,
    set_device,
)
from mneme.experiment import Experiment
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

        # Start the subprocess and its monitor thread
        self._spawn_process()
        self.monitor_thread = threading.Thread(target=self._monitor)
        self.monitor_thread.start()
        self._lock = threading.Lock()

    def _spawn_process(self):
        # Launch a fresh replay subprocess
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
            ),
            daemon=False,
        )
        self.process.start()
        print(f"[i] Worker {self.idx} started (pid {self.process.pid})")

    def _monitor(self):
        while True:
            # Wait for subprocess to exit
            self.process.join()
            exitcode = self.process.exitcode

            ## If shutdown requested, exit monitor
            if self.shutdown_event.is_set():
                print(f"[i] Monitor for worker {self.idx} shutting down.")
                break

            with self._lock:
                # Non-zero exit = crash
                if exitcode != 0:
                    print(
                        f"[!] Worker {self.idx} crashed with exit code {exitcode}. Respawning..."
                    )

                    if self.current is not None:
                        req = self.current
                        req["data"]["failed"] = exitcode
                        req["payload"] = "result"
                        req["llvm_ir"] = ""
                        req["exp_id"] = self.current["exp_id"]
                        print(
                            f"{self.idx} Failed Compiler pipeline, received {exitcode} for experiment {req['exp_id']}"
                        )
                        self.current = None
                        self.response_q.put(req)
                    else:
                        print("Current is None")
                    self._spawn_process()
                    continue

        print(f"[i] Monitor thread for worker {self.idx} ended.")

    def assign(self, experiment):
        with self._lock:
            # Track and send config
            self.current = copy.deepcopy(experiment)
            print(f"{self.idx} Was assigned experiment with {self.current['exp_id']}")
            self.request_q.put(experiment)

    def shutdown_process(self):  # Signal monitor thread
        self.shutdown_event.set()
        # Tell worker to exit cleanly
        print("Sending shutwdown to Q")
        self.request_q.put({"payload": "terminate"})

    def join_monitor(self):
        self.monitor_thread.join()
        print(f"[i] Worker {self.idx} fully shut down.")


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
                "GPSampler",
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
        print(
            f"Num devices are {self.num_devices} we will use {self.num_workers} workers"
        )
        self.LLVMPassManager = PipelineManager()

    def create_experiments(self, pipeline):
        # NOTE: Some of the fields of the experiment are misguiding.
        # For example, the 'internalize' field is ignored, cause this
        # happens earlier regardless of the value of the field itself.
        # it jus exists to setup properly tracking of experiments.
        experiments = [
            Experiment(
                specialize=self.specialize,
                max_threads=0,
                min_blocks_per_sm=0,
                specialize_dims=self.specialize,
                passes=self.LLVMPassManager.to_string(pipeline),
                prune=self.prune,
                internalize=self.internalize,
                codegen_opt=self.codegen_opt,
                rtc=self.rtc,
                device_arch=self.device_arch,
            )
        ]

        max_threads = int(
            self.kernel_descr.block_dim.x
            * self.kernel_descr.block_dim.y
            * self.kernel_descr.block_dim.z
        )

        min_blocks_per_sm = [
            i for i in range(0, int(math.ceil(1024 / max_threads)) + 1)
        ]

        # 0 indicates do not set launch bounds
        for mb in min_blocks_per_sm:
            experiments.append(
                Experiment(
                    specialize=self.specialize,
                    max_threads=max_threads,
                    min_blocks_per_sm=mb,
                    specialize_dims=self.specialize,
                    passes=self.LLVMPassManager.to_string(pipeline),
                    prune=self.prune,
                    internalize=self.internalize,
                    codegen_opt=self.codegen_opt,
                    rtc=self.rtc,
                    device_arch=self.device_arch,
                )
            )

        return experiments

    @staticmethod
    def run(cli_args):
        kwargs = vars(cli_args)
        tuner = kwargs.pop("tuner")
        kwargs.pop("command")
        kwargs.pop("func")
        executor = ReplayTuner(**kwargs)
        if tuner == "random":
            kwargs.pop("sampler")
            ReplayTuner.run_random(executor)
        elif tuner == "optuna":
            ReplayTuner.run_optuna(executor)

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
                executor.rtc,
                executor._iterations,
            )
            for i in range(executor.num_workers)
        ]

        db = MnemeDB(
            executor.db_dir,
            executor.kernel_descr.static_hash,
            executor.kernel_descr.dynamic_hash,
            executor.suffix,
        ).open()

        print("I am here")

        run_optuna_tune(
            db,
            executor,
            workers,
            completed_jobs_q,
            executor.num_trials,
            executor.sampler,
            executor.seed,
        )
        print("I am done")

        for w in workers:
            w.shutdown_process()
        for w in workers:
            w.join_monitor()

    @staticmethod
    def run_random(executor):
        def schedule_job(db, pending_experiments, exp_id, worker):
            while True:
                if len(pending_experiments) > 0:
                    e = pending_experiments.pop()
                    if not db.should_execute(e):
                        print(f"Skipping experiment {e.hash()}")
                        continue
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

        run_jobs_q = Queue()
        completed_jobs_q = Queue()
        exp_id = 0

        workers = [
            TuneWorkerHandle(
                i,
                run_jobs_q,
                completed_jobs_q,
                executor.db,
                executor.record_id,
                i,
                executor.prune,
                executor.internalize,
                executor.codegen_opt,
                executor.rtc,
                executor._iterations,
            )
            for i in range(executor.num_workers)
        ]

        passes = executor.LLVMPassManager.generate(
            executor.num_trials, executor.mean_size, 33, True, executor.seed
        )

        in_flight = {}
        total_experiments = []

        for p in passes:
            experiments = executor.create_experiments(p)
            for e in experiments:
                total_experiments.append(e)

        db = MnemeDB(
            executor.db_dir,
            executor.kernel_descr.static_hash,
            executor.kernel_descr.dynamic_hash,
            executor.suffix,
        ).open()

        root_ir = executor.link_ir()
        orig = db.save_ir(str(root_ir), "orig")

        print(f"Total experiments are {len(total_experiments)}")
        for w in workers:
            vals = schedule_job(db, total_experiments, exp_id, w)
            if vals is None:
                continue
            exp, exp_id = vals[0], vals[1]
            print("Experiment id is", exp_id)
            in_flight[exp_id] = (exp, w)

        while len(in_flight) != 0:
            res = completed_jobs_q.get()
            if res["payload"] == "result":
                exp1, worker = in_flight.pop(res["exp_id"])
                exp2 = Experiment.from_dict(**res["data"])
                if exp1.hash() != exp2.hash():
                    raise RuntimeError(
                        f"Received experiment should have same hash with workers experiment {exp1.hash()} {exp2.hash()}"
                    )
                exp2.dump()
                if not exp2.failed:
                    if "llvm_ir" not in res:
                        raise RuntimeError(
                            "Expected llvm ir to exist on non failed experiment"
                        )
                    final = db.save_ir(res["llvm_ir"], exp2.hash())
                    db.add(orig, final, exp2)

                if len(total_experiments) != 0:
                    vals = schedule_job(db, total_experiments, exp_id, worker)
                    if vals is None:
                        continue

                    exp, exp_id = vals[0], vals[1]
                    in_flight[exp_id] = (exp, worker)
            else:
                print("Unknown payload")
            sys.stdout.flush()

        for w in workers:
            w.shutdown_process()
        for w in workers:
            w.join_monitor()
