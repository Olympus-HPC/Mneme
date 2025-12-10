import argparse
import json
import os
import time
from datetime import datetime
from multiprocessing import Event, Queue
from typing import Tuple

from mneme.db import MnemeDB
from mneme.device import (
    DeviceModule,
    dim3,
    get_device_arch,
    get_device_count,
    set_device,
)
from mneme.experiment import Experiment
from mneme.fancy_out import PrettyTablePrinter
from mneme.llvm.buffer import MemBufferRef
from mneme.llvm.module import ModuleRef
from mneme.logging import logger
from mneme.page_manager import PageManagerRef
from mneme.pipeline import PipelineManager
from mneme.profile import init_profiler
from mneme.proteus import jit
from mneme.recorded_execution import MemStateRef, RecordedExecution
from mneme.transforms import transform
from mneme.utils import cond_gpu_time, cond_time


class BaseExecutor:
    @staticmethod
    def get_base_parser():
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "-rdb",
            "--record-database",
            dest="record_db",
            required=True,
            help="Path to Mneme JSON/db file",
        )

        parser.add_argument(
            "-record-id",
            "-rid",
            dest="record_id",
            required=True,
            help="Kernel ID to operate on",
        )

        parser.set_defaults()

        return parser

    def __init__(
        self,
        record_db: str = "",
        record_id: str = "",
        iterations: int = 3,
        device_id: int = 0,
    ):
        self.record_db = record_db
        self.record_id = record_id
        self.device_id = device_id
        logger.debug(
            f"BaseExecutor Got {self.record_db} and {self.record_id} and will run on device:{self.device_id}"
        )
        self.records = RecordedExecution.from_json(self.record_db)
        self.kernel_descr = self.records[self.record_id]
        self.device_arch = get_device_arch()
        self._epilogue = None
        self._prologue = None
        self._page_manager = None
        self._iterations = iterations
        self.num_devices = get_device_count()
        set_device(device_id)
        logger.debug(
            f"GPU Affinity of process was set to device:{self.device_id} out of {self.num_devices}"
        )

    def open(self):
        # Note the 'executor' allocates all resources and picks address space.
        self._page_manager = PageManagerRef(
            self.device_id, self.records.va_addr, self.records.va_size
        )
        self._prologue = self.kernel_descr.prologue.open()
        self._epilogue = self.kernel_descr.epilogue.open()
        return self

    @property
    def prologue(self):
        return self._prologue

    @property
    def epilogue(self):
        return self._epilogue

    def close(self):
        if self._epilogue is not None:
            self._epilogue.close()
            self._epilogue = None
        if self._prologue is not None:
            self._prologue.close()
            self._prologue = None
        if self._page_manager is not None:
            self._page_manager.close()
            self._page_manager = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def link_ir(self):
        return self.records.link_llvm_modules(prune=True, internalize=True)

    @cond_time("preprocess_ir_time")
    def _preprocess_ir(self, exp, llvm_ir):
        code_hash = self.kernel_descr.static_hash

        if exp.specialize:
            code_hash = jit.specialize_args(
                llvm_ir,
                code_hash,
                self.kernel_descr.kernel_name,
                self.prologue.args,
                self.prologue.num_args,
                self.kernel_descr.available_specializations,
            )

        if exp.specialize_dims:
            grid_dim = dim3(exp.grid_dim_x, exp.grid_dim_y, exp.grid_dim_z)
            block_dim = dim3(exp.block_dim_x, exp.block_dim_y, exp.block_dim_z)
            code_hash = jit.specialize_dims(
                llvm_ir,
                code_hash,
                self.kernel_descr.kernel_name,
                grid_dim,
                block_dim,
            )
        if exp.set_launch_bounds:
            code_hash = jit.set_launch_bounds(
                llvm_ir,
                code_hash,
                self.kernel_descr.kernel_name,
                exp.max_threads,
                exp.min_blocks_per_sm,
            )
        return code_hash, llvm_ir

    @cond_time("opt_time")
    def _optimize(self, exp, ir_module):
        jit.optimize(ir_module, self.device_arch, exp.passes, exp.codegen_opt)

    @cond_time("codegen_time")
    def _codegen(self, exp, ir_module):
        return jit.codegen_object(
            ir_module, self.device_arch, exp.codegen_method, exp.codegen_opt
        )

    @cond_gpu_time("exec_time")
    def _run_kernel(self, exp, kernel_name, device_func, iterations):
        grid_dim = dim3(exp.grid_dim_x, exp.grid_dim_y, exp.grid_dim_z)
        block_dim = dim3(exp.block_dim_x, exp.block_dim_y, exp.block_dim_z)
        return device_func.profile(
            grid_dim,
            block_dim,
            self._prologue._state,
            self._epilogue._state,
            exp.shared_mem,
            iterations,
        )

    def _build(
        self, exp: Experiment, ir_module: ModuleRef, track: bool
    ) -> MemBufferRef:
        self._preprocess_ir(exp, ir_module)
        self._optimize(exp, ir_module, profile=track)
        mem_buffer = self._codegen(exp, ir_module, profile=track)
        if track:
            exp.obj_size = mem_buffer.get_size()
        return mem_buffer

    def _run(self, exp: Experiment, mem_buffer: MemBufferRef, track: bool, iterations):
        with DeviceModule.from_MemBuffer(mem_buffer) as DeviceObj:
            device_func = DeviceObj.get_function(self.kernel_descr.kernel_name)
            self._run_kernel(
                exp,
                self.kernel_descr.kernel_name,
                device_func,
                iterations,
                profile=track,
            )
            if track:
                exp.reg_usage = device_func.reg_usage
                exp.const_mem = device_func.const_mem
                exp.local_mem = device_func.local_mem

    def _execute(
        self, exp: Experiment, ir_module: ModuleRef
    ) -> Tuple[Experiment, ModuleRef]:
        if self._prologue._state is None or self._epilogue._state is None:
            raise RuntimeError("States should never be none when executing a kernel")

        # NOTE: 1. First we need to verify.
        ver_mod = ir_module.clone()
        mem_buffer = self._build(exp, ver_mod, False)
        self._run(exp, mem_buffer, False, 1)
        exp.verified = self.prologue == self.epilogue

        # NOTE: 2. We apply a custom pass to delete all clang insered code.
        # It is hard to identify these cases, So we delete only things that have been attributed by clang
        ir_module = transform.remove_auto_initialize(ir_module.clone())
        # Done with verification. Moving to next stage

        # NOTE: 3. We build and run. We set tracking on and we always execute iterations +2,
        # to enalbe later computation of statistical metrics etc.
        mem_buffer = self._build(exp, ir_module, True)
        self._run(exp, mem_buffer, True, self._iterations + 2)
        exp.executed = True

        return exp, ir_module


class CLIExecutor(BaseExecutor):
    @staticmethod
    def set_cli_args(parser):
        parser.add_argument(
            "--results-db-dir",
            required=False,
            default=None,
            help="Directory to store the collected data to",
        )

        parser.add_argument(
            "--apply-increamentally",
            required=False,
            action=argparse.BooleanOptionalAction,
            dest="increamental",
            default=False,
            help="Apply passes one by one and increamentally build the pipelines. Useful for debugging",
        )

        parser.add_argument(
            "pipeline",
            help="Compilation pipeline of the kernel to execute",
        )

        parser.add_argument(
            "--specialize",
            default=False,
            required=False,
            action=argparse.BooleanOptionalAction,
            dest="specialize",
            help="Apply argument specialization on the kernel",
        )

        parser.add_argument(
            "--specialize-dims",
            "-sdims",
            dest="specialize_dims",
            default=False,
            required=False,
            action=argparse.BooleanOptionalAction,
            help="Specialize ThreadID.*, BlockDim.* and GridDim.* with constants and provide 'assume' instructions to llvm",
        )

        parser.add_argument(
            "--set-launch-bounds",
            "-slb",
            dest="set_launch_bounds",
            default=False,
            required=False,
            action=argparse.BooleanOptionalAction,
            help="Specialize ThreadID.*, BlockDim.* and GridDim.* with constants and provide 'assume' instructions to llvm",
        )

        parser.add_argument(
            "--max-threads",
            default=False,
            required=False,
            type=int,
            dest="max_threads",
            help="Set launch bound 'max_threads' of kernel to the executed number of threads",
        )

        parser.add_argument(
            "--min-threads-per-block",
            default=0,
            type=int,
            dest="min_blocks_per_sm",
            help="Set launch bound 'min_blocks_per_sm' of kernel to the provided value",
        )

        parser.add_argument(
            "-cm",
            "--codegen-method",
            dest="codegen_method",
            choices=["rtc", "serial", "parallel"],
            default="serial",
            help="Technology to use to lower to LLVM IR to a device object file instead of default Proteus Infrastructure",
        )

        parser.add_argument(
            "--codegen-opt",
            "-co",
            dest="codegen_opt",
            type=int,
            default=3,
            help="Optimization level to be used when generating machine code (back end optimizations)",
        )

        parser.add_argument(
            "--block-dim-x",
            "-bidx",
            dest="block_dim_x",
            type=int,
            default=None,
            help="Value of BlockDim.x during kernel replay, when omitted the recorded value is used",
        )

        parser.add_argument(
            "--block-dim-y",
            "-bidy",
            dest="block_dim_y",
            type=int,
            default=None,
            help="Value of BlockDim.y during kernel replay, when omitted the recorded value is used",
        )

        parser.add_argument(
            "--block-dim-z",
            "-bidz",
            dest="block_dim_z",
            type=int,
            default=None,
            help="Value of BlockDim.z during kernel replay, when omitted the recorded value is used",
        )

        parser.add_argument(
            "--grid-dim-x",
            "-gidx",
            dest="grid_dim_x",
            type=int,
            default=None,
            help="Value of GridDim.x during kernel replay, when omitted the recorded value is used",
        )

        parser.add_argument(
            "--grid-dim-y",
            "-gidy",
            dest="grid_dim_y",
            type=int,
            default=None,
            help="Value of GridDim.y during kernel replay, when omitted the recorded value is used",
        )

        parser.add_argument(
            "--grid-dim-z",
            "-gidz",
            dest="grid_dim_z",
            type=int,
            default=None,
            help="Value of GridDim.z during kernel replay, when omitted the recorded value is used",
        )

        parser.add_argument(
            "--shared-mem",
            "-shem",
            dest="shared_mem",
            type=int,
            default=None,
            help="Size of shared memory, if not set we default to recorded value",
        )

        parser.add_argument(
            "--iterations",
            "-it",
            required=False,
            type=int,
            help="The number of iterations to run every execution, used to get statistical meaningful results",
            default=3,
        )

        parser.set_defaults(func=CLIExecutor.run)

    def __init__(self, *args, **kwargs):
        init_profiler()
        self.pipeline = kwargs.pop("pipeline", None)
        self.increamental = kwargs.pop("increamental", False)
        self.specialize = kwargs.pop("specialize", False)
        self.max_threads = kwargs.pop("max_threads", None)
        self.min_blocks_per_sm = kwargs.pop("min_blocks_per_sm", 0)
        self.dims = kwargs.pop("dims", False)
        self.results_db_dir = kwargs.pop("results_db_dir", None)
        self.codegen_method = kwargs.pop("codegen_method", "serial")
        self.codegen_opt = kwargs.pop("codegen_opt", 3)
        self.block_dim_x = kwargs.pop("block_dim_x", None)
        self.block_dim_y = kwargs.pop("block_dim_y", None)
        self.block_dim_z = kwargs.pop("block_dim_z", None)
        self.grid_dim_x = kwargs.pop("grid_dim_x", None)
        self.grid_dim_y = kwargs.pop("grid_dim_y", None)
        self.grid_dim_z = kwargs.pop("grid_dim_z", None)
        self.set_launch_bounds = kwargs.pop("set_launch_bounds", False)
        self.specialize_dims = kwargs.pop("specialize_dims", False)
        self.shared_mem = kwargs.pop("shared_mem", None)

        print(json.dumps(kwargs, indent=6))
        super().__init__(*args, **kwargs)
        self.pass_manager = PipelineManager()
        if self.pipeline not in (
            "default<O3>",
            "default<O2>",
            "default<O1>",
            "default<O0>",
            "default<Os",
            "default<Oz>",
        ):
            self.passes = self.pass_manager.from_string(self.pipeline)
        else:
            self.passes = self.pipeline
        self._db = None

    def get_experiment(self, pipeline):
        self.block_dim_x = (
            self.kernel_descr.block_dim.x
            if (self.block_dim_x is None)
            else self.block_dim_x
        )
        self.block_dim_y = (
            self.kernel_descr.block_dim.y
            if (self.block_dim_y is None)
            else self.block_dim_y
        )
        self.block_dim_z = (
            self.kernel_descr.block_dim.z
            if (self.block_dim_z is None)
            else self.block_dim_z
        )

        self.grid_dim_x = (
            self.kernel_descr.grid_dim.x
            if (self.grid_dim_x is None)
            else self.grid_dim_x
        )
        self.grid_dim_y = (
            self.kernel_descr.grid_dim.y
            if (self.grid_dim_y is None)
            else self.grid_dim_y
        )
        self.grid_dim_z = (
            self.kernel_descr.grid_dim.z
            if (self.grid_dim_z is None)
            else self.grid_dim_z
        )

        max_threads = self.max_threads
        if self.set_launch_bounds:
            if self.max_threads == -1:
                max_threads = self.block_dim_x * self.block_dim_y * self.block_dim_z

        self.shared_mem = (
            self.kernel_descr.shared_mem if self.shared_mem is None else self.shared_mem
        )

        return Experiment(
            grid_dim_x=self.grid_dim_x,
            grid_dim_y=self.grid_dim_y,
            grid_dim_z=self.grid_dim_z,
            block_dim_x=self.block_dim_x,
            block_dim_y=self.block_dim_y,
            block_dim_z=self.block_dim_z,
            specialize=self.specialize,
            shared_mem=self.shared_mem,
            set_launch_bounds=self.set_launch_bounds,
            max_threads=max_threads,
            min_blocks_per_sm=self.min_blocks_per_sm,
            specialize_dims=self.specialize_dims,
            passes=pipeline,
            prune=True,
            internalize=True,
            codegen_opt=self.codegen_opt,
            codegen_method=self.codegen_method,
            device_arch=self.device_arch,
        )

    def __str__(self):
        return f"{self.__class__.__name__}"

    def execute(self, exp, ir_module, clone=False, orig=""):
        if not self.increamental:
            exp.pipeline = self.pipeline
            exp, generated_ir = super()._execute(exp, ir_module)
            if self._db is not None:
                final = self._db.save_ir(generated_ir, exp.hash())
                self._db.add(orig, final, exp)
            return

        results = []
        passes = [("", "")]
        for i, _ in enumerate(self.passes):
            passes.append(
                (
                    self.pass_manager.to_string(self.passes[: i + 1]),
                    self.pass_manager.to_string(self.passes[i : i + 1]),
                )
            )

        with PrettyTablePrinter() as printer:
            for pipeline, pass_name in passes:
                exp.passes = pipeline
                if self._db is not None and not self._db.should_execute(exp):
                    logger.debug(
                        f"Skipping experiment {str(exp.hash())}, already in replayed"
                    )
                    continue

                exp, generated_ir = super()._execute(exp, ir_module.clone(), pipeline)
                if self._db is not None:
                    final = self._db.save_ir(generated_ir, exp.hash())
                    self._db.add(orig, final, exp)
                printer.print_pass_result(pass_name, exp.exec_time, exp.verified)

    @staticmethod
    def run(args, verbosity):
        kwargs = vars(args)
        kwargs.pop("command")
        kwargs.pop("func")
        executor = CLIExecutor(**kwargs)

        # We currently link all LLVM IR modules together
        # NOTE: Does this break with externals on CUDA?
        root_ir = executor.link_ir()

        orig = ""
        if executor.results_db_dir is not None:
            executor._db = MnemeDB(
                executor.results_db_dir,
                executor.kernel_descr.static_hash,
                executor.kernel_descr.dynamic_hash,
            ).open()
            orig = executor._db.save_ir(root_ir, "orig")

        with executor as Memory:
            exp = executor.get_experiment(executor.pipeline)
            executor.execute(
                exp,
                root_ir.clone(),
                True,
            )

        return


class TuneWorker(BaseExecutor):
    def __init__(self, *args, **kwargs):
        init_profiler()
        super().__init__(*args, **kwargs)
        self.pass_manager = PipelineManager()

    def process_payload(self, ir_module, exp_dict) -> Tuple[Experiment, ModuleRef]:
        exp = Experiment.from_dict(**exp_dict)
        exp.start_time = datetime.utcnow().isoformat()
        exp, generated_ir = super()._execute(exp, ir_module)
        exp.end_time = datetime.utcnow().isoformat()
        exp.gpu_id = self.device_id
        return exp, generated_ir

    @staticmethod
    def run(
        request_q: Queue,
        response_q: Queue,
        record_db: str,
        record_id: str,
        device_id: int,
        iterations: int,
        results_db_dir: str,
        state: Event,
    ):
        # NOTE: We open a file for every individual executor and give persmisions, then we redirect stdout/stderr
        # to that file. We do this to not conflict our messages
        fd_out = os.open(
            f"{results_db_dir}/Worker-{device_id}.log",
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        )
        os.dup2(fd_out, 1)  # 1 = stdout
        os.dup2(fd_out, 2)  # 2 = stderr
        worker = TuneWorker(
            record_db=record_db,
            record_id=record_id,
            device_id=device_id,
            iterations=iterations,
        )
        # Open GPU memory, setup prologue epilogue and create a single
        # LLVM IR file to start working on optimizations
        root_ir = worker.link_ir()
        resdb = MnemeDB(
            results_db_dir,
            worker.kernel_descr.static_hash,
            worker.kernel_descr.dynamic_hash,
        ).open()

        with worker as Memory:
            state.set()
            logger.debug("Worker running on {worker.device_id} starts busy loop")
            while True:
                msg = request_q.get()
                if msg["payload"] == "terminate":
                    logger.debug(
                        f"Worker {worker.device_id} received terminate request, exiting ..."
                    )
                    break
                elif msg["payload"] == "process":
                    logger.debug(
                        f"Worker {worker.device_id} received processing request {msg['exp_id']}"
                    )
                    exp, ir = worker.process_payload(root_ir.clone(), msg["data"])
                    final = resdb.save_ir(ir, exp.hash())
                    logger.debug(
                        f"Worker {worker.device_id} finalized processing request {msg['exp_id']}"
                    )

                    response_q.put(
                        {
                            "exp_id": msg["exp_id"],
                            "payload": "result",
                            "data": exp.to_dict(),
                            "llvm_ir": final,
                        }
                    )
                else:
                    logger.warning(f"Received unknown message {msg}")

        return
