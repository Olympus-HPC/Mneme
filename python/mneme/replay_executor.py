import argparse
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
            "-db",
            "--record-database",
            dest="db",
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

        parser.add_argument(
            "--prune",
            action=argparse.BooleanOptionalAction,
            default=True,
            dest="prune",
            help="Aggressive Dead Code Elimination by assuming we will only execute the kernel identified by 'id'",
        )

        # default is True; user can override with --no-foo
        parser.add_argument(
            "--internalize",
            dest="internalize",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Internalize LLVM IR before optimization (reduces size of generated object file and may enable more aggressive inlining)",
        )

        parser.add_argument(
            "-cm",
            "--codegen-method",
            dest="codegen_method",
            choices=["rtc", "serial", "parallel"],
            default="serial",
            action=argparse.BooleanOptionalAction,
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
            "--iterations",
            "-it",
            required=False,
            type=int,
            help="The number of iterations to run every execution, used to get statistical meaningful results",
            default=3,
        )

        parser.add_argument(
            "--device-id",
            "-dev",
            dest="device_id",
            type=int,
            required=False,
            help="The GPU device ID to use",
            default=0,
        )

        parser.set_defaults(prune=True, internalize=False, codegen_method="serial")

        return parser

    def __init__(
        self,
        db: str = "",
        record_id: str = "",
        prune: bool = True,
        internalize: bool = False,
        iterations: int = 3,
        codegen_method: str = "serial",
        codegen_opt: int = 3,
        device_id: int = 0,
    ):
        self.db = db
        self.record_id = record_id
        logger.debug(
            f"BaseExecutor Got {db} and {record_id} and will run on device:{device_id}"
        )
        self.records = RecordedExecution.from_json(db)
        self.kernel_descr = self.records[record_id]
        self.device_arch = get_device_arch()
        self.device_id = device_id
        self.prune = prune
        self.internalize = internalize
        self.codegen_opt = codegen_opt
        self.codegen_method = codegen_method
        self._epilogue = None
        self._prologue = None
        self._page_manager = None
        self._iterations = iterations
        self.num_devices = get_device_count()
        set_device(device_id)
        logger.debug(
            f"GPU Affinity of process was set to device:{device_id} out of {self.num_devices}"
        )

    def open(self):
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
        return self.records.link_llvm_modules(
            prune=self.prune, internalize=self.internalize
        )

    @cond_time("opt_time")
    def _optimize(self, exp, ir_module, middle_end_opt):
        jit.optimize(ir_module, self.device_arch, middle_end_opt, self.codegen_opt)

    @cond_time("codegen_time")
    def _codegen(self, exp, ir_module):
        return jit.codegen_object(
            ir_module, self.device_arch, self.codegen_method, self.codegen_opt
        )

    @cond_gpu_time("exec_time")
    def _run_kernel(self, exp, kernel_name, device_func, iterations):
        return device_func.profile(
            self.kernel_descr.grid_dim,
            self.kernel_descr.block_dim,
            self._prologue._state,
            self._epilogue._state,
            self.kernel_descr.shared_mem,
            iterations,
        )

    def _build(
        self, exp: Experiment, ir_module: ModuleRef, middle_end_opt: str, track: bool
    ) -> MemBufferRef:
        self._optimize(exp, ir_module, middle_end_opt, profile=track)
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
        self, exp: Experiment, ir_module: ModuleRef, middle_end_opt: str
    ) -> Tuple[Experiment, ModuleRef]:
        if self._prologue._state is None or self._epilogue._state is None:
            raise RuntimeError("States should never be none when executing a kernel")

        # NOTE: 1. First we need to verify.
        ver_mod = ir_module.clone()
        mem_buffer = self._build(exp, ver_mod, middle_end_opt, False)
        self._run(exp, mem_buffer, False, 1)
        exp.verified = self.prologue == self.epilogue

        # NOTE: 2. We apply a custom pass to delete all clang insered code.
        # It is hard to identify these cases, So we delete only things that have been attributed by clang
        ir_module = transform.remove_auto_initialize(ir_module)
        # Done with verification. Moving to next stage

        mem_buffer = self._build(exp, ir_module, middle_end_opt, True)
        self._run(exp, mem_buffer, True, self._iterations + 2)
        exp.executed = True

        return exp, ir_module


class CLIExecutor(BaseExecutor):
    @staticmethod
    def set_cli_args(parser):
        parser.add_argument(
            "--db-dir",
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
            "--dims",
            default=False,
            required=False,
            action=argparse.BooleanOptionalAction,
            dest="dims",
            help="Specialize ThreadID.*, BlockDim.* and GridDim.* with constants and provide 'assume' instructions to llvm",
        )

        parser.add_argument(
            "--max-threads",
            default=False,
            required=False,
            action=argparse.BooleanOptionalAction,
            dest="max_threads",
            help="Set launch bound 'max_threads' of kernel to the executed number of threads",
        )

        parser.add_argument(
            "--min-threads-per-block",
            default=0,
            required=False,
            type=int,
            dest="min_blocks_per_sm",
            help="Set launch bound 'min_blocks_per_sm' of kernel to the provided value",
        )

        parser.set_defaults(func=CLIExecutor.run)

    def apply_user_options(self, llvm_ir, exp):
        code_hash = self.kernel_descr.static_hash
        _hash = exp.hash()
        if self.specialize:
            code_hash = jit.specialize_args(
                llvm_ir,
                code_hash,
                self.kernel_descr.kernel_name,
                self.prologue.args,
                self.prologue.num_args,
                self.kernel_descr.available_specializations,
            )

        if self.dims:
            code_hash = jit.specialize_dims(
                llvm_ir,
                code_hash,
                self.kernel_descr.kernel_name,
                self.kernel_descr.grid_dim,
                self.kernel_descr.block_dim,
            )

        if self.max_threads:
            self.max_threads = int(
                self.kernel_descr.block_dim.x
                * self.kernel_descr.block_dim.y
                * self.kernel_descr.block_dim.z
            )

            code_hash = jit.set_launch_bounds(
                llvm_ir,
                code_hash,
                self.kernel_descr.kernel_name,
                self.max_threads,
                self.min_blocks_per_sm,
            )

        return code_hash, llvm_ir

    def __init__(self, *args, **kwargs):
        init_profiler()
        self.pipeline = kwargs.pop("pipeline", None)
        self.increamental = kwargs.pop("increamental", False)
        self.specialize = kwargs.pop("specialize", False)
        self.max_threads = kwargs.pop("max_threads", False)
        self.min_blocks_per_sm = kwargs.pop("min_blocks_per_sm", 0)
        self.dims = kwargs.pop("dims", False)
        self.db_dir = kwargs.pop("db_dir", None)
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
        max_threads = self.max_threads
        if self.max_threads:
            max_threads = (
                self.kernel_descr.block_dim.x
                * self.kernel_descr.block_dim.y
                * self.kernel_descr.block_dim.z
            )
        return Experiment(
            specialize=self.specialize,
            max_threads=max_threads,
            min_blocks_per_sm=self.min_blocks_per_sm,
            specialize_dims=self.dims,
            passes=pipeline,
            prune=self.prune,
            internalize=self.internalize,
            codegen_opt=self.codegen_opt,
            codegen_method=self.codegen_method,
            device_arch=self.device_arch,
        )

    def __str__(self):
        return f"{self.__class__.__name__}"

    def execute(self, exp, ir_module, clone=False, orig=""):
        if not self.increamental:
            exp, generated_ir = super()._execute(exp, ir_module, self.pipeline)
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
    def run(args):
        kwargs = vars(args)
        kwargs.pop("command")
        kwargs.pop("func")
        executor = CLIExecutor(**kwargs)

        # We currently link all LLVM IR modules together
        # NOTE: Does this break with externals on CUDA?
        root_ir = executor.link_ir()

        orig = ""
        if executor.db_dir is not None:
            executor._db = MnemeDB(
                executor.db_dir,
                executor.kernel_descr.static_hash,
                executor.kernel_descr.dynamic_hash,
            ).open()
            orig = executor._db.save_ir(root_ir, "orig")

        with executor as Memory:
            exp = executor.get_experiment(executor.pipeline)
            code_hash, code = executor.apply_user_options(root_ir.clone(), exp)
            executor.execute(
                exp,
                code,
                True,
            )

        return


class TuneWorker(BaseExecutor):
    def preprocess_ir(self, llvm_ir, specialize, dims, max_threads, min_blocks_per_sm):
        code_hash = self.kernel_descr.static_hash
        if specialize:
            code_hash = jit.specialize_args(
                llvm_ir,
                code_hash,
                self.kernel_descr.kernel_name,
                self.prologue.args,
                self.prologue.num_args,
                self.kernel_descr.available_specializations,
            )

        if dims:
            code_hash = jit.specialize_dims(
                llvm_ir,
                code_hash,
                self.kernel_descr.kernel_name,
                self.kernel_descr.grid_dim,
                self.kernel_descr.block_dim,
            )
        if max_threads != 0:
            code_hash = jit.set_launch_bounds(
                llvm_ir,
                code_hash,
                self.kernel_descr.kernel_name,
                max_threads,
                min_blocks_per_sm,
            )
        return code_hash, llvm_ir

    def __init__(self, *args, **kwargs):
        init_profiler()
        super().__init__(*args, **kwargs)
        self.pass_manager = PipelineManager()

    def process_payload(self, ir_module, exp_dict) -> Tuple[Experiment, ModuleRef]:
        exp = Experiment.from_dict(**exp_dict)
        exp.start_time = datetime.utcnow().isoformat()
        code_hash, code = self.preprocess_ir(
            ir_module,
            exp.specialize,
            exp.specialize_dims,
            exp.max_threads,
            exp.min_blocks_per_sm,
        )
        exp, generated_ir = super()._execute(exp, code, exp.passes)
        exp.end_time = datetime.utcnow().isoformat()
        exp.gpu_id = self.device_id
        return exp, generated_ir

    @staticmethod
    def run(
        request_q: Queue,
        response_q: Queue,
        db: str,
        record_id: str,
        device_id: int,
        prune: bool,
        internalize: bool,
        codegen_opt: int,
        codegen_method: str,
        iterations: int,
        db_dir: str,
        state: Event,
    ):
        # We need this to actually run things...
        fd_out = os.open(
            f"{db_dir}/Worker-{device_id}.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND
        )
        os.dup2(fd_out, 1)  # 1 = stdout
        os.dup2(fd_out, 2)  # 2 = stderr
        worker = TuneWorker(
            db=db,
            record_id=record_id,
            device_id=device_id,
            prune=prune,
            internalize=internalize,
            codegen_opt=codegen_opt,
            codegen_method=codegen_method,
            iterations=iterations,
        )
        # Open GPU memory, setup prologue epilogue and create a single
        # LLVM IR file to start working on optimizations
        root_ir = worker.link_ir()
        resdb = MnemeDB(
            db_dir, worker.kernel_descr.static_hash, worker.kernel_descr.dynamic_hash
        ).open()

        with worker as Memory:
            state.set()
            logger.debug("Worker running on {worker.device_id} starts busy loop")
            while True:
                msg = request_q.get()
                if msg["payload"] == "terminate":
                    logger.debug(
                        "Worker {worker.device_id} received terminate request, exiting ..."
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
