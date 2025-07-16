import argparse
import time
from multiprocessing import Queue
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
from mneme.llvm.module import ModuleRef
from mneme.page_manager import PageManagerRef
from mneme.pipeline import PipelineManager
from mneme.proteus import jit
from mneme.recorded_execution import MemStateRef, RecordedExecution


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
            "--rtc",
            dest="rtc",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Use Vendor RTC when lowering to device object file instead of default Proteus Infrastructure",
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

        parser.set_defaults(prune=True, internalize=False, rtc=False)

        return parser

    def __init__(
        self,
        db: str = "",
        record_id: str = "",
        prune: bool = True,
        internalize: bool = False,
        iterations: int = 3,
        rtc: bool = False,
        codegen_opt: int = 3,
        device_id: int = 0,
    ):
        self.db = db
        self.record_id = record_id
        print(f"Got {db} and {record_id} for gpu {device_id}")
        self.records = RecordedExecution.from_json(db)
        self.kernel_descr = self.records[record_id]
        self.device_arch = get_device_arch()
        self.device_id = device_id
        self.prune = prune
        self.internalize = internalize
        self.codegen_opt = codegen_opt
        self.rtc = rtc
        self._epilogue = None
        self._prologue = None
        self._page_manager = None
        self._iterations = iterations
        self.num_devices = get_device_count()
        set_device(device_id)
        print(f"Setting Device ID to {device_id} out of {self.num_devices}")

    def open(self):
        self._page_manager = PageManagerRef(self.records.va_addr, self.records.va_size)
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

    def _execute(
        self, exp: Experiment, ir_module: ModuleRef, middle_end_opt: str
    ) -> Tuple[Experiment, ModuleRef]:
        m_start = time.perf_counter()
        jit.optimize(
            ir_module,
            self.device_arch,
            middle_end_opt,
            self.codegen_opt,
        )

        m_end = time.perf_counter()
        exp.opt_time = m_end - m_start
        opt_file = ir_module

        c_start = time.perf_counter()
        mem_buffer = jit.codegen_object(
            opt_file, self.device_arch, self.rtc, self.codegen_opt
        )
        c_end = time.perf_counter()
        exp.codegen_time = c_end - c_start
        exp.obj_size = mem_buffer.get_size()

        if self._prologue._state is None or self._epilogue._state is None:
            raise RuntimeError("States should never be none when executing a kernel")

        with DeviceModule.from_MemBuffer(mem_buffer) as DeviceObj:
            device_func = DeviceObj.get_function(self.kernel_descr.kernel_name)
            exp.exec_time = device_func.profile(
                self.kernel_descr.grid_dim,
                self.kernel_descr.block_dim,
                self._prologue._state,
                self._epilogue._state,
                self.kernel_descr.shared_mem,
                self._iterations,
            )
            exp.verified = self.prologue == self.epilogue
            exp.executed = True

            return exp, opt_file


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
            "--suffix",
            required=False,
            default=None,
            help="Suffix of the database file (e.g. <args.db_dir><static_hash><dynamic_hash><suffix>.csv)",
        )

        parser.add_argument(
            "--apply-increamentally",
            required=False,
            action=argparse.BooleanOptionalAction,
            dest="increamental",
            default=False,
            help="Apply passes one by one and incrementally build the pipelines. Useful for debugging",
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

    def apply_user_options(self, llvm_ir):
        code_hash = self.kernel_descr.static_hash
        code = llvm_ir.clone()
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
                code,
                code_hash,
                self.kernel_descr.kernel_name,
                self.kernel_descr.grid_dim,
                self.kernel_descr.block_dim,
            )
        if self.max_threads:
            code_hash = jit.set_launch_bounds(
                code,
                self.code_hash,
                self.kernel_descr.kernel_name,
                self.max_threads,
                self.min_blocks_per_sm,
            )
        return code_hash, code

    def __init__(self, *args, **kwargs):
        self.pipeline = kwargs.pop("pipeline", None)
        self.increamental = kwargs.pop("increamental", False)
        self.specialize = kwargs.pop("specialize", False)
        self.max_threads = kwargs.pop("max_threads", False)
        self.min_blocks_per_sm = kwargs.pop("min_blocks_per_sm", 0)
        self.dims = kwargs.pop("dims", False)
        self.db_dir = kwargs.pop("db_dir", None)
        self.db_suffix = kwargs.pop("suffix", None)
        super().__init__(*args, **kwargs)
        self.pass_manager = PipelineManager()
        self.passes = self.pass_manager.from_string(self.pipeline)
        self._db = None

    def get_experiment(self, pipeline):
        return Experiment(
            specialize=self.specialize,
            max_threads=self.max_threads,
            min_blocks_per_sm=self.min_blocks_per_sm,
            specialize_dims=self.dims,
            passes=pipeline,
            prune=self.prune,
            internalize=self.internalize,
            codegen_opt=self.codegen_opt,
            rtc=self.rtc,
            device_arch=self.device_arch,
        )

    def __str__(self):
        return f"{self.__class__.__name__}"

    def execute(self, ir_module, clone=False):
        orig = ""
        if self.db_dir is not None:
            self._db = MnemeDB(
                self.db_dir,
                self.kernel_descr.static_hash,
                self.kernel_descr.dynamic_hash,
                self.db_suffix,
            ).open()
            orig = self._db.save_ir(str(ir_module), "orig")
        if not self.increamental:
            exp = self.get_experiment(self.pipeline)
            print("Starting to execute")
            exp.dump()
            exp, generated_ir = super()._execute(exp, ir_module.clone(), self.pipeline)
            if self._db is not None:
                final = self._db.save_ir(str(generated_ir), exp.hash())
                self._db.add(orig, final, exp)
            exp.dump()
            return

        results = []
        for i, _ in enumerate(self.passes):
            passes = self.pass_manager.to_string(self.passes[: i + 1])
            exp = self.get_experiment(passes)
            if self._db is not None and not self._db.should_execute(exp):
                print("Skipping")
                continue

            exp, generated_ir = super()._execute(exp, ir_module.clone(), passes)
            if self._db is not None:
                print(f"Hash of code is {exp.hash()}")
                final = self._db.save_ir(str(generated_ir), exp.hash())
                self._db.add(orig, final, exp)
            exp.dump()

    @staticmethod
    def run(args):
        kwargs = vars(args)
        kwargs.pop("command")
        kwargs.pop("func")
        executor = CLIExecutor(**kwargs)

        # We currently link all LLVM IR modules together
        # NOTE: Does this break with externals on CUDA?
        root_ir = executor.link_ir()
        with executor as Memory:
            code_hash, code = executor.apply_user_options(root_ir)
            executor.execute(
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
        super().__init__(*args, **kwargs)
        self.pass_manager = PipelineManager()

    def process_payload(self, ir_module, exp_dict) -> Tuple[Experiment, ModuleRef]:
        exp = Experiment.from_dict(**exp_dict)
        code_hash, code = self.preprocess_ir(
            ir_module.clone(),
            exp.specialize,
            exp.specialize_dims,
            exp.max_threads,
            exp.min_blocks_per_sm,
        )
        exp, generated_ir = super()._execute(exp, code, exp.passes)
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
        rtc: bool,
        iterations: int,
    ):
        # We need this to actually run things...
        worker = TuneWorker(
            db=db,
            record_id=record_id,
            device_id=device_id,
            prune=prune,
            internalize=internalize,
            codegen_opt=codegen_opt,
            rtc=rtc,
            iterations=iterations,
        )
        # Open GPU memory, setup prologue epilogue and create a single
        # LLVM IR file to start working on optimizations
        root_ir = worker.link_ir()
        with worker as Memory:
            print("Starting busy loop")
            while True:
                msg = request_q.get()
                if msg["payload"] == "terminate":
                    print("Instructed to terminate normally, doing so...")
                    break
                elif msg["payload"] == "process":
                    print(
                        f"Worker {worker.device_id} Received message to start processing request",
                        msg["exp_id"],
                    )
                    exp, ir = worker.process_payload(root_ir, msg["data"])
                    response_q.put(
                        {
                            "exp_id": msg["exp_id"],
                            "payload": "result",
                            "data": exp.to_dict(),
                            "llvm_ir": str(ir),
                        }
                    )
                else:
                    print(f"Received uknown message {msg}")

        return
