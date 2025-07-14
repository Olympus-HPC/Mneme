import argparse
import time
from typing import Tuple

from mneme.device import DeviceModule, dim3, get_device_arch
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
            "-id", "--id", required=True, help="Kernel ID to operate on"
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

        parser.set_defaults(prune=True, internalize=False, rtc=False)

        return parser

    def __init__(
        self,
        db: str = "",
        id: str = "",
        prune: bool = True,
        internalize: bool = False,
        iterations: int = 3,
        rtc: bool = False,
        codegen_opt: int = 3,
    ):
        self.records = RecordedExecution.from_json(db)
        self.kernel_descr = self.records[id]
        self.device_arch = get_device_arch()
        self.prune = prune
        self.internalize = internalize
        self.codegen_opt = 3
        self.rtc = rtc
        self._epilogue = None
        self._prologue = None
        self._page_manager = None
        self._iteration = 3

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

    def execute(
        self,
        ir_module: ModuleRef,
        middle_end_opt: str,
        clone,
    ) -> Tuple[ModuleRef, float, float, float, float, bool]:
        m_start = time.perf_counter()
        jit.optimize(
            ir_module,
            self.device_arch,
            middle_end_opt,
            self.codegen_opt,
        )
        m_end = time.perf_counter()
        m_compile_time = m_end - m_start
        opt_file = ir_module
        if clone:
            opt_file = ir_module.clone()

        c_start = time.perf_counter()
        mem_buffer = jit.codegen_object(
            opt_file, self.device_arch, self.rtc, self.codegen_opt
        )
        c_end = time.perf_counter()
        b_compile_time = c_end - c_start
        object_size = mem_buffer.get_size()
        if self._prologue._state is None or self._epilogue._state is None:
            raise RuntimeError("States should never be none when executing a kernel")

        with DeviceModule.from_MemBuffer(mem_buffer) as DeviceObj:
            device_func = DeviceObj.get_function(self.kernel_descr.kernel_name)
            perf_time = device_func.profile(
                self.kernel_descr.grid_dim,
                self.kernel_descr.block_dim,
                self._prologue._state,
                self._epilogue._state,
                self.kernel_descr.shared_mem,
                self._iteration,
            )
            avg_time = sum(perf_time) / len(perf_time)
            return (
                opt_file,
                avg_time,
                m_compile_time,
                b_compile_time,
                object_size,
                self.prologue == self.epilogue,
            )


class CLIExecutor(BaseExecutor):
    @staticmethod
    def set_cli_args(parser):
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
        super().__init__(*args, **kwargs)
        self.pass_manager = PipelineManager()
        self.passes = self.pass_manager.from_string(self.pipeline)

    def __str__(self):
        return f"{self.__class__.__name__}"

    def execute(self, ir_module, clone=False):
        if not self.increamental:
            return super().execute(ir_module, self.pipeline, clone)

        results = []
        for i, _ in enumerate(self.passes):
            (
                opt_code,
                avg_time,
                m_c_time,
                b_c_time,
                object_size,
                verified,
            ) = super().execute(
                ir_module, self.pass_manager.to_string(self.passes[: i + 1]), True
            )

            print(
                f"Applied pipeline: {self.pass_manager.to_string(self.passes[: i + 1])} \t Time: {avg_time:.5f},\t Verified: \t{verified}, MCompile Time: \t {m_c_time:.5f}, BCompile Time: \t {b_c_time:.5f}\t Obj. Size: {object_size}"
            )

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
            (
                opt_code,
                avg_time,
                m_c_time,
                b_c_time,
                object_size,
                verified,
            ) = executor.execute(
                code,
                True,
            )

            print(
                f"[{code_hash}] \t Time: {avg_time:.5f},\t Verified: \t{verified}, MCompile Time: \t {m_c_time:.5f}, BCompile Time: \t {b_c_time:.5f}\t Obj. Size: {object_size}"
            )
        return
