import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from typing import Tuple

from mneme.device import DeviceModule, dim3, get_device_arch
from mneme.llvm.module import ModuleRef
from mneme.page_manager import PageManagerRef
from mneme.pipeline import PipelineManager
from mneme.proteus import jit
from mneme.recorded_execution import MemStateRef, RecordedExecution
from mneme.replay_executor import BaseExecutor, CLIExecutor


def stable_hash(obj):
    # Convert to a stable string representation
    serialized = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def append_to_csv(filename, fieldnames, data):
    # Check if the file exists
    file_exists = os.path.isfile(filename)

    csv_row = {k: v for k, v in zip(fieldnames, data)}

    # Open the file in append mode
    with open(filename, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # If file didn't exist, write the header
        if not file_exists:
            writer.writeheader()

        # Write the row
        writer.writerow(csv_row)


def read_column(filename, column_name):
    values = []
    file_exists = os.path.isfile(filename)

    if not file_exists:
        return set()

    with open(filename, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            values.append(row[column_name])
    return set(values)


def execute(
    ir_module: ModuleRef,
    kernel_descr: RecordedExecution.KernelInstance,
    prologue: MemStateRef,
    epilogue: MemStateRef,
    device_arch: str,
    middle_end_opt: str,
    back_end_opt: int,
    rtc: bool,
    num_iterations: int,
    clone: bool = False,
) -> Tuple[ModuleRef, float, float, float, float, bool]:
    m_start = time.perf_counter()
    jit.optimize(
        ir_module,
        device_arch,
        middle_end_opt,
        back_end_opt,
    )
    m_end = time.perf_counter()
    m_compile_time = m_end - m_start
    opt_file = ir_module
    if clone:
        opt_file = ir_module.clone()

    c_start = time.perf_counter()
    mem_buffer = jit.codegen_object(opt_file, device_arch, rtc, back_end_opt)
    c_end = time.perf_counter()
    b_compile_time = c_end - c_start
    object_size = mem_buffer.get_size()
    if prologue._state is None or epilogue._state is None:
        raise RuntimeError("States should never be none when executing a kernel")

    with DeviceModule.from_MemBuffer(mem_buffer) as DeviceObj:
        device_func = DeviceObj.get_function(kernel_descr.kernel_name)
        perf_time = device_func.profile(
            kernel_descr.grid_dim,
            kernel_descr.block_dim,
            prologue._state,
            epilogue._state,
            kernel_descr.shared_mem,
            num_iterations,
        )
        avg_time = sum(perf_time) / len(perf_time)
        return (
            opt_file,
            avg_time,
            m_compile_time,
            b_compile_time,
            object_size,
            prologue == epilogue,
        )


#        exp_id += 1


def main():
    parser = argparse.ArgumentParser(prog="mneme", description="Mneme Tool")
    parent_parser = BaseExecutor.get_base_parser()

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        required=True,
        description="The mneme available commands",
    )

    p_exec = subparsers.add_parser(
        "execute", parents=[parent_parser], help="run a recorded kernel"
    )

    p_tune = subparsers.add_parser(
        "tune", parents=[parent_parser], help="tune the pipeline of a recorded kernel"
    )

    CLIExecutor.set_cli_args(p_exec)

    args = parser.parse_args()
    args.func(args)
    print("Done")


#    parser.add_argument(
#        "--num-pipelines",
#        "-p",
#        required=False,
#        type=int,
#        default=50,
#        help="How many random pipelines we should generate",
#    )
#
#    parser.add_argument(
#        "--apply-passes-incrementally",
#        "-ap",
#        required=False,
#        dest="individual_passes",
#        action="store_true",
#        help="Apply passes one by one and incrementally build the pipelines. Useful for debugging",
#    )
#
#    parser.add_argument(
#        "--no-apply-passes-incrementally",
#        "-nap",
#        required=False,
#        dest="individual_passes",
#        action="store_false",
#        help="Turn off (default) Apply passes one by one and incrementally build the pipelines. Useful for debugging",
#    )
#
#    parser.set_defaults(
#        prune=True, internalize=False, rtc=False, tune=True, individual_passes=False
#    )
#
#    parser.add_argument(
#        "--seed",
#        required=False,
#        type=int,
#        default=0,
#        dest="seed",
#        help="Seed number to initialize the random pass generation module",
#    )
#
#    args = parser.parse_args()
#    records = RecordedExecution.from_json(args.input)
#
#    # Allocate PM manager addresses
#    with PageManagerRef(records.va_addr, records.va_size) as PM:
#        llvm_ir = records.link_llvm_modules(
#            prune=args.prune, internalize=args.internalize
#        )
#        # The obtained ModuleRef is usable across all kernels in the
#        # record database. So we can just use it. The 'llvm_ir' is already
#        # internalized, and potentially pruned
#        kernel_descr = records[args.record_id]
#        code_hash = kernel_descr.static_hash
#        dynamic_hash = args.record_id
#        arch = get_device_arch()
#        db_name = (
#            f"{kernel_descr.static_hash}.{kernel_descr.dynamic_hash}.{args.seed}.csv"
#        )
#        executed_experiments = read_column(db_name, "exp_hash")
#        with kernel_descr.prologue as prologue:
#            with kernel_descr.epilogue as epilogue:
#                exp_id = 0
#                max_threads = int(
#                    kernel_descr.block_dim.x
#                    * kernel_descr.block_dim.y
#                    * kernel_descr.block_dim.z,
#                )
#
#                default_pipelines = [
#                    "default<O1>",
#                    "default<O2>",
#                    "default<O3>",
#                    "default<Os>",
#                    "default<Oz>",
#                ]
#                default_exec_times = {k: -1.0 for k in default_pipelines}
#
#                LLVMPassManager = PipelineManager()
#
#                passes = LLVMPassManager.generate(
#                    args.num_pipelines, 120, 33, True, args.seed
#                )
#
#                exploration_passes = {"default": default_pipelines, "random": passes}
#
#                back_opts = [1, 2, 3]
#                specializations = [0, 1]
#                dimensions = [0, 1]
#                min_blocks_per_sm = [
#                    i for i in range(0, int(math.ceil(1024 / max_threads)) + 1)
#                ]
#
#                if not args.tune:
#                    middle_opts = ["default<O3>"]
#                    back_opts = [3]
#                    specializations = [0]
#                    dimensions = [0]
#                    min_blocks_per_sm = [0]
#
#                for pass_type in [
#                    "default",
#                    "random",
#                ]:  # I need to access keys like this, to enforce order and report some "meaningful" speedup
#                    middle_opts = exploration_passes[pass_type]
#                    for middle_opt in middle_opts:
#                        for back_opt in back_opts:
#                            for spec in specializations:
#                                for dims in dimensions:
#                                    for lb in min_blocks_per_sm:
#                                        # We clone, as we are going to modify the code in the next steps
#                                        code = llvm_ir.clone()
#                                        start = time.time()
#                                        if spec:
#                                            code_hash = jit.specialize_args(
#                                                code,
#                                                code_hash,
#                                                kernel_descr.kernel_name,
#                                                prologue.args,
#                                                prologue.num_args,
#                                                kernel_descr.available_specializations,
#                                            )
#                                        if dims:
#                                            code_hash = jit.specialize_dims(
#                                                code,
#                                                code_hash,
#                                                kernel_descr.kernel_name,
#                                                kernel_descr.grid_dim,
#                                                kernel_descr.block_dim,
#                                            )
#                                        if lb:
#                                            code_hash = jit.set_launch_bounds(
#                                                code,
#                                                code_hash,
#                                                kernel_descr.kernel_name,
#                                                max_threads,
#                                                lb,
#                                            )
#                                        if (
#                                            args.individual_passes
#                                            and pass_type != "default"
#                                        ):
#                                            pipeline = ""
#                                            for j, opt in enumerate(middle_opt):
#                                                mopt = PipelineManager.to_string([opt])
#                                                pipeline += "," + mopt
#
#                                                # TODO:  We need a better way to hash things. spec is too simple, and we are unaware on how to redo this in pure proteus executions.
#                                                exp_hash = stable_hash(
#                                                    (
#                                                        code_hash,
#                                                        dynamic_hash,
#                                                        max_threads,
#                                                        spec,
#                                                        dims,
#                                                        lb,
#                                                        kernel_descr.grid_dim.x,
#                                                        kernel_descr.grid_dim.y,
#                                                        kernel_descr.grid_dim.z,
#                                                        kernel_descr.block_dim.x,
#                                                        kernel_descr.block_dim.y,
#                                                        kernel_descr.block_dim.z,
#                                                        mopt,
#                                                        back_opt,
#                                                        arch,
#                                                        args.rtc,
#                                                        args.internalize,
#                                                        args.prune,
#                                                    )
#                                                )
#
#                                                if exp_hash in executed_experiments:
#                                                    print(
#                                                        f"Skipping experiment {exp_hash}"
#                                                    )
#                                                    continue
#
#                                                (
#                                                    opt_code,
#                                                    avg_time,
#                                                    m_c_time,
#                                                    b_c_time,
#                                                    object_size,
#                                                    verified,
#                                                ) = execute(
#                                                    code,
#                                                    kernel_descr,
#                                                    prologue,
#                                                    epilogue,
#                                                    arch,
#                                                    mopt,
#                                                    back_opt,
#                                                    args.rtc,
#                                                    args.iterations,
#                                                    True,
#                                                )
#                                                print(
#                                                    f"[{exp_id}-{code_hash}] \t ME {hash(mopt)}: BE {back_opt}, LB ({max_threads}, {lb}) \t Spec: {spec} DimSet: {dims} Time: {avg_time:.5f},\t Verified: \t{verified}, MCompile Time: \t {m_c_time:.5f}, BCompile Time: \t {b_c_time:.5f}, \t Obj. Size: {object_size}"
#                                                )
#                                                exp_id += 1
#                                                append_to_csv(
#                                                    db_name,
#                                                    [
#                                                        "exp_hash",
#                                                        "code_hash",
#                                                        "dynamic_hash",
#                                                        "device_architecture",
#                                                        "block_dim.x",
#                                                        "block_dim.y",
#                                                        "block_dim.z",
#                                                        "grid_dim.x",
#                                                        "grid_dim.y",
#                                                        "grid_dim.z",
#                                                        "rtc",
#                                                        "internalize",
#                                                        "prune",
#                                                        "mpipeline",
#                                                        "bpipeline",
#                                                        "max_threads",
#                                                        "min_threads_per_block",
#                                                        "specialize",
#                                                        "dim",
#                                                        "binary_size (bytes)",
#                                                        "middle_end_compile_time (s)",
#                                                        "back_end_compile_time (s)",
#                                                        "execution-time (ms)",
#                                                        "verified",
#                                                    ],
#                                                    [
#                                                        exp_hash,
#                                                        code_hash,
#                                                        dynamic_hash,
#                                                        arch,
#                                                        kernel_descr.block_dim.x,
#                                                        kernel_descr.block_dim.y,
#                                                        kernel_descr.block_dim.z,
#                                                        kernel_descr.grid_dim.x,
#                                                        kernel_descr.grid_dim.y,
#                                                        kernel_descr.grid_dim.z,
#                                                        args.rtc,
#                                                        args.internalize,
#                                                        args.prune,
#                                                        mopt,
#                                                        back_opt,
#                                                        max_threads,
#                                                        lb,
#                                                        spec,
#                                                        dims,
#                                                        object_size,
#                                                        m_c_time,
#                                                        b_c_time,
#                                                        avg_time,
#                                                        verified,
#                                                    ],
#                                                )
#                                                executed_experiments.add(exp_hash)
#                                        else:
#                                            mopt = middle_opt
#                                            if pass_type != "default":
#                                                mopt = PipelineManager.to_string(
#                                                    middle_opt
#                                                )
#                                            # TODO:  We need a better way to hash things. spec is too simple, and we are unaware on how to redo this in pure proteus executions.
#                                            exp_hash = stable_hash(
#                                                (
#                                                    code_hash,
#                                                    dynamic_hash,
#                                                    max_threads,
#                                                    spec,
#                                                    dims,
#                                                    lb,
#                                                    kernel_descr.grid_dim.x,
#                                                    kernel_descr.grid_dim.y,
#                                                    kernel_descr.grid_dim.z,
#                                                    kernel_descr.block_dim.x,
#                                                    kernel_descr.block_dim.y,
#                                                    kernel_descr.block_dim.z,
#                                                    mopt,
#                                                    back_opt,
#                                                    arch,
#                                                    args.rtc,
#                                                    args.internalize,
#                                                    args.prune,
#                                                )
#                                            )
#
#                                            if exp_hash in executed_experiments:
#                                                continue
#
#                                            (
#                                                opt_code,
#                                                avg_time,
#                                                m_c_time,
#                                                b_c_time,
#                                                object_size,
#                                                verified,
#                                            ) = execute(
#                                                code,
#                                                kernel_descr,
#                                                prologue,
#                                                epilogue,
#                                                arch,
#                                                mopt,
#                                                back_opt,
#                                                args.rtc,
#                                                args.iterations,
#                                                True,
#                                            )
#
#                                            print(
#                                                f"[{exp_id}-{code_hash}] \t ME {hash(mopt)}: BE {back_opt}, LB ({max_threads}, {lb}) \t Spec: {spec} DimSet: {dims} Time: {avg_time:.5f},\t Verified: \t{verified}, MCompile Time: \t {m_c_time:.5f}, BCompile Time: \t {b_c_time:.5f}\t Obj. Size: {object_size}"
#                                            )
#                                            append_to_csv(
#                                                db_name,
#                                                [
#                                                    "exp_hash",
#                                                    "code_hash",
#                                                    "dynamic_hash",
#                                                    "device_architecture",
#                                                    "block_dim.x",
#                                                    "block_dim.y",
#                                                    "block_dim.z",
#                                                    "grid_dim.x",
#                                                    "grid_dim.y",
#                                                    "grid_dim.z",
#                                                    "rtc",
#                                                    "internalize",
#                                                    "prune",
#                                                    "mpipeline",
#                                                    "bpipeline",
#                                                    "max_threads",
#                                                    "min_threads_per_block",
#                                                    "specialize",
#                                                    "dim",
#                                                    "binary_size (bytes)",
#                                                    "middle_end_compile_time (s)",
#                                                    "back_end_compile_time (s)",
#                                                    "execution-time (ms)",
#                                                    "verified",
#                                                ],
#                                                [
#                                                    exp_hash,
#                                                    code_hash,
#                                                    dynamic_hash,
#                                                    arch,
#                                                    kernel_descr.block_dim.x,
#                                                    kernel_descr.block_dim.y,
#                                                    kernel_descr.block_dim.z,
#                                                    kernel_descr.grid_dim.x,
#                                                    kernel_descr.grid_dim.y,
#                                                    kernel_descr.grid_dim.z,
#                                                    args.rtc,
#                                                    args.internalize,
#                                                    args.prune,
#                                                    mopt,
#                                                    back_opt,
#                                                    max_threads,
#                                                    lb,
#                                                    spec,
#                                                    dims,
#                                                    object_size,
#                                                    m_c_time,
#                                                    b_c_time,
#                                                    avg_time,
#                                                    verified,
#                                                ],
#                                            )
#                                            executed_experiments.add(exp_hash)
#                                            exp_id += 1
#

if __name__ == "__main__":
    sys.exit(main())
