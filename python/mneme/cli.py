import argparse
import time
import sys
from mneme.recorded_execution import RecordedExecution
from mneme.device import get_device_arch

from mneme.proteus import jit
from mneme.page_manager import PageManagerRef
from mneme.device import DeviceModule, dim3


def main():
    # print(get_device_arch())
    parser = argparse.ArgumentParser(description="Mneme auto-tuning tool")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="JSON file containing the record DB of a kernel",
    )
    # default is True; user can override with --no-foo
    parser.add_argument(
        "--prune",
        dest="prune",
        action="store_true",
        help="prune LLVM IR before optimization",
    )
    parser.add_argument(
        "--no-prune",
        dest="prune",
        action="store_false",
        help="Do NOT prune LLVM IR before optimization",
    )

    # default is True; user can override with --no-foo
    parser.add_argument(
        "--internalize",
        dest="internalize",
        action="store_true",
        help="internalize LLVM IR before optimization",
    )
    parser.add_argument(
        "--no-internalize",
        dest="internalize",
        action="store_false",
        help="Do NOT prune LLVM IR before optimization",
    )

    parser.add_argument(
        "--rtc",
        dest="rtc",
        action="store_true",
        help="internalize LLVM IR before optimization",
    )
    parser.add_argument(
        "--no-rtc",
        dest="rtc",
        action="store_false",
        help="Do NOT prune LLVM IR before optimization",
    )

    parser.add_argument(
        "--record-id",
        "-id",
        required=True,
        help="The record -id (instance-key) to replay",
    )

    parser.add_argument(
        "--iterations",
        "-it",
        required=False,
        type=int,
        help="Help the number of iterations to run every configuration",
        default=5,
    )

    parser.set_defaults(prune=True, internalize=False, rtc=False)
    args = parser.parse_args()
    records = RecordedExecution.from_json(args.input)

    # Allocate PM manager addresses
    with PageManagerRef(records.va_addr, records.va_size) as PM:
        llvm_ir = records.link_llvm_modules(
            prune=args.prune, internalize=args.internalize
        )
        # The obtained ModuleRef is usable across all kernels in the
        # record database. So we can just use it. The 'llvm_ir' is already
        # internalized, and potentially pruned
        kernel_descr = records[args.record_id]
        arch = get_device_arch()
        with kernel_descr.prologue as prologue:
            with kernel_descr.epilogue as epilogue:
                exp_id = 0
                for middle_opt in ["0", "1", "2", "3", "s", "z"]:
                    for back_opt in [0, 1, 2, 3]:
                        # We clone, as we are going to modify the code in the next steps
                        code = llvm_ir.clone()
                        start = time.time()
                        jit.optimize(code, arch, middle_opt, back_opt)
                        end = time.time()
                        mem_buffer = jit.codegen_object(code, arch, args.rtc)
                        with DeviceModule.from_MemBuffer(mem_buffer) as DeviceObj:
                            device_func = DeviceObj.get_function(
                                kernel_descr.kernel_name
                            )
                            perf_time = device_func.profile(
                                kernel_descr.grid_dim,
                                kernel_descr.block_dim,
                                prologue._state,
                                epilogue._state,
                                kernel_descr.shared_mem,
                                args.iterations,
                            )
                            avg_time = sum(perf_time) / len(perf_time)
                            compile_time = end - start
                            print(
                                f"[{exp_id}] \t Enabled Optimizations: midle-end {middle_opt}: back end {back_opt} \t Time: {avg_time:.5f},\t Result Verified: \t{prologue == epilogue}, Compile Time: \t {compile_time:.5f}"
                            )
                            exp_id += 1


if __name__ == "__main__":
    sys.exit(main())
