import argparse
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
    parser.add_argument(
        "--record-id",
        "-id",
        required=True,
        help="The record -id (instance-key) to replay",
    )

    parser.set_defaults(prune=True)
    args = parser.parse_args()
    records = RecordedExecution.from_json(args.input)

    # Allocate PM manager addresses
    with PageManagerRef(records.va_addr, records.va_size) as PM:
        llvm_ir = records.link_llvm_modules(prune=args.prune)
        # The obtained ModuleRef is usable across all kernels in the
        # record database. So we can just use it. The 'llvm_ir' is already
        # internalized, and potentially pruned

        kernel_descr = records[args.record_id]
        arch = get_device_arch()
        with kernel_descr.prologue as prologue:
            with kernel_descr.epilogue as epilogue:
                print("Here")
                for middle_opt in ["0", "1", "2", "3"]:
                    for back_opt in [0, 1, 2, 3]:
                        print("Here")
                        # We clone, as we are going to modify the code in the next steps
                        code = llvm_ir.clone()
                        jit.optimize(code, arch, middle_opt, back_opt)
                        mem_buffer = jit.codegen_object(code, arch)
                        with DeviceModule.from_MemBuffer(mem_buffer) as DeviceObj:
                            print("Here")
                            device_func = DeviceObj.get_function(
                                kernel_descr.kernel_name
                            )
                            time = device_func.profile(
                                kernel_descr.grid_dim,
                                kernel_descr.block_dim,
                                prologue._state,
                                kernel_descr.shared_mem,
                            )
                            print(f"Average Execution time is {sum(time)/len(time)}")


if __name__ == "__main__":
    sys.exit(main())
