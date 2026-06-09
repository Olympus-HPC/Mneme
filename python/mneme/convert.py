from mneme.recorded_execution import RecordedExecution
from mneme.mneme_types import ExperimentConfiguration
from mneme.llvm import module
from mneme.mneme_logging import logger
from mneme.proteus.jit import get_proteus_codegen_method
from typing import Iterable, Optional, Tuple
from pathlib import Path
import json


def _attribute_line(params: Iterable[int]) -> str:
    # e.g. __attribute__((annotate("jit", 16,17,...)))
    parts = ",".join(str(p) for p in params)
    return f'__attribute__((annotate("jit", {parts})))'


def _find_kernel_location(
    recorded_execution: RecordedExecution,
    kernel_descr: RecordedExecution.KernelInstance,
) -> Tuple[str, int]:
    """ Try to find the source file and line number of the kernel function.
        If not found, return empty string and -1.
    """

    fsrc = ""
    loc = -1
    for ll in recorded_execution.llvm_files:
        with open(ll, "rb") as fd:
            ir = fd.read()
            mod = module.parse_bitcode(ir)
            try:
                func = mod.get_function(kernel_descr.kernel_name)
            except NameError:
                logger.debug(
                    f"Could not find function in {kernel_descr.kernel_name} in file {ll}"
                )
                continue

            root, src, loc = func.get_function_location()
            if src is not None and root is not None:
                fsrc = str((Path(root) / Path(src)).resolve())
            break
    return fsrc, loc


def _proteus_payload(
    kernel_descr: RecordedExecution.KernelInstance,
    config: ExperimentConfiguration,
) -> dict:
    """ Based on a kernel config (eg best found config after tuning) create a 
        payload that proteus can consume to run the kernel.
    """
    data = {}

    res = {}
    res["CodeGen"] = get_proteus_codegen_method()
    res["LaunchBounds"] = config.set_launch_bounds
    res["SpecializeDims"] = config.specialize_dims
    res["SpecializeDimsRange"] = config.specialize_dims
    res["OptLevel"] = 1
    res["Pipeline"] = config.passes
    res["CodeGenOptLevel"] = config.codegen_opt
    res["TunedMaxThreads"] = config.max_threads
    res["MinBlocksPerSM"] = config.min_blocks_per_sm
    data[kernel_descr.kernel_name] = res

    return data


def _usage_text(
    filename: str,
    kernel_descr: RecordedExecution.KernelInstance,
    source_location: str,
    loc: int,
) -> str:
    """ build a message on how to use the resulting tuned kernel """

    attribute_line = _attribute_line(
        [i + 1 for i, v in enumerate(kernel_descr.specializations) if v]
    )
    location = source_location
    if location and loc != -1:
        location = f"{location}:{loc}"
    elif not location:
        location = "unknown"

    return (
        "To use the optimized kernel version for this kernel, apply this attribute:\n\n"
        f"{attribute_line}\n\n"
        "Kernel source location:\n"
        f"{location}\n\n"
        "Configure the application to build with Proteus, then run:\n\n"
        f"export PROTEUS_TUNED_KERNELS={str(Path(filename).resolve())}\n"
    )


def export_proteus_tuned_kernel(
    filename: str,
    recorded_execution: RecordedExecution,
    kernel_descr: RecordedExecution.KernelInstance,
    config: ExperimentConfiguration,
    *,
    usage_filename: Optional[str] = None,
) -> None:
    """ Export the tuned kernel configuration to a JSON file that can be consumed by Proteus.
         If usage_filename is provided, also write a usage message to that file. Otherwise, print the usage message to stdout.
    """

    data = _proteus_payload(kernel_descr, config)
    fsrc, loc = _find_kernel_location(recorded_execution, kernel_descr)
    with open(filename, "w") as fd:
        json.dump(data, fd, indent=2)

    usage = _usage_text(filename, kernel_descr, fsrc, loc)
    if usage_filename is not None:
        with open(usage_filename, "w") as fd:
            fd.write(usage)
        return

    attribute_line = _attribute_line(
        [i + 1 for i, v in enumerate(kernel_descr.specializations) if v]
    )
    print("\n\n")
    print(
        f"To use the optimized kernel version for the kernel'{recorded_execution.demangled_name}'"
    )
    print("apply this attribute:")
    print(attribute_line)

    if fsrc != "":
        msg = f"The kernel is defined in:\n{fsrc}"
        if loc != -1:
            msg += f":{loc}"
        print(msg)
    print("... and configure your project to build with proteus")
    print(
        f"lastly execute the application by\nexport PROTEUS_TUNED_KERNELS={str(Path(filename).resolve())}"
    )
    print("\n\n")


def convert_to_json(
    filename: str,
    recorded_execution: RecordedExecution,
    kernel_descr: RecordedExecution.KernelInstance,
    config: ExperimentConfiguration,
):
    """ only supports one path right now -- json file that can be consumed by proteus. """
    export_proteus_tuned_kernel(filename, recorded_execution, kernel_descr, config)
