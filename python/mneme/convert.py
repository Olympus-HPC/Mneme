from mneme.recorded_execution import RecordedExecution
from mneme.mneme_types import ExperimentConfiguration
from mneme.llvm import module
from mneme.mneme_logging import logger
from mneme.proteus.jit import get_proteus_codegen_method
from typing import List, Iterable
from pathlib import Path
import json


def _attribute_line(params: Iterable[int]) -> str:
    # e.g. __attribute__((annotate("jit", 16,17,...)))
    parts = ",".join(str(p) for p in params)
    return f'__attribute__((annotate("jit", {parts})))'


def convert_to_json(
    filename: str,
    recorded_execution: RecordedExecution,
    kernel_descr: RecordedExecution.KernelInstance,
    config: ExperimentConfiguration,
):
    data = {}

    mod = None
    src = None
    root = None
    loc = -1
    for ll in recorded_execution.llvm_files:
        with open(ll, "rb") as fd:
            ir = fd.read()
            mod = module.parse_bitcode(ir)
            try:
                Func = mod.get_function(kernel_descr.kernel_name)
            except NameError:
                logger.debug(
                    f"Could not find function in {kernel_descr.kernel_name} in file {ll}"
                )
                continue

            root, src, loc = Func.get_function_location()
            break

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

    fsrc = ""
    if src is not None and root is not None:
        fsrc = (Path(root) / Path(src)).resolve()

    with open(filename, "w") as fd:
        json.dump(data, fd, indent=2)

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
