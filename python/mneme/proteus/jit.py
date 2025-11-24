from ctypes import POINTER, c_bool, c_char, c_char_p, c_int, c_uint, c_uint64, c_void_p
from typing import List

from ..llvm import ffi as ffi
from ..llvm.buffer import MemBufferRef
from ..llvm.common import _decode_string, _encode_string
from ..llvm.context import get_global_context
from ..llvm.module import ModuleRef
from ..mneme_types import dim3

ffi.lib.ProteusPY_pruneIR.argtypes = [ffi.LLVMModuleRef]
ffi.lib.ProteusPY_optimize.argtypes = [ffi.LLVMModuleRef, c_char_p, c_char_p, c_uint]
ffi.lib.ProteusPY_internalize.argtypes = [ffi.LLVMModuleRef, c_char_p]
ffi.lib.ProteusPY_codeGenObject.argtypes = [
    ffi.LLVMModuleRef,
    c_char_p,
    c_char_p,
    c_uint,
]
ffi.lib.ProteusPY_codeGenObject.restype = ffi.LLVMMemBufferRef
ffi.lib.ProteusPY_linkModules.argtypes = [
    POINTER(c_char_p),
    c_int,
    ffi.LLVMContextRef,
    c_char_p,
    c_bool,
    c_bool,
]
ffi.lib.ProteusPY_linkModules.restype = ffi.LLVMModuleRef
ffi.lib.ProteusPY_specializeArguments.argtypes = [
    ffi.LLVMModuleRef,  # Module
    c_uint64,  # Hash
    c_char_p,  # KernelName
    POINTER(c_void_p),  # KernelArguments
    c_int,  # Number of Arguments
    POINTER(c_int),  # Indexes to specialize
    c_int,  # num Indexes
]
ffi.lib.ProteusPY_specializeArguments.restype = c_uint64

ffi.lib.ProteusPY_specializeDims.argtypes = [
    ffi.LLVMModuleRef,
    c_uint64,
    c_char_p,
    dim3,
    dim3,
]
ffi.lib.ProteusPY_specializeDims.restype = c_uint64

ffi.lib.ProteusPY_setLaunchBounds.argtypes = [
    ffi.LLVMModuleRef,
    c_uint64,
    c_char_p,
    c_int,
    c_int,
]

ffi.lib.ProteusPY_setLaunchBounds.restype = c_uint64


def pruneIR(mod: ModuleRef):
    """
    @brief Remove unused functions, globals, and IR constructs from an LLVM module.

    This calls Proteus' C++ pruning pass through the FFI to eliminate dead IR and
    reduce module size before further specialization or optimization.

    @param mod LLVM module to prune.
    @throws TypeError If `mod` is not a ModuleRef instance.
    """
    if not isinstance(mod, ModuleRef):
        raise TypeError(f"Expecting type of ModuleRef instead got {type(mod)}")
    ffi.lib.ProteusPY_pruneIR(mod)


def optimize(mod: ModuleRef, device_arch: str, opt_level: str, codegen_opt_level: int):
    """
    @brief Run Proteus optimization passes on an LLVM module.

    Applies middle-end optimization passes customized for a target device
    architecture and a chosen LLVM optimization level. Also configures
    code-generation optimization intensity.

    @param mod LLVM module to optimize.
    @param device_arch Device architecture string (e.g., "gfx942").
    @param opt_level LLVM optimization level ("O1", "O2", "O3", "Os", "Oz").
    @param codegen_opt_level Codegen optimization level in [0,3].

    @throws TypeError If `mod` is not a ModuleRef.
    @throws ValueError If codegen_opt_level is outside [0,3].
    """
    if not isinstance(mod, ModuleRef):
        raise TypeError(f"Expecting type of ModuleRef instead got {type(mod)}")

    if not (codegen_opt_level >= 0 and codegen_opt_level <= 3):
        raise ValueError(
            f"Expected the codegen_opt_level to be between 0-3 instead got {codegen_opt_level}"
        )
    if len(opt_level) == 0:
        return

    ffi.lib.ProteusPY_optimize(
        mod,
        _encode_string(device_arch),
        _encode_string(opt_level),
        int(codegen_opt_level),
    )


def internalize(mod: ModuleRef, kernel_name: str):
    """
    @brief Mark all symbols except the given kernel as internal.

    This applies Proteus' internalization pass, restricting symbol visibility
    to reduce linking overhead and enable more aggressive optimization.

    @param mod LLVM module to update.
    @param kernel_name Name of the kernel whose symbol must remain public.

    @throws TypeError If `mod` is not a ModuleRef.
    """
    if not isinstance(mod, ModuleRef):
        raise TypeError(f"Expecting type of ModuleRef instead got {type(mod)}")

    ffi.lib.ProteusPY_internalize(mod, _encode_string(kernel_name))


def codegen_object(
    mod: ModuleRef, device_arch, codegen_type="serial", codegen_opt_level: int = 3
):
    """
    @brief Generate a compiled device object (ELF/HSACO/etc.) from an LLVM module.

    Invokes Proteus' backend code generator for the given architecture and
    returns the compiled binary wrapped in a MemBufferRef.

    @param mod LLVM module to compile.
    @param device_arch Target architecture string.
    @param codegen_type Codegen mode ("serial", "parallel", etc.).
    @param codegen_opt_level Backend optimization level in [1,3].

    @return MemBufferRef containing the produced code object.
    @throws TypeError If `mod` is not a ModuleRef.
    @throws RuntimeError If `codegen_opt_level` is outside (0,3].
    """
    if not isinstance(mod, ModuleRef):
        raise TypeError(f"Expecting type of ModuleRef instead got {type(mod)}")

    if codegen_opt_level < 1 or codegen_opt_level > 3:
        raise RuntimeError(
            f"codegen optimization level must be in range (0,3], instead it was {codegen_opt_level}"
        )
    result = MemBufferRef(
        ffi.lib.ProteusPY_codeGenObject(
            mod,
            _encode_string(device_arch),
            _encode_string(codegen_type),
            codegen_opt_level,
        )
    )
    return result


def link_llvm_modules(
    modules: List[str], kernel_name: str, prune: bool, internalize: bool
):
    """
    @brief Link multiple LLVM IR files into a single unified module.

    This constructs a new module by invoking Proteus' linker. Optionally
    performs pruning and internalization during the link stage.

    @param modules List of filesystem paths to LLVM IR modules.
    @param kernel_name Name of the kernel entry function to preserve.
    @param prune Whether to prune dead IR after linking.
    @param internalize Whether to internalize symbols except the kernel.
    @return A new linked ModuleRef.
    """
    c_strings = [c_char_p(s.encode("utf-8")) for s in modules]
    ArrayType = c_char_p * len(c_strings)
    c_array = ArrayType(*c_strings)
    Mod = ModuleRef(
        ffi.lib.ProteusPY_linkModules(
            c_array,
            len(modules),
            get_global_context(),
            kernel_name.encode("utf-8"),
            prune,
            internalize,
        ),
        get_global_context(),
    )
    return Mod


def specialize_args(
    mod: ModuleRef,
    mod_hash: int,
    kernel_name: str,
    kernel_args,
    num_args: int,
    specialize_indexes,
):
    """
    @brief Specialize a subset of kernel arguments inside an LLVM module.

    Performs constant propagation and IR rewriting based on the provided
    runtime arguments, updating the module hash to reflect specialization.

    @param mod LLVM module to modify.
    @param mod_hash Current module hash before specialization.
    @param kernel_name Name of the kernel whose arguments are specialized.
    @param kernel_args Raw pointers to argument values (FFI-compatible).
    @param num_args Number of kernel arguments.
    @param specialize_indexes Indices of arguments to specialize.

    @return New module hash after specialization.
    @throws RuntimeError If more indexes are specialized than available args.
    """
    if num_args < len(specialize_indexes):
        raise RuntimeError("Trying to specialize more indexes than available")

    indexes = (c_int * len(specialize_indexes))()
    for i, v in enumerate(specialize_indexes):
        indexes[i] = v

    return int(
        ffi.lib.ProteusPY_specializeArguments(
            mod,
            c_uint64(mod_hash),
            _encode_string(kernel_name),
            kernel_args,
            num_args,
            indexes,
            len(specialize_indexes),
        )
    )


def specialize_dims(
    mod: ModuleRef, mod_hash: int, kernel_name: str, grid_dim: dim3, block_dim: dim3
):
    """
    @brief Specialize launch dimensions (grid/block) inside the LLVM module.

    Embeds compile-time constants for launch configuration, enabling more
    aggressive loop unrolling and simplification.

    @param mod LLVM module to update.
    @param mod_hash Previous module hash.
    @param kernel_name Kernel to specialize.
    @param grid_dim Grid dimensions (dim3).
    @param block_dim Block dimensions (dim3).

    @return Updated module hash.
    """
    return int(
        ffi.lib.ProteusPY_specializeDims(
            mod, c_uint64(mod_hash), _encode_string(kernel_name), grid_dim, block_dim
        )
    )


def set_launch_bounds(
    mod: ModuleRef,
    mod_hash: int,
    kernel_name: str,
    max_threads_per_block: int,
    min_blocks_per_sm: int,
):
    """
    @brief Apply CUDA/HIP-style launch bounds metadata for the kernel.

    Sets the `__launch_bounds__` metadata on the kernel to restrict maximum
    threads per block and required occupancy, influencing register allocation
    and performance.

    @param mod LLVM module to annotate.
    @param mod_hash Current module hash.
    @param kernel_name Name of the kernel function.
    @param max_threads_per_block Maximum threads-per-block bound (≤ 1024).
    @param min_blocks_per_sm Minimum required blocks per SM.

    @return Updated module hash.
    @throws RuntimeError If `max_threads_per_block` exceeds 1024.
    """
    if max_threads_per_block > 1024:
        raise RuntimeError("Max threads cannot be larger than 1024")

    return int(
        ffi.lib.ProteusPY_setLaunchBounds(
            mod,
            c_uint64(mod_hash),
            _encode_string(kernel_name),
            max_threads_per_block,
            min_blocks_per_sm,
        )
    )
