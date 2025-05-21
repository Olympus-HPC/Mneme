from ctypes import POINTER, c_bool, c_char, c_char_p, c_int, c_uint, c_uint64, c_void_p
from typing import List

from ..llvm import ffi as ffi
from ..llvm.buffer import MemBufferRef
from ..llvm.common import _decode_string, _encode_string
from ..llvm.context import get_global_context
from ..llvm.module import ModuleRef
from ..mneme_types import dim3

ffi.lib.ProteusPY_pruneIR.argtypes = [ffi.LLVMModuleRef]
ffi.lib.ProteusPY_optimize.argtypes = [ffi.LLVMModuleRef, c_char_p, c_char, c_uint]
ffi.lib.ProteusPY_internalize.argtypes = [ffi.LLVMModuleRef, c_char_p]
ffi.lib.ProteusPY_codeGenObject.argtypes = [ffi.LLVMModuleRef, c_char_p, c_bool]
ffi.lib.ProteusPY_codeGenObject.restype = ffi.LLVMMemBufferRef
ffi.lib.ProteusPY_linkModules.argtypes = [POINTER(c_char_p), c_int, ffi.LLVMContextRef]
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
    if not isinstance(mod, ModuleRef):
        raise TypeError(f"Expecting type of ModuleRef instead got {type(mod)}")
    ffi.lib.ProteusPY_pruneIR(mod)


def optimize(mod: ModuleRef, device_arch: str, opt_level: str, codegen_opt_level: int):
    valid_vals = {"0", "1", "2", "3", "s", "z"}
    if not isinstance(mod, ModuleRef):
        raise TypeError(f"Expecting type of ModuleRef instead got {type(mod)}")
    if len(opt_level) != 1:
        raise ValueError(
            f"Expected the opt_level to be of a single character '{valid_vals}' but got {opt_level}"
        )
    if opt_level not in valid_vals:
        raise ValueError(
            f"Expected the opt_level to be one of '{valid_vals}' but got {opt_level}"
        )
    if not (codegen_opt_level >= 0 and codegen_opt_level <= 3):
        raise ValueError(
            f"Expected the codegen_opt_level to be between 0-3 instead got {codegen_opt_level}"
        )
    ffi.lib.ProteusPY_optimize(
        mod,
        _encode_string(device_arch),
        opt_level.encode("utf-8")[0],
        int(codegen_opt_level),
    )


def internalize(mod: ModuleRef, kernel_name: str):
    if not isinstance(mod, ModuleRef):
        raise TypeError(f"Expecting type of ModuleRef instead got {type(mod)}")

    ffi.lib.ProteusPY_internalize(mod, _encode_string(kernel_name))


def codegen_object(mod: ModuleRef, device_arch, use_rtc=False):
    if not isinstance(mod, ModuleRef):
        raise TypeError(f"Expecting type of ModuleRef instead got {type(mod)}")
    result = MemBufferRef(
        ffi.lib.ProteusPY_codeGenObject(mod, _encode_string(device_arch), use_rtc)
    )
    return result


def link_llvm_modules(modules: List[str]):
    c_strings = [c_char_p(s.encode("utf-8")) for s in modules]
    ArrayType = c_char_p * len(c_strings)
    c_array = ArrayType(*c_strings)
    Mod = ModuleRef(
        ffi.lib.ProteusPY_linkModules(c_array, len(modules), get_global_context()),
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
