import typing
from ..llvm.module import ModuleRef
from ..llvm import ffi as ffi
from ..llvm.common import _decode_string, _encode_string
from ctypes import (
    POINTER,
    byref,
    cast,
    c_char_p,
    c_char,
    c_double,
    c_int,
    c_int64,
    c_size_t,
    c_uint,
    c_uint8,
    c_uint64,
    c_bool,
    c_void_p,
)

ffi.lib.ProteusPY_pruneIR.argtypes = [ffi.LLVMModuleRef]
ffi.lib.ProteusPY_optimize.argtypes = [ffi.LLVMModuleRef, c_char_p, c_char, c_uint]
ffi.lib.ProteusPY_internalize.argtypes = [ffi.LLVMModuleRef, c_char_p]


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
