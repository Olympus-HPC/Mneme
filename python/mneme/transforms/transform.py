from ctypes import c_char_p

from ..llvm import ffi as ffi
from ..llvm.common import _encode_string
from ..llvm.module import ModuleRef

ffi.lib.TransformPy_RemoveAutoInitMemset.argtypes = [ffi.LLVMModuleRef]


def remove_auto_initialize(mod: ModuleRef) -> ModuleRef:
    ffi.lib.TransformPy_RemoveAutoInitMemset(mod)
    return mod
