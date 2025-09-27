from mneme.llvm.common import _encode_string

from . import ffi, module


def get_source_and_line(mod: module.ModuleRef, fn_name: str):
    Func = mod.get_function(fn_name)
    print(type(Func))
    return mod.source_file, Func.get_function_location()
