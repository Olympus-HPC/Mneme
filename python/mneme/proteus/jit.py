import typing
from ../llvm/module.py import ModuleRef
from ../llvm/ import ffi as ffi

ffi.lib.ProteusPY_pruneIR.argtypes = [ffi.LLVMModuleRef]

def pruneIR(mod: ModuleRef): 
    if not isinstance(mod, ModuleRef):
        raise TypeError(f"Expecting type of ModuleRef instead got {type(mod)}")
    ffi.lib.ProteusPY_pruneIR(mod)


