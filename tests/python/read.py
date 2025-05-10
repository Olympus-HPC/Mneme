import mneme
import mneme.llvm as llvm
import mneme.llvm.buffer as MemBuffer
import mneme.proteus.jit as jit
from mneme.device import DeviceModule, dim3
import sys

module_bc = sys.argv[1]

with open(module_bc, "rb") as fd:
    bitcode = fd.read()

Mod = llvm.module.parse_bitcode(bitcode)
jit.pruneIR(Mod)
jit.internalize(Mod, "foo")
jit.optimize(Mod, "gfx90a", "3", 1)
mem_buffer = jit.codegen_object(Mod, "gfx90a")

with DeviceModule.from_MemBuffer(mem_buffer) as VendorModule:
    print("Module loaded")
    StaticFunc = VendorModule.get_function("foo")
    StaticFunc.launch(dim3(1, 1, 1), dim3(1, 1, 1))
StaticFunc.launch(dim3(1, 1, 1), dim3(1, 1, 1))
