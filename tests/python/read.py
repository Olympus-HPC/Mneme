import mneme
import mneme.llvm as llvm
import mneme.proteus.jit as jit
import sys

module_bc = sys.argv[1]

with open(module_bc, "rb") as fd:
    bitcode = fd.read()

Mod = llvm.module.parse_bitcode(bitcode)

for i, func in enumerate(Mod.functions):
    print(i, func.name)

print(jit.pruneIR)
jit.pruneIR(Mod)

print("After")
for i, func in enumerate(Mod.functions):
    print(i, func.name)

jit.optimize(Mod, "gfx90a", "3", 1)
