#include "llvm-c/Core.h"
#include <proteus/CoreLLVM.hpp>
#include <proteus/CoreLLVMDevice.hpp>

using namespace proteus;

extern "C" {
void ProteusPY_pruneIR(LLVMModuleRef Mod) { pruneIR(*llvm::unwrap(Mod)); }

void ProteusPY_internalize(LLVMModuleRef Mod, const char *KernelSym) {
  auto *M = llvm::unwrap(Mod);
  internalize(*M, KernelSym);
}

void ProteusPY_optimize(LLVMModuleRef Mod, const char *DeviceArch,
                        const char OptLevel, unsigned CodegenOptLevel) {
  auto *M = llvm::unwrap(Mod);
  optimizeIR(*M, DeviceArch, OptLevel, CodegenOptLevel);
}
}
