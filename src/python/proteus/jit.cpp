#include "../llvm/core.h"
#include "llvm-c/Core.h"
#include <iostream>
#include <proteus/CoreLLVM.hpp>
#include <proteus/CoreLLVMDevice.hpp>

using namespace proteus;

extern "C" {
API_EXPORT(void) ProteusPY_pruneIR(LLVMModuleRef Mod) {
  pruneIR(*llvm::unwrap(Mod));
}

API_EXPORT(void)
ProteusPY_internalize(LLVMModuleRef Mod, const char *KernelSym) {
  std::cout << "Internalizing kernel " << KernelSym << "\n";
  auto *M = llvm::unwrap(Mod);
  internalize(*M, KernelSym);
}

API_EXPORT(void)
ProteusPY_optimize(LLVMModuleRef Mod, const char *DeviceArch,
                   const char OptLevel, unsigned CodegenOptLevel) {
  std::cout << "Received DeviceArch of " << DeviceArch << " OptLevel "
            << OptLevel << " CodegenOptLevel: " << CodegenOptLevel << "\n";
  auto *M = llvm::unwrap(Mod);
  optimizeIR(*M, DeviceArch, OptLevel, CodegenOptLevel);
}

API_EXPORT(LLVMMemoryBufferRef)
ProteusPY_codeGenObject(LLVMModuleRef Mod, const char *DeviceArch) {
  llvm::SmallPtrSet<void *, 8> GlobalLinkedBinaries;
  auto *M = llvm::unwrap(Mod);
  auto DeviceObject =
      proteus::codegenObject(*M, DeviceArch, GlobalLinkedBinaries);
  if (!DeviceObject)
    return nullptr;
  auto *ptr = DeviceObject.release();
  return wrap(ptr);
}
}
