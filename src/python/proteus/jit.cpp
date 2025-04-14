#include "../llvm/core.h"
#include "llvm-c/Core.h"
#include "llvm/IR/Module.h"
#include <iostream>
#include <llvm-c/Types.h>
#include <llvm/IR/Module.h>
#include <proteus/CoreLLVM.hpp>
#include <proteus/CoreLLVMDevice.hpp>

using namespace proteus;

extern "C" {
API_EXPORT(void) ProteusPY_pruneIR(LLVMModuleRef Mod) {
  pruneIR(*llvm::unwrap(Mod));
}

API_EXPORT(void)
ProteusPY_internalize(LLVMModuleRef Mod, const char *KernelSym) {
  auto *M = llvm::unwrap(Mod);
  internalize(*M, KernelSym);
}

API_EXPORT(void)
ProteusPY_optimize(LLVMModuleRef Mod, const char *DeviceArch,
                   const char OptLevel, unsigned CodegenOptLevel) {
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

API_EXPORT(LLVMModuleRef)
ProteusPY_linkIRFiles(void **Modules, int size, LLVMContextRef context) {
  SmallVector<std::unique_ptr<llvm::Module>> LLVMMods;
  for (int i = 0; i < size; i++) {
    LLVMMods.push_back(std::unique_ptr<llvm::Module>((Module *)(Modules[i])));
  }

  auto Mod = proteus::linkModules(*unwrap(context), LLVMMods);

  // Python is the owner, so we can just release here
  for (auto &M : LLVMMods) {
    auto *ptr = M.release();
  }

  return wrap(Mod.release());
}
}
